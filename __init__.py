import asyncio
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Union

import httpx
from mem0 import Memory
from nonebot.adapters.onebot.v11 import Bot, Message, MessageEvent
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from pydantic import Field

from nekro_agent.api.core import get_qdrant_config, logger
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core.config import ModelConfigGroup
from nekro_agent.core.config import config as core_config
from nekro_agent.matchers.command import command_guard, finish_with, on_command
from nekro_agent.models.db_chat_channel import DefaultPreset
from nekro_agent.models.db_preset import DBPreset
from nekro_agent.services.agent.creator import OpenAIChatMessage
from nekro_agent.services.agent.openai import gen_openai_chat_response
from nekro_agent.services.message.message_service import message_service
from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin, SandboxMethodType

# 扩展元数据
plugin = NekroPlugin(
    name="记忆模块",
    module_name="nekro_plugin_memory",
    description="长期记忆管理系统,支持记忆的增删改查及语义搜索",
    version="0.1.1",
    author="Zaxpris",
    url="https://github.com/zxjwzn/nekro-plugin-memory",
)


# 在现有import后添加以下代码
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def encode_base62(number: int) -> str:
    """将大整数编码为Base62字符串"""
    if number == 0:
        return BASE62_ALPHABET[0]
    digits = []
    while number > 0:
        number, remainder = divmod(number, 62)
        digits.append(BASE62_ALPHABET[remainder])
    return "".join(reversed(digits))


def decode_base62(encoded: str) -> int:
    """将Base62字符串解码回大整数"""
    number = 0
    for char in encoded:
        number = number * 62 + BASE62_ALPHABET.index(char)
    return number


def encode_id(original_id: str) -> str:
    """将UUID转换为短ID"""
    try:
        uuid_obj = uuid.UUID(original_id)
        return encode_base62(uuid_obj.int)
    except ValueError as err:
        raise ValueError("无效的UUID格式") from err


def decode_id(encoded_id: str) -> str:
    """将短ID转换回原始UUID"""
    try:
        number = decode_base62(encoded_id)
        return str(uuid.UUID(int=number))
    except (ValueError, AttributeError) as err:
        raise ValueError("无效的短ID格式") from err


def format_memories(results: List[Dict], score_threshold: float = 0.0) -> str:
    """格式化记忆列表为字符串"""
    if not results:
        return "未找到任何记忆"

    formatted = []
    for idx, mem in enumerate(results, 1):
        metadata = mem.get("metadata", {})
        created_at = mem.get("created_at", "未知时间")
        score = mem.get("score", "暂无")
        if score != "暂无" and float(score) < score_threshold:
            continue
        formatted.append(
            f"{idx}. [ID: {encode_id(mem['id'])}]\n"  # 使用短ID
            f"内容: {mem['memory']}\n"
            f"元数据: {metadata}\n"
            f"创建时间: {created_at}\n"
            f"匹配度: {score}\n",
        )
    return "\n".join(formatted)


def format_beijing_time(iso_timestamp_str: str) -> str:
    """将 ISO 8601 格式的时间字符串转换为北京时间 (年-月-日 时:分)"""
    if not isinstance(iso_timestamp_str, str):  # 添加类型检查以防万一
        return "未知时间"
    try:
        # 解析 ISO 8601 格式的字符串，自动处理时区信息
        dt = datetime.fromisoformat(
            iso_timestamp_str.replace("Z", "+00:00"),
        )  # 处理 'Z' 时区标识符

        # 定义北京时区 (UTC+8)
        beijing_tz = timezone(timedelta(hours=8))

        # 将时间转换为北京时间
        dt_beijing = dt.astimezone(beijing_tz)

        # 格式化为 "年-月-日 时:分"
        return dt_beijing.strftime("%Y年%m月%d日 %H:%M")
    except (ValueError, TypeError):
        return "未知时间"


def extract_facts_content(content: str) -> str:
    """
    从指定内容中提取 <facts> 和 </facts> 标签之间的内容。

    Args:
        content: 要提取的内容。

    Returns:
        如果找到标签之间的内容，则返回该内容字符串；否则返回 None。
    """
    try:
        match = re.search(r"<facts>(.*?)</facts>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""  # noqa: TRY300
    except Exception:
        return ""


async def get_preset(_ctx: AgentCtx) -> Union[DBPreset, DefaultPreset]:
    from nekro_agent.models.db_chat_channel import DBChatChannel

    db_chat_channel: DBChatChannel = await DBChatChannel.get_channel(
        chat_key=_ctx.from_chat_key,
    )
    preset = await db_chat_channel.get_preset()
    logger.info(f"当前人设{preset.name}")
    return preset


# 根据模型名获取模型组配置项
def get_model_group_info(model_name: str) -> ModelConfigGroup:
    try:
        return core_config.MODEL_GROUPS[model_name]
    except KeyError as e:
        raise ValueError(f"模型组 '{model_name}' 不存在，请确认配置正确") from e


@plugin.mount_config()
class MemoryConfig(ConfigBase):
    """基础配置"""

    MEMORY_MANAGE_MODEL: str = Field(
        default="default",
        title="记忆管理模型",
        description="用于将传入的记忆内容简化整理的对话模型组",
        json_schema_extra={"ref_model_groups": True, "required": True},
    )
    TEXT_EMBEDDING_MODEL: str = Field(
        default="default",
        title="向量嵌入模型",
        description="用于将传入的记忆进行向量嵌入的嵌入模型组",
        json_schema_extra={"ref_model_groups": True, "required": True},
    )
    TEXT_EMBEDDING_DIMENSION: int = Field(
        default=1024,
        title="嵌入维度",
        description="嵌入维度",
    )
    MEMORY_SEARCH_SCORE_THRESHOLD: float = Field(
        default=0.0,
        title="记忆匹配度阈值",
        description="搜索记忆时，匹配度低于该值的记忆将被过滤掉，取值范围0-1",
    )
    SESSION_ISOLATION: bool = Field(
        default=True,
        title="记忆会话隔离",
        description="开启后bot存储的记忆只对当前会话有效,在其他会话中无法获取",
    )
    PRESET_ISOLATION: bool = Field(
        default=True,
        title="人设会话隔离",
        description="开启后会根据当前人设存储记忆,更换到其他人设时无法访问",
    )
    AUTO_MEMORY_CONTEXT_MESSAGE_COUNT: int = Field(
        default=5,
        title="上下文消息数",
        description="可获取到的上下文消息数量",
    )


# 获取最新配置的函数
def get_memory_config() -> MemoryConfig:
    """获取最新的记忆模块配置"""
    return plugin.get_config(MemoryConfig)


_mem0_instance = None
_last_config_hash = None
_thread_pool = None  # 创建一个线程池用于执行同步操作

# 添加记忆注入缓存，避免短时间内重复执行
_memory_inject_cache = {}


@plugin.mount_init_method()
async def init_memory():
    global _mem0_instance, _last_config_hash, _thread_pool, _memory_inject_cache
    # 初始化mem0客户端
    _mem0_instance = None
    _last_config_hash = None
    _thread_pool = ThreadPoolExecutor(max_workers=5)  # 创建一个线程池用于执行同步操作

    # 添加记忆注入缓存，避免短时间内重复执行
    _memory_inject_cache = {}


# 异步创建Memory实例
async def create_memory_async(config: Dict[str, Any]) -> Memory:
    """将Memory.from_config包装成异步函数，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: Memory.from_config(config),
    )


# 将同步方法包装成异步方法
async def async_mem0_search(mem0, query: str, user_id: str, limit: int | None = None):
    """异步执行mem0.search，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: mem0.search(query=query, user_id=user_id, limit=limit),
    )


async def async_mem0_get_all(mem0, user_id: str, limit: int | None = None):
    """异步执行mem0.get_all，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: mem0.get_all(user_id=user_id, limit=limit),
    )


async def async_mem0_add(mem0, messages: str, user_id: str, metadata: Dict[str, Any]):
    """异步执行mem0.add，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: mem0.add(messages=messages, user_id=user_id, metadata=metadata),
    )


async def async_mem0_update(mem0, memory_id: str, data: str):
    """异步执行mem0.update，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: mem0.update(memory_id=memory_id, data=data),
    )


async def async_mem0_history(mem0, memory_id: str):
    """异步执行mem0.history，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: mem0.history(memory_id=memory_id),
    )


async def async_mem0_delete(mem0, memory_id: str):
    """异步执行mem0.delete，避免阻塞事件循环"""
    return await asyncio.get_running_loop().run_in_executor(
        _thread_pool,
        lambda: mem0.delete(memory_id=memory_id),
    )


async def get_mem0_client_async(_ctx: AgentCtx):
    """异步获取mem0客户端实例"""
    global _mem0_instance, _last_config_hash
    memory_config = get_memory_config()  # 始终获取最新配置
    qdrant_config = get_qdrant_config()
    # 获取当前人设
    preset = await get_preset(_ctx)
    # 计算当前配置的哈希值
    current_config = {
        "MEMORY_MANAGE_MODEL": memory_config.MEMORY_MANAGE_MODEL,
        "TEXT_EMBEDDING_MODEL": memory_config.TEXT_EMBEDDING_MODEL,
        "TEXT_EMBEDDING_DIMENSION": memory_config.TEXT_EMBEDDING_DIMENSION,
        "llm_model_name": get_model_group_info(
            memory_config.MEMORY_MANAGE_MODEL,
        ).CHAT_MODEL,
        "PRESET_ISOLATION": memory_config.PRESET_ISOLATION,
        "llm_api_key": get_model_group_info(memory_config.MEMORY_MANAGE_MODEL).API_KEY,
        "llm_base_url": get_model_group_info(
            memory_config.MEMORY_MANAGE_MODEL,
        ).BASE_URL,
        "embedder_model_name": get_model_group_info(
            memory_config.TEXT_EMBEDDING_MODEL,
        ).CHAT_MODEL,
        "embedder_api_key": get_model_group_info(
            memory_config.TEXT_EMBEDDING_MODEL,
        ).API_KEY,
        "embedder_base_url": get_model_group_info(
            memory_config.TEXT_EMBEDDING_MODEL,
        ).BASE_URL,
        "qdrant_url": qdrant_config.url,
        "qdrant_api_key": qdrant_config.api_key,
        "preset_name": preset.name,
    }

    # 验证字段不能为空字符串
    errors = []

    if not current_config["llm_model_name"]:
        errors.append(
            f"模型组 '{memory_config.MEMORY_MANAGE_MODEL}' 的CHAT_MODEL不能为空",
        )
    if not current_config["llm_api_key"]:
        errors.append(f"模型组 '{memory_config.MEMORY_MANAGE_MODEL}' 的API_KEY不能为空")
    if not current_config["llm_base_url"]:
        errors.append(
            f"模型组 '{memory_config.MEMORY_MANAGE_MODEL}' 的BASE_URL不能为空",
        )
    if not current_config["embedder_model_name"]:
        errors.append(
            f"模型组 '{memory_config.TEXT_EMBEDDING_MODEL}' 的CHAT_MODEL不能为空",
        )
    if not current_config["embedder_api_key"]:
        errors.append(
            f"模型组 '{memory_config.TEXT_EMBEDDING_MODEL}' 的API_KEY不能为空",
        )
    if not current_config["embedder_base_url"]:
        errors.append(
            f"模型组 '{memory_config.TEXT_EMBEDDING_MODEL}' 的BASE_URL不能为空",
        )

    if errors:
        error_message = "记忆模块配置错误：\n" + "\n".join(
            [f"- {error}" for error in errors],
        )
        logger.error(error_message)
        raise ValueError(error_message)

    current_hash = hash(frozenset(current_config.items()))

    # 如果配置变了或者实例不存在，重新初始化
    if _mem0_instance is None or current_hash != _last_config_hash:
        # 重新构建配置
        mem0_client_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "url": current_config["qdrant_url"],
                    "api_key": current_config["qdrant_api_key"],
                    "collection_name": current_config["preset_name"],
                    "embedding_model_dims": current_config["TEXT_EMBEDDING_DIMENSION"],
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": current_config["llm_api_key"],
                    "model": current_config["llm_model_name"],
                    "openai_base_url": current_config["llm_base_url"],
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": current_config["embedder_api_key"],
                    "model": current_config["embedder_model_name"],
                    "openai_base_url": current_config["embedder_base_url"],
                    "embedding_dims": current_config["TEXT_EMBEDDING_DIMENSION"],
                },
            },
            "version": "v1.1",
        }

        # 异步创建新实例
        _mem0_instance = await create_memory_async(mem0_client_config)
        _last_config_hash = current_hash
        logger.info("记忆管理器已重新初始化")

    return _mem0_instance


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="记忆模块使用提示",
    description="这是有关记忆模块的提示 Do Not Call This Function!!!",
)
async def add_memory_notice(_ctx: AgentCtx):
    """
    This is a prompt regarding the memory module. Do Not Call This Function!!!
    ⚠️ Critical Attention：
    - When utilizing the memory module for operations such as memory storage and retrieval (e.g., add_memory, search_memory), these operations should, as a best practice, be processed at the end of the code, particularly before functions like send_msg_text or send_msg_file.
    - The user_id must strictly and accurately identify the owner of the memory. Fields within metadata cannot serve as a substitute for the user_id.
    - If the memory content to be stored includes temporal information, the use of relative time expressions (e.g., "yesterday," "the day before yesterday," "later") is prohibited. Absolute and specific date and time formats (e.g., YYYY-MM-DD HH:MM) must be employed.
    - For virtual characters, their full names must be rendered in lowercase English, for instance, ("hatsune_miku", "takanashi_hoshino").
    - If the memory content pertains to a user participating in the current dialogue, the user_id for storage must be the ID of that specific user. For example, if a user with QQ ID "123456" states, "My nickname is Bob," then the user_id shall be "123456", and the memory content shall be "Nickname is Bob."
    - If the memory content pertains to a third party, the user_id for storage must be the ID of that third party. For example, if a user with QQ ID "123456" states, "@114514 likes swimming," then the user_id shall be "114514", and the memory content shall be "likes swimming."
    """
    return ""


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="添加记忆",
    description="指定用户id并添加长期记忆",
)
async def add_memory(
    _ctx: AgentCtx,
    memory: str,
    user_id: str,
    metadata: Dict[str, Any],
) -> str:
    """Adds a new memory to the user's profile.

    Args:
        memory (str): The text content of the memory to add.
        user_id (str): The associated user ID. This should be the user's QQ number, for example, 2708583339, not a chat_key. If an empty string is passed, it means querying memories related to oneself.
        metadata (Dict[str, Any]): Metadata tags, e.g., {"category": "hobbies"}.

    Returns:
        str: The memory ID.

    Example:
        add_memory("Likes to play cricket on weekends", "114514", {"category": "hobbies", "sport_type": "cricket"})
        add_memory("Likes to eat pizza", "123456", {"category": "hobbies", "food_type": "pizza"})
        add_memory("Likes to play CSGO", "114514", {"category": "hobbies", "game_type": "csgo"})
        add_memory("Nickname is Miaomiao", "123456", {"category": "name", "nickname": "Miaomiao"})
    """
    memory_config = get_memory_config()
    mem0 = await get_mem0_client_async(_ctx)
    if user_id == "":
        if memory_config.PRESET_ISOLATION:
            preset = await get_preset(_ctx)
            user_id = preset.name
        else:
            user_id = core_config.BOT_QQ

    if memory_config.SESSION_ISOLATION:
        user_id = _ctx.from_chat_key + user_id

    user_id = user_id.replace(" ", "_")

    try:
        result = await async_mem0_add(
            mem0, messages=memory, user_id=user_id, metadata=metadata,
        )
        logger.info(f"添加记忆结果: {result}")
        if result.get("results"):
            memory_id = result["results"][0]["id"]
            short_id = encode_id(memory_id)  # 添加编码
            return f"记忆添加成功,ID：{short_id}"
        return ""  # noqa: TRY300
    except httpx.HTTPError as e:
        logger.error(f"网络请求失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"添加记忆失败: {e!s}")
        raise RuntimeError(f"记忆添加失败: {e!s}") from e


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="搜索记忆",
    description="通过模糊描述对有关记忆进行搜索",
)
async def search_memory(_ctx: AgentCtx, query: str, user_id: str) -> str:
    """Searches memories.

    Args:
        query (str): The text content of the memory to search for. It can be a question, for example, "What do I like to eat?", "When is my birthday?".
        user_id (str): The unique identifier of the user to query. This should be the user's QQ number, for example, 123456, not a chat_key. If an empty string is passed, it means querying memories related to oneself.

    Examples:
        search_memory("What was eaten on March 1, 2025", "123456")
    """
    memory_config = get_memory_config()
    mem0 = await get_mem0_client_async(_ctx)
    if user_id == "":
        if memory_config.PRESET_ISOLATION:
            preset = await get_preset(_ctx)
            user_id = preset.name
        else:
            user_id = core_config.BOT_QQ

    if memory_config.SESSION_ISOLATION:
        user_id = _ctx.from_chat_key + user_id

    user_id = user_id.replace(" ", "_")

    try:
        result = await async_mem0_search(mem0, query=query, user_id=user_id)
        logger.info(f"搜索记忆结果: {result}")
        return "以下是你对该用户的记忆:\n" + format_memories(result.get("results", []), memory_config.MEMORY_SEARCH_SCORE_THRESHOLD)
    except httpx.HTTPError as e:
        logger.error(f"网络请求失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"搜索记忆失败: {e!s}")
        raise RuntimeError(f"搜索记忆失败: {e!s}") from e


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="获取记忆",
    description="获取有关该用户的所有记忆",
)
async def get_all_memories(_ctx: AgentCtx, user_id: str) -> str:
    """Gets all memories for a user.

    Args:
        user_id (str): The unique identifier of the user to query. This should be the user's QQ number, for example, 123456, not a chat_key. If an empty string is passed, it means querying memories related to oneself.

    Returns:
        str: A formatted string list of memories, including memory content and metadata.

    Example:
        get_all_memories("123456")
    """
    memory_config = get_memory_config()  # 获取最新配置
    mem0 = await get_mem0_client_async(_ctx)
    if user_id == "":
        if memory_config.PRESET_ISOLATION:
            preset = await get_preset(_ctx)
            user_id = preset.name
        else:
            user_id = core_config.BOT_QQ

    if memory_config.SESSION_ISOLATION:
        user_id = _ctx.from_chat_key + user_id

    user_id = user_id.replace(" ", "_")

    try:
        result = await async_mem0_get_all(mem0, user_id=user_id)
        logger.info(f"获取所有记忆结果: {result}")
        return "以下是你搜索到的记忆:\n" + format_memories(result.get("results", []))
    except httpx.HTTPError as e:
        logger.error(f"网络请求失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"获取记忆失败: {e!s}")
        raise RuntimeError(f"获取记忆失败: {e!s}") from e


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="更新记忆",
    description="根据记忆id更新记忆",
)
async def update_memory(_ctx: AgentCtx, memory_id: str, new_content: str) -> str:
    """Updates existing memory content.

    Args:
        memory_id (str): The ID of the memory to update.
        new_content (str): The new memory content text, at least 10 characters long.
    Returns:
        str: Status message of the operation result.

    Example:
        update_memory("bf4d4092...", "Likes to play tennis on weekends")
    """
    mem0 = await get_mem0_client_async(_ctx)
    try:
        original_id = decode_id(memory_id)  # 解码短ID
    except ValueError as e:
        logger.error(f"无效的记忆ID: {e!s}")
        raise ValueError(f"无效的记忆ID格式: {e!s}") from e

    try:
        result = await async_mem0_update(mem0, memory_id=original_id, data=new_content)
        logger.info(f"更新记忆结果: {result}")
        return result.get("message", "记忆更新成功")
    except httpx.HTTPError as e:
        logger.error(f"更新失败: {e!s}")
        raise RuntimeError(f"网络请求失败: {e!s}") from e
    except Exception as e:
        logger.error(f"更新失败: {e!s}")
        raise RuntimeError(f"记忆更新失败: {e!s}") from e


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="查询记忆修改记录",
    description="查询指定记忆的修改记录",
)
async def get_memory_history(_ctx: AgentCtx, memory_id: str) -> str:
    """Gets the modification history of a memory, allowing querying of memory modification history.

    Args:
        memory_id (str): The ID of the memory to query.

    Returns:
        str: A formatted string of the history record, containing the memory modification history.

    Example:
        get_memory_history("bf4d4092...")
    """
    mem0 = await get_mem0_client_async(_ctx)
    try:
        original_id = decode_id(memory_id)  # 解码短ID
    except ValueError as e:
        logger.error(f"无效的记忆ID: {e!s}")
        raise ValueError(f"无效的记忆ID格式: {e!s}") from e

    try:
        records = await async_mem0_history(mem0, memory_id=original_id)
        logger.info(f"获取历史记录结果: {records}")
        if not records:
            return "该记忆暂无历史记录"

        formatted = []
        for idx, r in enumerate(records, 1):
            formatted.append(
                f"{idx}. [事件更改类型: {r['event']}]\n"
                f"旧内容: {r['old_memory'] or '无'}\n"
                f"新内容: {r['new_memory'] or '无'}\n"
                f"时间: {format_beijing_time(r['created_at'])}\n",
            )
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"获取历史失败: {e!s}")
        raise RuntimeError(f"获取记忆历史记录失败: {e!s}") from e


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    name="删除记忆",
    description="删除指定记忆",
)
async def delete_memory(_ctx: AgentCtx, memory_id: str) -> None:
    """Deletes the specified memory. Try using this tool to delete when you find that the retrieved relevant memory content is irrelevant to the conversation.

    Args:
        memory_id (str): The ID of the memory to delete.

    Returns:
        None

    Example:
        delete_memory("bf4d4092...")
    """
    mem0 = await get_mem0_client_async(_ctx)
    mem_id = decode_id(memory_id)
    await async_mem0_delete(mem0, mem_id)

def split_by_last_space(text: str) -> tuple[str, str]:
    match = re.match(r"^(.*)\s+(\S+)$", text.strip())
    if match:
        return match.group(1), match.group(2)
    return "", ""
    
@on_command("memory_search", aliases={"memory-search"}, priority=5, block=True).handle()
async def _(matcher: Matcher, event: MessageEvent, bot: Bot, arg: Message = CommandArg()):
    username, cmd_content, chat_key, chat_type = await command_guard(event, bot, arg, matcher)
    if not cmd_content:
        return await matcher.finish("请输入要搜索的记忆内容")
    _ctx = AgentCtx(
        from_chat_key=chat_key,
    )
    query,userid  = split_by_last_space(cmd_content)
    result = await search_memory(_ctx, query, userid)
    await finish_with(matcher, message=result)
    return None

@on_command("memory_get_all", aliases={"memory-get-all"}, priority=5, block=True).handle()
async def _(matcher: Matcher, event: MessageEvent, bot: Bot, arg: Message = CommandArg()):
    username, cmd_content, chat_key, chat_type = await command_guard(event, bot, arg, matcher)
    _ctx = AgentCtx( 
        from_chat_key=chat_key,
    )
    if not cmd_content:
        return await matcher.finish("请输入要搜索的用户")
    result = await get_all_memories(_ctx, cmd_content)

    await finish_with(matcher, message=result)
    return None

@on_command("memory_delete", aliases={"memory-delete"}, priority=5, block=True).handle()
async def _(matcher: Matcher, event: MessageEvent, bot: Bot, arg: Message = CommandArg()):
    username, cmd_content, chat_key, chat_type = await command_guard(event, bot, arg, matcher)
    if not cmd_content:
        return await matcher.finish("请输入要删除的记忆ID")
    _ctx = AgentCtx(
        from_chat_key=chat_key,
    )
    await delete_memory(_ctx, cmd_content)

    await finish_with(matcher, message="记忆删除成功")
    return None

@plugin.mount_cleanup_method()
async def clean_up():
    global _mem0_instance, _last_config_hash, _thread_pool, _memory_inject_cache
    _mem0_instance = None
    _last_config_hash = None
    _thread_pool.shutdown()  # type: ignore
    _memory_inject_cache = {}
    """清理插件"""
