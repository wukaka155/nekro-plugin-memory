# Nekro Agent 插件 - 记忆模块 (nekro-plugin-memory)

[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/zxjwzn/nekro-plugin-memory)
[![Author](https://img.shields.io/badge/author-Zaxpris-brightgreen)](https://github.com/zxjwzn)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE) <!-- 请根据实际情况添加 LICENSE 文件 -->

一个为 [Nekro Agent](https://github.com/KroMiose/nekro-agent) <!-- 替换为 Nekro Agent 的实际链接 --> 提供长期记忆管理能力的插件，基于 [mem0](https://github.com/mem0ai/mem0) 和 [Qdrant](https://qdrant.tech/) 构建。

## 功能特性

*   **长期记忆存储**: 将对话信息、用户偏好或其他重要数据作为长期记忆存储。
*   **增删改查**: 支持对记忆的添加、搜索、获取、更新和历史记录查询。
*   **语义搜索**: 通过自然语言描述智能查找相关记忆。
*   **自动记忆检索**:
    *   在对话开始时自动检索相关记忆。
    *   可配置基于最近对话内容或通过 LLM 总结的话题进行检索。
    *   自动将检索到的记忆注入 Agent 上下文。
*   **会话隔离**: 可选配置，使记忆仅在当前会话中有效。
*   **短 ID**: 使用 Base62 编码的短 ID 方便引用和操作记忆。
*   **灵活配置**: 支持自定义用于记忆处理和向量嵌入的模型组、嵌入维度、自动检索参数等。

## 安装

1.  确保已安装 Nekro Agent。
2.  通过WebUI进行插件下载
3.  确保 Nekro Agent 配置了可用的 Qdrant 实例以及所需的模型组（用于记忆处理和嵌入）。

## 配置

插件提供以下配置项 (`MemoryConfig`)：

*   `MEMORY_MANAGE_MODEL` (str): 用于简化和整理记忆内容的 LLM 模型组名称 (必填)。
*   `TEXT_EMBEDDING_MODEL` (str): 用于将记忆内容向量化的嵌入模型组名称 (必填)。
*   `TEXT_EMBEDDING_DIMENSION` (int): 文本嵌入的维度 (默认: 1024)。
*   `SESSION_ISOLATION` (bool): 是否开启会话隔离 (默认: True)。开启后，记忆仅对当前会话有效。
*   `AUTO_MEMORY_ENABLED` (bool): 是否启用自动记忆检索 (默认: True)。
*   `AUTO_MEMORY_SEARCH_LIMIT` (int): 自动检索时返回的最大记忆条数 (默认: 5)。
*   `AUTO_MEMORY_CONTEXT_MESSAGE_COUNT` (int): 自动检索时参考的上下文消息数量 (默认: 5)。
*   `AUTO_MEMORY_USE_TOPIC_SEARCH` (bool): 是否启用基于 LLM 话题总结的记忆搜索 (默认: True)。启用可能增加响应时间。
*   `TOPIC_CACHE_EXPIRE_SECONDS` (int): 话题缓存的有效时间（秒） (默认: 60)。

请在 Nekro Agent 的WebUI中设置这些参数。务必确保引用的模型组已正确配置且 API Key/Base URL 等信息有效。

## 内置方法

该插件向 Agent 提供了以下可调用的工具 (Tool) / 方法 (Behavior/Agent):

*   **`添加记忆` (Tool)**
    *   **描述**: 为指定用户添加一条长期记忆。
    *   **参数**:
        *   `memory` (str): 需要记忆的内容。
        *   `user_id` (str): 关联的用户 ID。
        *   `metadata` (dict): 附加的元数据 (可选)。
    *   **返回**: 成功或失败的消息。

*   **`搜索记忆` (Agent)**
    *   **描述**: 根据模糊描述搜索用户的相关记忆。
    *   **参数**:
        *   `query` (str): 用于搜索的自然语言描述。
        *   `user_id` (str): 关联的用户 ID。
    *   **返回**: 格式化后的记忆搜索结果列表 (包含短 ID)。

*   **`获取记忆` (Agent)**
    *   **描述**: 获取指定用户的所有记忆。
    *   **参数**:
        *   `user_id` (str): 关联的用户 ID。
    *   **返回**: 格式化后的该用户所有记忆列表 (包含短 ID)。

*   **`更新记忆` (Behavior)**
    *   **描述**: 根据记忆的短 ID 更新其内容。
    *   **参数**:
        *   `memory_id` (str): 要更新的记忆的短 ID。
        *   `new_content` (str): 新的记忆内容。
    *   **返回**: 成功或失败的消息。

*   **`查询记忆修改记录` (Agent)**
    *   **描述**: 查询指定记忆的修改历史。
    *   **参数**:
        *   `memory_id` (str): 要查询的记忆的短 ID。
    *   **返回**: 格式化后的记忆修改历史。

此外，当 `AUTO_MEMORY_ENABLED` 开启时，插件会自动在后台工作，检索相关记忆并注入提示词，无需显式调用。

## 注意事项

*   请确保 Nekro Agent 配置的 Qdrant 服务可用。
*   请确保 `MEMORY_MANAGE_MODEL` 和 `TEXT_EMBEDDING_MODEL` 指向的模型组配置正确且可用（包括 API Key, Base URL 等）。

## 贡献

欢迎提交 Issues 和 Pull Requests。

## 作者

Zaxpris ([https://github.com/zxjwzn](https://github.com/zxjwzn))
实际上是Gemini小姐,不想写readme,桀桀桀
