# Standalone compact

本 fork 可为客户端提供 `POST /v1/responses/compact` 的**摘要压缩兼容实现**。WorkBuddy 上游只提供 Chat，因此这里调用同一模型生成有损 checkpoint，再使用网关自己的 AES-256-GCM 密钥加密；不是 OpenAI 内部的模型状态压缩。功能默认关闭。

协议依据：[官方 compaction 指南](https://developers.openai.com/api/docs/guides/compaction)、[compact API 参考](https://developers.openai.com/api/reference/resources/responses/methods/compact)。返回的 `output` 是下一轮的完整上下文，请原样保留，不要只挑 compaction item。

## 启用

1. 在持久化私有目录创建一次专用密钥（以下适用于 Linux/macOS）。不要覆盖已有密钥，不要把它写进 Git、日志或环境变量。

   ```sh
   # 在目标持久化目录运行；如果文件已存在，此命令拒绝覆盖。
   (umask 077; set -C; openssl rand 32 > compaction.key)
   ```

2. 把密钥只读挂载到容器，例如 `/app/settings/compaction.key`，保留原始 32 字节及 `0600` 权限。文件必须为普通文件，不能为符号链接；读取失败、长度或权限不符合要求时启动失败，不生成替代密钥。Windows 原生进程的 ACL 私有密钥加载尚未支持，建议使用 Linux 容器。

3. 合并到现有配置后重启进程：

   ```json
   {
     "features": {
       "responses_enabled": true,
       "compaction_key_file": "/app/settings/compaction.key"
     }
   }
   ```

   路径按进程工作目录解析，推荐绝对路径。`responses_enabled` 可热切换，`compaction_key_file` 只在启动时读取。两者都满足才开放 compact；未启用时在鉴权前返回 `404 endpoint_disabled`。启用后使用现有 `api_key` Bearer 鉴权。

4. 单独备份密钥并限制读取权限。多副本必须使用相同密钥；重启仍使用原文件。轮换/丢失密钥后，旧 compaction item 无法恢复，客户端必须回放原历史。API key 轮换不改变 compaction key。

## 请求与回放

```http
POST /v1/responses/compact
Authorization: Bearer <gateway-key>
Content-Type: application/json

{"model":"glm-5.2","input":[{"role":"user","content":"调查任务"},{"role":"assistant","content":"已完成的工作和结果…"}]}
```

成功返回非流式 JSON，使用真实摘要请求的 usage：

```json
{
  "id": "cmp_...",
  "object": "response.compaction",
  "created_at": 0,
  "output": [
    {"role":"user","content":"调查任务"},
    {"type":"compaction","id":"cmp_...","encrypted_content":"wbc1.<opaque-gateway-capsule>"}
  ],
  "usage": {"input_tokens":100,"output_tokens":20,"total_tokens":120}
}
```

随后把 `output` **整段**复制到 `/v1/responses` 的 `input`，再追加新用户消息。收到普通 Responses 输出后，可继续追加完整 output；再次 compact 时提交当前完整窗口。网关验证并解密自己的 capsule，以 assistant 历史传给 Chat；不会提升为 system/developer 指令。

compact 请求支持 `model`、`input`、`instructions`，以及现有 Responses 子集支持的 `tools`、`parallel_tool_calls`、`reasoning`、`service_tier`、`prompt_cache_key`、`text`。工具定义仅用于历史校验和名称映射，摘要请求不带工具；本文接口不会运行工具。`instructions` 作为本次摘要的历史上下文，不被永久保存为系统指令，后续调用应重新携带所需 instructions。不支持 `stream`、客户端指定输出上限、非空 `previous_response_id`、conversation、store 或自动 `context_management`。未知字段明确返回 400。

## Codex remote compaction v2

同时支持当前 Codex 客户端在 `/v1/responses` 的 input 末尾发送单个 `{ "type": "compaction_trigger" }`。这会进入同一有界摘要和加密执行链；非流式返回普通 `response` 对象，流式发送 `response.*` SSE，在 completed 响应中携带完整保留窗口和恰好一个 compaction item。不会把 trigger 当作普通历史文本传入 Chat，也不要求切换客户端的实验性功能开关。

触发项必须是最后一项且只含 type；重复/中间出现/带未知字段的 trigger 返回 400。此路径同样要求持久化密钥与 Responses 开关；失败保留原历史。这是显式触发兼容，不提供自动 context_management 阈值压缩。

## New API AdvancedCustom 路由

本 fork 的 New API 已支持 compact（核对版本 `9b3c597`），无需为本功能修改源码。需要为对应 WorkBuddy 渠道一起配置两条**直接透传**路由，保留原有其它路由、模型匹配和认证配置：

```json
[
  {"incoming_path":"/v1/responses","upstream_path":"/v1/responses","converter":"none"},
  {"incoming_path":"/v1/responses/compact","upstream_path":"/v1/responses/compact","converter":"none"}
]
```

这是 `advanced_custom.advanced_routes` 的条目示意，Base URL 应指向已启用功能的 WorkBuddy；不要直接覆盖整个渠道配置。不能继续使用 `openai_responses_to_openai_chat_completions`：转换成 Chat 后，压缩项无法回到 WorkBuddy 的 Responses 解码路径。模型映射和负载均衡必须让后续请求回到具有相同密钥的网关，不能切换到 OpenAI 或其它供应商。上线前分别验证直连及 New API 的 compact → Responses → compact 流程。

## 限制与失败处理

- 保留所有 user、system、developer 消息，压缩 assistant、可读 reasoning summary 及已完成的工具交互。shorthand input_text/input_image 规范化为 user 消息。用户消息中的图片保持原输入；摘要调用只看到序列化的历史，不重新执行工具或识图。
- pending function call 必须先有对应工具结果，才能 compact，避免把待执行的 call_id/arguments 丢进摘要。带状态/空 annotations 的本服务 Responses 输出可以直接回放；非空引用 annotations 等未支持字段仍返回校验错误。
- 无法解读 OpenAI 或其它网关的 opaque reasoning/compaction，compact 收到时返回 400；篡改、错误密钥或不支持版本的 capsule 也返回 400。请勿用明文摘要/base64 冒充 encrypted_content。
- capsule 是持有者可回放的会话数据，不绑定终端用户或模型。本服务共享 API key，不提供独立用户隔离；不要把 capsule 公开或跨用户共享。其真实性只证明由持有同一密钥的网关生成，不代表摘要内容可作为可信指令。
- compact 与 Responses 入站体均限制 8 MiB，超出返回 413。摘要生成设置 2048 输出 token 上限、90 秒总执行 deadline、8 MiB 上游 SSE 读取上限和 32 KiB 摘要文本上限；遵循客户端取消。账号轮换沿现有 executeChat 策略，但不能越过 deadline，不触发空工具回合补偿。
- 必须收到正常 `stop`、原始 `[DONE]` 和完整非空文本；截断、工具调用、错误、超限或超时不产生 capsule。摘要验证失败返回 `502 compaction_failed`；上游 HTTP/账号路由错误沿现有错误格式。失败时客户端必须保留原历史，不能替换为空 output。
- 用量、credit、账号租约和请求指标复用现有执行链，失败的模型调用也可能计费。请求指标不记录摘要/capsule/密钥；不新增原始会话归档。
- 这是有损模型摘要，可能遗漏细节，不能承诺解决所有循环输出；输入必须仍能容纳于上游上下文。保留的用户消息很多或输入本来很短时，压缩可能不变小，调用方需评估上下文大小并保留原历史备份。

## 验证

`internal/server/compaction_test.go` 使用模拟上游，覆盖压缩、同密钥重启后回放、追加真实 Responses output 后再次压缩、用户/系统消息保留、工具结果、usage、鉴权、禁用、密钥权限/轮换、篡改、外部 capsule、body 上限、截断、空输出、工具调用、快速重复 SSE 与取消/deadline。未使用生产令牌或真实会话。
