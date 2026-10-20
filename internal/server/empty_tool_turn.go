package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

const (
	validatedResponseLimit      = 8 << 20
	codexCompactionPromptPrefix = "You are performing a CONTEXT CHECKPOINT COMPACTION."
	// emptyToolTurnMinRetryRemaining 防止在 75 秒预算只剩几秒时再发起一次
	// 必然被截断的上游请求。它只约束“是否值得开始第二次尝试”，不截断第一次。
	emptyToolTurnMinRetryRemaining = 15 * time.Second
)

var errValidatedResponseTooLarge = errors.New("upstream response exceeded the 8 MiB validation buffer")

type emptyToolTurnRetryState struct {
	maxRetries int
	retries    int
	deadline   time.Time
	usage      map[string]any
}

func (h *Handler) newEmptyToolTurnRetryState(started time.Time) emptyToolTurnRetryState {
	live := h.loadLive()
	state := emptyToolTurnRetryState{maxRetries: live.EmptyToolTurnMaxRetries}
	if live.EmptyToolTurnRetryBudget > 0 {
		state.deadline = started.Add(live.EmptyToolTurnRetryBudget)
	}
	return state
}

func (s *emptyToolTurnRetryState) noteUsage(response map[string]any) {
	if usage := usageMap(response); usage != nil {
		s.usage = mergeUsageMaps(s.usage, usage)
	}
}

func (s *emptyToolTurnRetryState) hasRetried() bool { return s.retries > 0 }

// canRetry 返回 false 时，reason 直接作为请求归档的终态标签。
func (s *emptyToolTurnRetryState) canRetry(now time.Time, hasAlternate bool) (bool, string) {
	if s.retries >= s.maxRetries {
		return false, "exhausted"
	}
	if !hasAlternate {
		return false, "no_alternate_account"
	}
	if !s.deadline.IsZero() && now.Add(emptyToolTurnMinRetryRemaining).After(s.deadline) {
		return false, "deadline"
	}
	return true, "retry_account"
}

func (s *emptyToolTurnRetryState) markRetry() { s.retries++ }

// attemptContext 只给换号后的后续尝试加本地截止时间；第一次尝试保持现有
// 传输层/客户端超时语义，不被空工具回合预算强行截断。
func (s *emptyToolTurnRetryState) attemptContext(parent context.Context, now time.Time) (context.Context, context.CancelFunc, bool) {
	if s.retries == 0 || s.deadline.IsZero() || !now.Before(s.deadline) {
		return parent, func() {}, false
	}
	ctx, cancel := context.WithDeadline(parent, s.deadline)
	return ctx, cancel, true
}

func (s *emptyToolTurnRetryState) deadlineHit(localDeadline bool, ctx context.Context) bool {
	if !localDeadline || s.deadline.IsZero() {
		return false
	}
	return errors.Is(ctx.Err(), context.DeadlineExceeded) || !time.Now().Before(s.deadline)
}

func (h *Handler) hasAlternateEmptyToolTurnAccount(model, realm string, tried map[string]bool) bool {
	for _, uid := range h.cfg.Pool.AvailableUIDsForModelRealm(model, realm) {
		if !tried[uid] {
			return true
		}
	}
	return false
}

func (h *Handler) emptyToolTurnRetryEnabled(model string, requestBody []byte) bool {
	if !requestUsesAutoFunctionTools(requestBody) {
		return false
	}
	// Codex uses the same tool-enabled Chat shape for its context compaction
	// turn. The compaction response is intentionally a summary with
	// finish_reason=stop and no tool calls, so treating it as an empty tool
	// turn would retry the summary and return a terminal error.
	if isCodexCompactionRequest(requestBody) {
		log.Printf("INFO: [server] empty_tool_turn skipped reason=codex_compaction model=%s", model)
		return false
	}
	realm, bareModel := resolveModel(model)
	for _, configured := range h.loadLive().EmptyToolTurnRetryModels {
		if configured == "*" && strings.TrimSpace(model) != "" {
			return true
		}
		configuredRealm, configuredModel := resolveModel(configured)
		if configuredRealm == realm && configuredModel == bareModel {
			return true
		}
	}
	return false
}

func isCodexCompactionRequest(body []byte) bool {
	var request struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if json.Unmarshal(body, &request) != nil {
		return false
	}
	for i := len(request.Messages) - 1; i >= 0; i-- {
		message := request.Messages[i]
		if !strings.EqualFold(strings.TrimSpace(message.Role), "user") {
			continue
		}
		text := messageContentText(message.Content)
		return strings.HasPrefix(strings.TrimSpace(text), codexCompactionPromptPrefix)
	}
	return false
}

func messageContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var textBuilder strings.Builder
	for _, part := range parts {
		if part.Type == "text" || part.Type == "input_text" {
			textBuilder.WriteString(part.Text)
		}
	}
	return textBuilder.String()
}

func requestUsesAutoFunctionTools(body []byte) bool {
	var request struct {
		Tools []struct {
			Type     string          `json:"type"`
			Function json.RawMessage `json:"function"`
		} `json:"tools"`
		ToolChoice json.RawMessage `json:"tool_choice"`
	}
	if json.Unmarshal(body, &request) != nil {
		return false
	}
	functionToolDeclared := false
	for _, tool := range request.Tools {
		if len(tool.Function) > 0 && !bytes.Equal(bytes.TrimSpace(tool.Function), []byte("null")) && (tool.Type == "" || tool.Type == "function") {
			functionToolDeclared = true
			break
		}
	}
	if !functionToolDeclared {
		return false
	}
	choice := bytes.TrimSpace(request.ToolChoice)
	if len(choice) == 0 || bytes.Equal(choice, []byte("null")) {
		return true
	}
	var value string
	return json.Unmarshal(choice, &value) == nil && value == "auto"
}

func readValidatedResponse(r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, validatedResponseLimit+1))
	if err != nil {
		return nil, err
	}
	if len(data) > validatedResponseLimit {
		return nil, errValidatedResponseTooLarge
	}
	return data, nil
}

// mergeUsageMaps combines counters from earlier rejected attempts (prior) and
// the accepted attempt (current). Non-numeric metadata prefers the final
// response; total_tokens is recomputed from prompt/completion when both exist.
func mergeUsageMaps(prior, current map[string]any) map[string]any {
	if prior == nil && current == nil {
		return nil
	}
	out := mergeUsageValues(prior, current)
	if prompt, okP := usageNumberValue(out["prompt_tokens"]); okP {
		if completion, okC := usageNumberValue(out["completion_tokens"]); okC {
			out["total_tokens"] = prompt + completion
		}
	}
	return out
}

func mergeUsageValues(prior, current map[string]any) map[string]any {
	out := make(map[string]any, len(prior)+len(current))
	for key, value := range prior {
		out[key] = value
	}
	for key, value := range current {
		if priorValue, ok := prior[key]; ok {
			priorMap, priorIsMap := priorValue.(map[string]any)
			currentMap, currentIsMap := value.(map[string]any)
			if priorIsMap && currentIsMap {
				out[key] = mergeUsageValues(priorMap, currentMap)
				continue
			}
			if a, okA := usageNumberValue(priorValue); okA {
				if b, okB := usageNumberValue(value); okB {
					out[key] = a + b
					continue
				}
			}
		}
		out[key] = value
	}
	return out
}

func usageNumberValue(value any) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case int32:
		return float64(number), true
	case json.Number:
		result, err := number.Float64()
		return result, err == nil
	default:
		return 0, false
	}
}

func addPriorUsageToChatSSE(raw []byte, prior map[string]any) ([]byte, error) {
	if len(prior) == 0 {
		return raw, nil
	}
	parts := strings.SplitAfter(string(raw), "\n")
	usageIndex, doneIndex := -1, -1
	var finalFrame map[string]any
	var finalUsage map[string]any
	var id, model string
	var created any
	for index, part := range parts {
		line := strings.TrimRight(part, "\r\n")
		if line == "data: [DONE]" {
			doneIndex = index
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var frame map[string]any
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame) != nil {
			continue
		}
		if id == "" {
			id, _ = frame["id"].(string)
			model, _ = frame["model"].(string)
			created = frame["created"]
		}
		if usage, ok := frame["usage"].(map[string]any); ok {
			usageIndex, finalFrame, finalUsage = index, frame, usage
		}
	}
	combined := mergeUsageMaps(prior, finalUsage)
	if usageIndex >= 0 {
		finalFrame["usage"] = combined
		encoded, err := json.Marshal(finalFrame)
		if err != nil {
			return nil, err
		}
		parts[usageIndex] = "data: " + string(encoded) + lineEnding(parts[usageIndex])
	} else {
		if id == "" {
			id = "chatcmpl-wb2api"
		}
		if created == nil {
			created = time.Now().Unix()
		}
		if model == "" {
			model = "unknown"
		}
		frame := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created,
			"model": model, "choices": []any{}, "usage": combined,
		}
		encoded, err := json.Marshal(frame)
		if err != nil {
			return nil, err
		}
		insertAt := len(parts)
		if doneIndex >= 0 {
			insertAt = doneIndex
		}
		parts = append(parts, "")
		copy(parts[insertAt+1:], parts[insertAt:])
		parts[insertAt] = "data: " + string(encoded) + "\n\n"
	}
	return []byte(strings.Join(parts, "")), nil
}

func lineEnding(part string) string {
	if strings.HasSuffix(part, "\r\n") {
		return "\r\n"
	}
	if strings.HasSuffix(part, "\n") {
		return "\n"
	}
	return ""
}

// writeChatStreamPreamble commits a protocol-valid, content-free Chat SSE
// preface. It is used only after a real empty tool turn has been identified,
// immediately before an account retry or terminal stream failure. The preamble
// makes the downstream Responses converter emit response.created while the
// rejected attempts remain fully buffered and cannot leak their text.
func writeChatStreamPreamble(w http.ResponseWriter, model string) error {
	if w == nil {
		return errors.New("nil stream writer")
	}
	header := w.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	header.Set("X-Accel-Buffering", "no")
	frame := map[string]any{
		"id":      "chatcmpl-wb2api-preface",
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         map[string]any{"role": "assistant"},
			"finish_reason": nil,
		}},
	}
	raw, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: "+string(raw)+"\n\n"); err != nil {
		return err
	}
	flushResponseWriter(w)
	return nil
}

// writeChatStreamError terminates an already-committed Chat SSE response.
// NewAPI recognizes the top-level error object and converts it to a Responses
// response.failed event; direct Chat clients also receive an unambiguous error
// followed by [DONE].
func writeChatStreamError(w http.ResponseWriter, code, message, hint string) error {
	if w == nil {
		return errors.New("nil stream writer")
	}
	errorBody := map[string]any{
		"message": message,
		"type":    "api_error",
		"code":    code,
	}
	if hint != "" {
		errorBody["gateway_hint"] = hint
	}
	raw, err := json.Marshal(map[string]any{"error": errorBody})
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: "+string(raw)+"\n\n"); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	flushResponseWriter(w)
	return nil
}

func flushResponseWriter(w http.ResponseWriter) {
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

type bufferedStreamWriter struct {
	header http.Header
	status int
	body   bytes.Buffer
}

func newBufferedStreamWriter() *bufferedStreamWriter {
	return &bufferedStreamWriter{header: make(http.Header)}
}

func (w *bufferedStreamWriter) Header() http.Header { return w.header }
func (w *bufferedStreamWriter) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
	}
}
func (w *bufferedStreamWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.body.Write(p)
}
func (w *bufferedStreamWriter) Flush() {}

func (w *bufferedStreamWriter) CommitTo(dst http.ResponseWriter, priorUsage map[string]any) error {
	body := w.body.Bytes()
	if len(priorUsage) > 0 {
		var err error
		body, err = addPriorUsageToChatSSE(body, priorUsage)
		if err != nil {
			return err
		}
	}
	for key, values := range w.header {
		dst.Header()[key] = append([]string(nil), values...)
	}
	if w.status != 0 {
		dst.WriteHeader(w.status)
	}
	_, err := dst.Write(body)
	return err
}

func usageMap(response map[string]any) map[string]any {
	usage, _ := response["usage"].(map[string]any)
	return usage
}

func applyUsageToChatStat(st *chatStat, usage map[string]any) {
	if st == nil || usage == nil {
		return
	}
	if value, ok := usageNumberValue(usage["prompt_tokens"]); ok {
		st.promptTokens = int64(value)
	}
	if value, ok := usageNumberValue(usage["completion_tokens"]); ok {
		st.completionTokens = int64(value)
		st.toks = int(value)
	}
	if value, ok := usageNumberValue(usage["total_tokens"]); ok {
		st.totalTokens = int64(value)
	}
	if value, ok := usageNumberValue(usage["credit"]); ok {
		st.credit = value
		st.hasCredit = true
	}
	hit, hasHit := usageNumberValue(usage["prompt_cache_hit_tokens"])
	miss, hasMiss := usageNumberValue(usage["prompt_cache_miss_tokens"])
	if hasHit || hasMiss {
		st.cacheHit, st.cacheMiss = int64(hit), int64(miss)
		st.hasCache = true
	}
}

func attemptUsageDetails(usage map[string]any) (prompt, completion, total, cacheHit, cacheMiss, reasoning int64, credit float64, hasUsage, hasCredit bool) {
	if usage == nil {
		return
	}
	promptF, hasPrompt := usageNumberValue(usage["prompt_tokens"])
	completionF, hasCompletion := usageNumberValue(usage["completion_tokens"])
	totalF, hasTotal := usageNumberValue(usage["total_tokens"])
	prompt, completion, total = int64(promptF), int64(completionF), int64(totalF)
	if !hasTotal && (hasPrompt || hasCompletion) {
		total = prompt + completion
	}
	hasUsage = hasPrompt || hasCompletion || hasTotal
	if v, ok := usageNumberValue(usage["credit"]); ok {
		credit, hasCredit = v, true
	}
	if v, ok := usageNumberValue(usage["prompt_cache_hit_tokens"]); ok {
		cacheHit = int64(v)
	}
	if v, ok := usageNumberValue(usage["prompt_cache_miss_tokens"]); ok {
		cacheMiss = int64(v)
	}
	if details, ok := usage["completion_tokens_details"].(map[string]any); ok {
		if v, ok := usageNumberValue(details["reasoning_tokens"]); ok {
			reasoning = int64(v)
		}
	}
	return
}
