package server

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

// Explicit opt-in local Codex wire diagnostic; never uses credentials or providers.
func TestCodexNamespaceLoopback(t *testing.T) {
	if os.Getenv("WB_CODEX_LOOPBACK") != "1" {
		t.Skip("local opt-in fixture")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		req, err := parseResponsesRequest(raw)
		if err != nil {
			writeResponsesValidationError(w, err)
			return
		}
		var chat map[string]any
		json.Unmarshal(req.ChatBody, &chat)
		hasOutput := false
		for _, v := range chat["messages"].([]any) {
			if v.(map[string]any)["role"] == "tool" {
				hasOutput = true
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		encoder := &responseSSEEncoder{w: w, flush: func() { w.(http.Flusher).Flush() }}
		state := newResponseState(req, true, encoder)
		delta := map[string]any{"content": "WB_CODEX_OK"}
		finish := "stop"
		if strings.Contains(r.URL.Path, "namespace") && !hasOutput {
			wire, e := req.ToolNames.wire("multi_agent_v1", "wait_agent", "fixture")
			if e != nil {
				t.Error(e)
				return
			}
			delta = map[string]any{"tool_calls": []any{map[string]any{"index": float64(0), "id": "call_namespace_local", "type": "function", "function": map[string]any{"name": wire, "arguments": `{"targets":["local-schema-probe-not-an-agent"],"timeout_ms":10000}`}}}}
			finish = "tool_calls"
		}
		chunk := map[string]any{"choices": []any{map[string]any{"delta": delta, "finish_reason": finish}}, "usage": map[string]any{"prompt_tokens": float64(1), "completion_tokens": float64(1), "total_tokens": float64(2)}}
		if err := state.consumeChunk(chunk); err != nil {
			t.Error(err)
		}
		state.finish(true, nil)
		t.Logf("fixture path=%s replay=%t status=%v", r.URL.Path, hasOutput, state.response["status"])
	})
	t.Fatal(http.ListenAndServe(":17865", mux))
}
