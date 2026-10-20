package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/livecfg"
	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
	"github.com/linguo2625469/workbuddy2api-panel/internal/session"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

func TestRequestUsesAutoFunctionTools(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{"default-auto", `{"tools":[{"type":"function","function":{"name":"f"}}]}`, true},
		{"explicit-auto", `{"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto"}`, true},
		{"forced-choice", `{"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":{"type":"function","function":{"name":"f"}}}`, false},
		{"none", `{"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none"}`, false},
		{"required", `{"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"required"}`, false},
		{"no-function-tools", `{"tools":[{"type":"custom","custom":{"name":"f"}}]}`, false},
		{"empty-tools", `{"tools":[]}`, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := requestUsesAutoFunctionTools([]byte(tc.body)); got != tc.want {
				t.Fatalf("requestUsesAutoFunctionTools=%v want %v", got, tc.want)
			}
		})
	}
}

func TestIsCodexCompactionRequest(t *testing.T) {
	prompt := codexCompactionPromptPrefix + " Create a handoff summary."
	cases := []struct {
		name     string
		messages []any
		want     bool
	}{
		{
			name: "string-content",
			messages: []any{
				map[string]any{"role": "user", "content": prompt},
			},
			want: true,
		},
		{
			name: "text-content-part",
			messages: []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "text", "text": prompt},
				}},
			},
			want: true,
		},
		{
			name: "input-text-content-part",
			messages: []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "input_text", "text": prompt},
				}},
			},
			want: true,
		},
		{
			name: "last-user-message-wins",
			messages: []any{
				map[string]any{"role": "user", "content": prompt},
				map[string]any{"role": "user", "content": "normal task"},
			},
			want: false,
		},
		{
			name: "system-message-does-not-count",
			messages: []any{
				map[string]any{"role": "system", "content": prompt},
			},
			want: false,
		},
		{
			name: "non-text-content",
			messages: []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AA=="}},
				}},
			},
			want: false,
		},
		{
			name:     "no-messages",
			messages: nil,
			want:     false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{"messages": tc.messages})
			if err != nil {
				t.Fatal(err)
			}
			if got := isCodexCompactionRequest(body); got != tc.want {
				t.Fatalf("isCodexCompactionRequest=%v want %v body=%s", got, tc.want, body)
			}
		})
	}
	if isCodexCompactionRequest([]byte("{")) {
		t.Fatal("malformed JSON must not be treated as a compaction request")
	}
}

func TestEmptyToolTurnRetrySkipsCodexCompaction(t *testing.T) {
	live := livecfg.New(livecfg.Snapshot{EmptyToolTurnRetryModels: []string{"*"}})
	h := NewHandler(Config{Live: live})
	tools := []any{map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "run", "parameters": map[string]any{"type": "object"},
		},
	}}
	compactionBody, err := json.Marshal(map[string]any{
		"messages":    []any{map[string]any{"role": "user", "content": codexCompactionPromptPrefix + " Create a handoff summary."}},
		"tools":       tools,
		"tool_choice": "auto",
	})
	if err != nil {
		t.Fatal(err)
	}
	if h.emptyToolTurnRetryEnabled("deepseek-v4.1-flash", compactionBody) {
		t.Fatal("Codex compaction request must not enable empty-tool-turn retry")
	}

	normalBody, err := json.Marshal(map[string]any{
		"messages":    []any{map[string]any{"role": "user", "content": "normal task"}},
		"tools":       tools,
		"tool_choice": "auto",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !h.emptyToolTurnRetryEnabled("deepseek-v4.1-flash", normalBody) {
		t.Fatal("normal tool request must keep empty-tool-turn retry enabled")
	}
}

func TestEmptyToolTurnRetryModelMatchingIsRealmAwareAndSupportsAllModels(t *testing.T) {
	body := []byte(`{"tools":[{"type":"function","function":{"name":"f"}}]}`)
	live := livecfg.New(livecfg.Snapshot{EmptyToolTurnRetryModels: []string{"cn:deepseek-v4.1-flash"}})
	h := NewHandler(Config{Live: live})
	if !h.emptyToolTurnRetryEnabled("deepseek-v4.1-flash", body) {
		t.Fatal("bare CN model should match its configured realm-qualified name")
	}
	if h.emptyToolTurnRetryEnabled("global:deepseek-v4.1-flash", body) {
		t.Fatal("configured CN model must not enable the global realm")
	}
	live.Store(livecfg.Snapshot{EmptyToolTurnRetryModels: []string{"*"}})
	for _, model := range []string{"deepseek-v4.1-flash", "global:glm-5.3", "cn:another-model"} {
		if !h.emptyToolTurnRetryEnabled(model, body) {
			t.Errorf("wildcard should enable %q", model)
		}
	}
	if h.emptyToolTurnRetryEnabled("", body) {
		t.Fatal("wildcard must not enable a request with no model")
	}
	live.Store(livecfg.Snapshot{})
	if h.emptyToolTurnRetryEnabled("deepseek-v4.1-flash", body) {
		t.Fatal("empty live allowlist must disable retries")
	}
}

func TestReadValidatedResponseLimit(t *testing.T) {
	exact := bytes.Repeat([]byte{'x'}, validatedResponseLimit)
	got, err := readValidatedResponse(bytes.NewReader(exact))
	if err != nil || len(got) != validatedResponseLimit {
		t.Fatalf("exact-limit read len=%d err=%v", len(got), err)
	}
	tooLarge := bytes.Repeat([]byte{'x'}, validatedResponseLimit+1)
	if _, err := readValidatedResponse(bytes.NewReader(tooLarge)); !errors.Is(err, errValidatedResponseTooLarge) {
		t.Fatalf("over-limit error=%v want errValidatedResponseTooLarge", err)
	}
}

func TestMergeUsageMapsAddsAllCountersAndRecomputesTotal(t *testing.T) {
	prior := map[string]any{
		"prompt_tokens": float64(10), "completion_tokens": float64(2), "total_tokens": float64(12), "credit": float64(0.1),
		"prompt_cache_hit_tokens":   float64(8),
		"prompt_tokens_details":     map[string]any{"cached_tokens": float64(8)},
		"completion_tokens_details": map[string]any{"reasoning_tokens": float64(1)},
	}
	current := map[string]any{
		"prompt_tokens": float64(20), "completion_tokens": float64(5), "total_tokens": float64(25), "credit": float64(0.2),
		"prompt_cache_hit_tokens":   float64(12),
		"prompt_tokens_details":     map[string]any{"cached_tokens": float64(12)},
		"completion_tokens_details": map[string]any{"reasoning_tokens": float64(2)},
	}
	got := mergeUsageMaps(prior, current)
	for key, want := range map[string]float64{
		"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37,
		"credit": 0.3, "prompt_cache_hit_tokens": 20,
	} {
		if value, _ := usageNumberValue(got[key]); value < want-1e-9 || value > want+1e-9 {
			t.Errorf("%s=%v want %v", key, got[key], want)
		}
	}
	if details := got["completion_tokens_details"].(map[string]any); details["reasoning_tokens"] != float64(3) {
		t.Fatalf("reasoning counters=%#v", details)
	}
}

func TestAddPriorUsageToChatSSERewritesOrInjectsTerminalUsage(t *testing.T) {
	prior := map[string]any{"prompt_tokens": float64(4), "completion_tokens": float64(2), "total_tokens": float64(6), "credit": float64(0.1)}
	withUsage := []byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13,\"credit\":0.2}}\n\ndata: [DONE]\n\n")
	patched, err := addPriorUsageToChatSSE(withUsage, prior)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(patched, []byte(`"prompt_tokens":14`)) || !bytes.Contains(patched, []byte(`"completion_tokens":5`)) || !bytes.Contains(patched, []byte(`"total_tokens":19`)) {
		t.Fatalf("existing usage was not aggregated: %s", patched)
	}

	withoutUsage := []byte("data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	patched, err = addPriorUsageToChatSSE(withoutUsage, prior)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(patched), `"usage"`) != 1 || !strings.Contains(string(patched), `"prompt_tokens":4`) || strings.Index(string(patched), `"usage"`) > strings.Index(string(patched), "[DONE]") {
		t.Fatalf("usage-only frame was not injected before DONE: %s", patched)
	}
}

type emptyToolTurnCapturedRequest struct {
	Authorization string
	Body          []byte
	TurnRequestID string
}

func emptyToolTurnFixture(t *testing.T, hasTool bool, promptTokens, completionTokens, reasoningTokens int, credit float64) string {
	t.Helper()
	frames := []map[string]any{}
	delta := map[string]any{"reasoning_content": "reasoning from this attempt"}
	if hasTool {
		delta = map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "id": "call_test", "type": "function",
			"function": map[string]any{"name": "run", "arguments": "{}"},
		}}}
	}
	frames = append(frames, map[string]any{
		"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-v4.1-flash",
		"choices": []any{map[string]any{"index": 0, "delta": delta}},
	})
	usage := map[string]any{
		"prompt_tokens": promptTokens, "completion_tokens": completionTokens,
		"total_tokens": promptTokens + completionTokens, "credit": credit,
		"prompt_cache_hit_tokens":   promptTokens / 2,
		"prompt_cache_miss_tokens":  promptTokens - promptTokens/2,
		"prompt_tokens_details":     map[string]any{"cached_tokens": promptTokens / 2},
		"completion_tokens_details": map[string]any{"reasoning_tokens": reasoningTokens},
	}
	finish := "stop"
	if hasTool {
		finish = "tool_calls"
	}
	frames = append(frames, map[string]any{
		"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-v4.1-flash",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}},
		"usage":   usage,
	})
	var b strings.Builder
	for _, frame := range frames {
		encoded, err := json.Marshal(frame)
		if err != nil {
			t.Fatal(err)
		}
		b.WriteString("data: ")
		b.Write(encoded)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func emptyToolTurnEmptyFixture(t *testing.T, promptTokens, completionTokens, reasoningTokens int, credit float64) string {
	t.Helper()
	frames := []map[string]any{
		{
			"id": "chatcmpl-empty", "object": "chat.completion.chunk", "created": 1,
			"model": "deepseek-v4.1-flash",
			"choices": []any{map[string]any{
				"index": 0, "delta": map[string]any{},
			}},
		},
		{
			"id": "chatcmpl-empty", "object": "chat.completion.chunk", "created": 1,
			"model": "deepseek-v4.1-flash",
			"choices": []any{map[string]any{
				"index": 0, "delta": map[string]any{}, "finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens": promptTokens, "completion_tokens": completionTokens,
				"total_tokens": promptTokens + completionTokens, "credit": credit,
				"completion_tokens_details": map[string]any{"reasoning_tokens": reasoningTokens},
			},
		},
	}
	var b strings.Builder
	for _, frame := range frames {
		encoded, err := json.Marshal(frame)
		if err != nil {
			t.Fatal(err)
		}
		b.WriteString("data: ")
		b.Write(encoded)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func emptyToolTurnContentFixture(t *testing.T, content string, promptTokens, completionTokens, reasoningTokens int, credit float64) string {
	t.Helper()
	frames := []map[string]any{
		{
			"id": "chatcmpl-content", "object": "chat.completion.chunk", "created": 1,
			"model": "deepseek-v4.1-flash",
			"choices": []any{map[string]any{
				"index": 0, "delta": map[string]any{"content": content},
			}},
		},
		{
			"id": "chatcmpl-content", "object": "chat.completion.chunk", "created": 1,
			"model": "deepseek-v4.1-flash",
			"choices": []any{map[string]any{
				"index": 0, "delta": map[string]any{}, "finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens": promptTokens, "completion_tokens": completionTokens,
				"total_tokens": promptTokens + completionTokens, "credit": credit,
				"completion_tokens_details": map[string]any{"reasoning_tokens": reasoningTokens},
			},
		},
	}
	var b strings.Builder
	for _, frame := range frames {
		encoded, err := json.Marshal(frame)
		if err != nil {
			t.Fatal(err)
		}
		b.WriteString("data: ")
		b.Write(encoded)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func codexCompactionSummaryFixture(t *testing.T, summary string) string {
	t.Helper()
	frames := []map[string]any{
		{
			"id": "chatcmpl-compaction", "object": "chat.completion.chunk", "created": 1,
			"model": "deepseek-v4.1-flash",
			"choices": []any{map[string]any{
				"index": 0, "delta": map[string]any{"content": summary},
			}},
		},
		{
			"id": "chatcmpl-compaction", "object": "chat.completion.chunk", "created": 1,
			"model": "deepseek-v4.1-flash",
			"choices": []any{map[string]any{
				"index": 0, "delta": map[string]any{}, "finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
			},
		},
	}
	var b strings.Builder
	for _, frame := range frames {
		encoded, err := json.Marshal(frame)
		if err != nil {
			t.Fatal(err)
		}
		b.WriteString("data: ")
		b.Write(encoded)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func newEmptyToolTurnTestUpstream(t *testing.T, next func(int) string, captures *[]emptyToolTurnCapturedRequest) *upstream.Client {
	t.Helper()
	calls := 0
	return &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			time.Sleep(time.Millisecond)
			body, err := io.ReadAll(r.Body)
			if err != nil {
				return nil, err
			}
			*captures = append(*captures, emptyToolTurnCapturedRequest{
				Authorization: r.Header.Get("Authorization"),
				Body:          append([]byte(nil), body...),
				TurnRequestID: r.Header.Get("X-Conversation-Request-ID"),
			})
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(next(calls))),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
}

func assertEmptyTurnRequestsSwitchedAccount(t *testing.T, captured []emptyToolTurnCapturedRequest) {
	t.Helper()
	if len(captured) < 2 {
		t.Fatalf("captured %d attempts, want at least 2", len(captured))
	}
	if captured[0].Authorization == captured[1].Authorization {
		t.Fatalf("retry reused the same account authorization: %q", captured[0].Authorization)
	}
	var firstBody, secondBody map[string]any
	if err := json.Unmarshal(captured[0].Body, &firstBody); err != nil {
		t.Fatalf("decode first body: %v", err)
	}
	if err := json.Unmarshal(captured[1].Body, &secondBody); err != nil {
		t.Fatalf("decode second body: %v", err)
	}
	firstCacheKey, _ := firstBody["prompt_cache_key"].(string)
	secondCacheKey, _ := secondBody["prompt_cache_key"].(string)
	if firstCacheKey == "" || secondCacheKey == "" || firstCacheKey == secondCacheKey {
		t.Fatalf("account-specific prompt_cache_key not recomputed: %q vs %q", firstCacheKey, secondCacheKey)
	}
	delete(firstBody, "prompt_cache_key")
	delete(secondBody, "prompt_cache_key")
	normalizedFirst, err := json.Marshal(firstBody)
	if err != nil {
		t.Fatal(err)
	}
	normalizedSecond, err := json.Marshal(secondBody)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(normalizedFirst, normalizedSecond) {
		t.Fatalf("logical request body changed across accounts:\nfirst=%s\nsecond=%s", normalizedFirst, normalizedSecond)
	}
	for i := 1; i < len(captured); i++ {
		if captured[i].TurnRequestID != captured[0].TurnRequestID {
			t.Errorf("attempt %d changed turn request ID", i+1)
		}
	}
}

func TestChatEmptyToolTurnRetrySwitchesAccountAndAggregatesUsage(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "non-stream", true: "stream"}[stream], func(t *testing.T) {
			var captured []emptyToolTurnCapturedRequest
			calls := 0
			up := newEmptyToolTurnTestUpstream(t, func(int) string {
				calls++
				if calls == 1 {
					return emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)
				}
				return emptyToolTurnFixture(t, true, 20, 5, 2, 0.2)
			}, &captured)
			p := testPoolWith(
				&auth.Auth{UID: "retry-a", AccessToken: "test-at-a", ExpiresAt: 9999999999},
				&auth.Auth{UID: "retry-b", AccessToken: "test-at-b", ExpiresAt: 9999999999},
			)
			reqLog := reqlog.New(reqlog.Config{})
			defer reqLog.Close()
			h := NewHandler(Config{
				Pool: p, Upstream: up, RequestLog: reqLog,
				EmptyToolTurnRetryModels: []string{"*"},
				EmptyToolTurnMaxRetries:  1,
				EmptyToolTurnRetryBudget: 75 * time.Second,
			})
			body, err := json.Marshal(map[string]any{
				"model": "deepseek-v4.1-flash", "stream": stream,
				"messages":    []any{map[string]any{"role": "user", "content": "run a tool"}},
				"tools":       []any{map[string]any{"type": "function", "function": map[string]any{"name": "run", "parameters": map[string]any{"type": "object"}}}},
				"tool_choice": "auto",
			})
			if err != nil {
				t.Fatal(err)
			}
			observation, err := upstream.InspectEmptyToolTurn([]byte(emptyToolTurnFixture(t, true, 20, 5, 2, 0.2)), body)
			if err != nil || observation.ValidToolCalls != 1 {
				t.Fatalf("test tool-call fixture not recognized: observation=%+v err=%v", observation, err)
			}
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
			if rec.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
			}
			if calls != 2 {
				t.Fatalf("upstream calls=%d want 2", calls)
			}
			assertEmptyTurnRequestsSwitchedAccount(t, captured)
			if stream {
				if !strings.Contains(rec.Body.String(), `"finish_reason":"tool_calls"`) ||
					!strings.Contains(rec.Body.String(), `"prompt_tokens":30`) ||
					!strings.Contains(rec.Body.String(), `"completion_tokens":7`) ||
					!strings.Contains(rec.Body.String(), `"total_tokens":37`) ||
					!strings.Contains(rec.Body.String(), `"reasoning_tokens":2`) {
					t.Fatalf("stream output/usage not recovered and aggregated: %s", rec.Body.String())
				}
			} else {
				var response map[string]any
				if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				choices := response["choices"].([]any)
				message := choices[0].(map[string]any)["message"].(map[string]any)
				if len(message["tool_calls"].([]any)) != 1 {
					t.Fatalf("final tool call missing: %#v", message)
				}
				usage := response["usage"].(map[string]any)
				if usage["prompt_tokens"] != float64(30) || usage["completion_tokens"] != float64(7) || usage["total_tokens"] != float64(37) || usage["credit"] != float64(0.30000000000000004) {
					t.Fatalf("combined usage=%#v", usage)
				}
				details := usage["completion_tokens_details"].(map[string]any)
				if details["reasoning_tokens"] != float64(2) {
					t.Fatalf("reasoning usage=%#v", details)
				}
			}
			event := reqLog.Snapshot().Recent[0]
			if event.Attempts != 2 || event.PromptTokens != 30 || event.CompletionTokens != 7 || event.TotalTokens != 37 || len(event.EmptyToolTurnAttempts) != 2 {
				t.Fatalf("request archive did not aggregate attempts: %+v", event)
			}
			if event.EmptyToolTurnAttempts[0].Outcome != "retry_account" || event.EmptyToolTurnAttempts[1].Outcome != "recovered" || event.EmptyToolTurnAttempts[0].ReasoningTokens != 0 {
				t.Fatalf("per-attempt record=%+v", event.EmptyToolTurnAttempts)
			}
			if event.EmptyToolTurnAttempts[0].DurationMs <= 0 || event.EmptyToolTurnAttempts[1].DurationMs <= 0 {
				t.Fatalf("per-attempt durations not recorded: %+v", event.EmptyToolTurnAttempts)
			}
			if event.EmptyToolTurnAttempts[0].Account == "" || event.EmptyToolTurnAttempts[1].Account == "" ||
				event.EmptyToolTurnAttempts[0].Account == event.EmptyToolTurnAttempts[1].Account {
				t.Fatalf("per-attempt accounts not distinct: %+v", event.EmptyToolTurnAttempts)
			}
			for _, uid := range []string{"retry-a", "retry-b"} {
				status, ok := p.Status(uid)
				if !ok || status.Cooling || status.Disabled || status.ErrTotal != 0 {
					t.Fatalf("semantic retries penalized %s: %+v", uid, status)
				}
			}
		})
	}
}

func TestEmptyToolTurnRetryDoesNotDiscardFinalTextAfterToolResult(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		return emptyToolTurnContentFixture(t, "The final answer is row 105.", 10, 2, 0, 0.1)
	}, &captured)
	p := testPoolWith(
		&auth.Auth{UID: "final-a", AccessToken: "final-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "final-b", AccessToken: "final-at-b", ExpiresAt: 9999999999},
	)
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	body := `{
		"model":"deepseek-v4.1-flash","stream":true,
		"messages":[
			{"role":"user","content":"read line 105"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"line\":105}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"TrainingData row 105"}
		],
		"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],
		"tool_choice":"auto"
	}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if calls != 1 || len(captured) != 1 {
		t.Fatalf("upstream calls=%d captured=%d want one final-text attempt", calls, len(captured))
	}
	if !strings.Contains(rec.Body.String(), "The final answer is row 105.") ||
		!strings.Contains(rec.Body.String(), "finish_reason") {
		t.Fatalf("final text response was not preserved: %s", rec.Body.String())
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 1 || len(event.EmptyToolTurnAttempts) != 0 || !event.OK {
		t.Fatalf("final text was misclassified as an empty tool turn: %+v", event)
	}
}

func TestEmptyToolTurnRetryDoesNotRetryReasoningOnly(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		return emptyToolTurnFixture(t, false, 10, 2, 1, 0.1)
	}, &captured)
	p := testPoolWith(
		&auth.Auth{UID: "reason-a", AccessToken: "reason-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "reason-b", AccessToken: "reason-at-b", ExpiresAt: 9999999999},
	)
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	body := `{"model":"deepseek-v4.1-flash","stream":true,"messages":[{"role":"user","content":"answer"}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK || calls != 1 || len(captured) != 1 {
		t.Fatalf("status=%d calls=%d captured=%d body=%s", rec.Code, calls, len(captured), rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "reasoning from this attempt") {
		t.Fatalf("reasoning-only response was not passed through: %s", rec.Body.String())
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 1 || len(event.EmptyToolTurnAttempts) != 0 || !event.OK {
		t.Fatalf("reasoning-only response was retried: %+v", event)
	}
}

func TestResponsesEmptyToolTurnRetryBuffersAndAggregatesUsage(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		if calls == 1 {
			return emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)
		}
		return emptyToolTurnFixture(t, true, 20, 5, 2, 0.2)
	}, &captured)
	p := testPoolWith(
		&auth.Auth{UID: "responses-a", AccessToken: "responses-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "responses-b", AccessToken: "responses-at-b", ExpiresAt: 9999999999},
	)
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	body := `{"model":"deepseek-v4.1-flash","input":"run a tool","stream":true,"store":false,"tools":[{"type":"function","name":"run","parameters":{"type":"object"}}],"tool_choice":"auto"}`
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if calls != 2 {
		t.Fatalf("upstream calls=%d want 2", calls)
	}
	assertEmptyTurnRequestsSwitchedAccount(t, captured)
	if strings.Count(rec.Body.String(), "event: response.created") != 1 || strings.Contains(rec.Body.String(), "reasoning from this attempt") {
		t.Fatalf("rejected attempt leaked Responses events: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "event: response.completed") || !strings.Contains(rec.Body.String(), `"input_tokens":30`) || !strings.Contains(rec.Body.String(), `"output_tokens":7`) || !strings.Contains(rec.Body.String(), `"total_tokens":37`) || !strings.Contains(rec.Body.String(), `"reasoning_tokens":2`) {
		t.Fatalf("final Responses event lacks aggregate usage: %s", rec.Body.String())
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 2 || len(event.EmptyToolTurnAttempts) != 2 || event.TotalTokens != 37 {
		t.Fatalf("request archive=%+v", event)
	}
	if event.EmptyToolTurnAttempts[0].Outcome != "retry_account" || event.EmptyToolTurnAttempts[1].Outcome != "recovered" {
		t.Fatalf("per-attempt outcomes=%+v", event.EmptyToolTurnAttempts)
	}
}

func TestResponsesFinalTextIsNotRetriedAsEmptyToolTurn(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		return emptyToolTurnContentFixture(t, "final response text", 10, 2, 0, 0.1)
	}, &captured)
	p := testPoolWith(
		&auth.Auth{UID: "responses-final-a", AccessToken: "responses-final-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "responses-final-b", AccessToken: "responses-final-at-b", ExpiresAt: 9999999999},
	)
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	body := `{"model":"deepseek-v4.1-flash","input":"answer","stream":true,"store":false,"tools":[{"type":"function","name":"read","parameters":{"type":"object"}}],"tool_choice":"auto"}`
	r := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusOK || calls != 1 || len(captured) != 1 {
		t.Fatalf("status=%d calls=%d captured=%d body=%s", rec.Code, calls, len(captured), rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "event: response.completed") ||
		!strings.Contains(rec.Body.String(), "final response text") {
		t.Fatalf("final Responses text was not completed: %s", rec.Body.String())
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 1 || len(event.EmptyToolTurnAttempts) != 0 || !event.OK {
		t.Fatalf("final Responses text was retried: %+v", event)
	}
}

func TestEmptyToolTurnExhaustionStopsAfterOneAccountSwitchWithoutPenalizingAccounts(t *testing.T) {
	for _, route := range []string{"/v1/chat/completions", "/v1/responses"} {
		t.Run(route, func(t *testing.T) {
			var captured []emptyToolTurnCapturedRequest
			calls := 0
			up := newEmptyToolTurnTestUpstream(t, func(int) string {
				calls++
				return emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)
			}, &captured)
			p := testPoolWith(
				&auth.Auth{UID: "exhausted-a", AccessToken: "exhausted-at-a", ExpiresAt: 9999999999},
				&auth.Auth{UID: "exhausted-b", AccessToken: "exhausted-at-b", ExpiresAt: 9999999999},
			)
			reqLog := reqlog.New(reqlog.Config{})
			defer reqLog.Close()
			h := NewHandler(Config{
				Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true, RequestLog: reqLog,
				EmptyToolTurnRetryModels: []string{"cn:deepseek-v4.1-flash"},
				EmptyToolTurnMaxRetries:  1,
				EmptyToolTurnRetryBudget: 75 * time.Second,
			})
			var body string
			if route == "/v1/responses" {
				body = `{"model":"deepseek-v4.1-flash","input":"run","stream":true,"store":false,"tools":[{"type":"function","name":"run","parameters":{"type":"object"}}],"tool_choice":"auto"}`
			} else {
				body = `{"model":"deepseek-v4.1-flash","stream":true,"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
			}
			r := httptest.NewRequest(http.MethodPost, route, strings.NewReader(body))
			r.Header.Set("Authorization", "Bearer key")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, r)
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "upstream_empty_tool_turn") {
				t.Fatalf("status=%d body=%s want committed stream error", rec.Code, rec.Body.String())
			}
			if calls != 2 || len(captured) != 2 {
				t.Fatalf("upstream calls=%d captured=%d want 2 (original + 1 account switch)", calls, len(captured))
			}
			assertEmptyTurnRequestsSwitchedAccount(t, captured)
			if strings.Contains(rec.Body.String(), "reasoning from this attempt") {
				t.Fatalf("failed attempt body leaked to client: %s", rec.Body.String())
			}
			if route == "/v1/responses" {
				if strings.Count(rec.Body.String(), "event: response.created") != 1 ||
					!strings.Contains(rec.Body.String(), "event: response.failed") ||
					!strings.Contains(rec.Body.String(), `"code":"upstream_empty_tool_turn"`) {
					t.Fatalf("Responses terminal stream error missing: %s", rec.Body.String())
				}
			} else if !strings.Contains(rec.Body.String(), `"code":"upstream_empty_tool_turn"`) ||
				!strings.Contains(rec.Body.String(), "data: [DONE]") {
				t.Fatalf("Chat terminal stream error missing: %s", rec.Body.String())
			}
			for _, uid := range []string{"exhausted-a", "exhausted-b"} {
				status, ok := p.Status(uid)
				if !ok || status.Cooling || status.Disabled || status.ErrTotal != 0 {
					t.Fatalf("empty tool turn penalized %s: %+v", uid, status)
				}
			}
			event := reqLog.Snapshot().Recent[0]
			if event.Attempts != 2 || len(event.EmptyToolTurnAttempts) != 2 ||
				event.EmptyToolTurnAttempts[0].Outcome != "retry_account" ||
				event.EmptyToolTurnAttempts[1].Outcome != "exhausted" {
				t.Fatalf("attempt history=%+v", event)
			}
			if event.Status != http.StatusOK || event.OK || event.Outcome != reqlog.OutcomeStreamError {
				t.Fatalf("terminal stream archive=%+v", event)
			}
		})
	}
}

func TestEmptyToolTurnMaxRetriesZeroReturnsCommittedStreamErrorWithoutSwitching(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		return emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)
	}, &captured)
	p := testPoolWith(
		&auth.Auth{UID: "zero-a", AccessToken: "zero-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "zero-b", AccessToken: "zero-at-b", ExpiresAt: 9999999999},
	)
	store := newBindStore()
	sess := session.New(session.Config{
		TTL:       time.Minute,
		Store:     store,
		Available: func() []string { return []string{"zero-a", "zero-b"} },
	})
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, Session: sess, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  0,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	sess.Bind("conv-zero", "zero-a")
	body := `{"model":"deepseek-v4.1-flash","stream":true,"messages":[{"role":"user","content":"run"}],"metadata":{"conversation_id":"conv-zero"},"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "upstream_empty_tool_turn") || !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if calls != 1 || len(captured) != 1 {
		t.Fatalf("calls=%d captured=%d want one attempt", calls, len(captured))
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 1 || len(event.EmptyToolTurnAttempts) != 1 || event.EmptyToolTurnAttempts[0].Outcome != "exhausted" {
		t.Fatalf("attempt history=%+v", event)
	}
	if event.Status != http.StatusOK || event.OK || event.Outcome != reqlog.OutcomeStreamError {
		t.Fatalf("terminal stream archive=%+v", event)
	}
	if uid, ok := store.lastUID("conv-zero"); ok {
		t.Fatalf("terminal empty tool turn left sticky binding on %q", uid)
	}
}

func TestEmptyToolTurnReturnsNoAlternateAccountWithoutSecondCall(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		return emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)
	}, &captured)
	p := testPoolWith(&auth.Auth{UID: "only", AccessToken: "only-at", ExpiresAt: 9999999999})
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	body := `{"model":"deepseek-v4.1-flash","stream":true,"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "upstream_empty_tool_turn") {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if calls != 1 || len(captured) != 1 {
		t.Fatalf("calls=%d captured=%d want one attempt", calls, len(captured))
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 1 || len(event.EmptyToolTurnAttempts) != 1 || event.EmptyToolTurnAttempts[0].Outcome != "no_alternate_account" {
		t.Fatalf("attempt history=%+v", event)
	}
	if event.Status != http.StatusOK || event.OK || event.Outcome != reqlog.OutcomeStreamError {
		t.Fatalf("terminal stream archive=%+v", event)
	}
}

type flushSignalRecorder struct {
	*httptest.ResponseRecorder
	once    sync.Once
	flushed chan struct{}
}

func newFlushSignalRecorder() *flushSignalRecorder {
	return &flushSignalRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		flushed:          make(chan struct{}),
	}
}

func (w *flushSignalRecorder) Flush() {
	w.ResponseRecorder.Flush()
	w.once.Do(func() { close(w.flushed) })
}

func TestEmptyToolTurnPreambleFlushesBeforeRetryWait(t *testing.T) {
	var (
		calls         int
		capture       []emptyToolTurnCapturedRequest
		secondStarted = make(chan *io.PipeWriter, 1)
	)
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			body, err := io.ReadAll(r.Body)
			if err != nil {
				return nil, err
			}
			capture = append(capture, emptyToolTurnCapturedRequest{
				Authorization: r.Header.Get("Authorization"),
				Body:          body,
				TurnRequestID: r.Header.Get("X-Conversation-Request-ID"),
			})
			var respBody io.ReadCloser
			if calls == 1 {
				respBody = io.NopCloser(strings.NewReader(emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)))
			} else {
				pipeR, pipeW := io.Pipe()
				secondStarted <- pipeW
				respBody = pipeR
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       respBody,
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(
		&auth.Auth{UID: "flush-a", AccessToken: "flush-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "flush-b", AccessToken: "flush-at-b", ExpiresAt: 9999999999},
	)
	reqLog := reqlog.New(reqlog.Config{})
	defer reqLog.Close()
	h := NewHandler(Config{
		Pool: p, Upstream: up, RequestLog: reqLog,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	body := `{"model":"deepseek-v4.1-flash","stream":true,"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
	rec := newFlushSignalRecorder()
	done := make(chan struct{})
	go func() {
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
		close(done)
	}()
	select {
	case <-rec.flushed:
	case <-time.After(3 * time.Second):
		t.Fatal("stream preamble was not flushed before the retry wait")
	}
	var pipeW *io.PipeWriter
	select {
	case pipeW = <-secondStarted:
	case <-time.After(3 * time.Second):
		t.Fatal("second upstream attempt did not reach the blocking response")
	}
	if _, err := io.WriteString(pipeW, emptyToolTurnFixture(t, true, 20, 5, 2, 0.2)); err != nil {
		t.Fatal(err)
	}
	_ = pipeW.Close()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("handler did not finish after releasing the retry")
	}
	if rec.Code != http.StatusOK || calls != 2 || len(capture) != 2 {
		t.Fatalf("status=%d calls=%d capture=%d body=%s", rec.Code, calls, len(capture), rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"role":"assistant"`) || !strings.Contains(rec.Body.String(), `"finish_reason":"tool_calls"`) {
		t.Fatalf("preamble or recovered tool call missing: %s", rec.Body.String())
	}
	event := reqLog.Snapshot().Recent[0]
	if event.Attempts != 2 || len(event.EmptyToolTurnAttempts) != 2 {
		t.Fatalf("attempt archive=%+v", event)
	}
}

func TestEmptyToolTurnRetryStateHonorsBudgetAndDeadline(t *testing.T) {
	now := time.Now()
	state := emptyToolTurnRetryState{
		maxRetries: 1,
		deadline:   now.Add(10 * time.Second),
	}
	if canRetry, outcome := state.canRetry(now, true); canRetry || outcome != "deadline" {
		t.Fatalf("canRetry=%v outcome=%s want deadline", canRetry, outcome)
	}
	state.deadline = now.Add(30 * time.Second)
	if canRetry, outcome := state.canRetry(now, true); !canRetry || outcome != "retry_account" {
		t.Fatalf("canRetry=%v outcome=%s want retry_account", canRetry, outcome)
	}
	state.markRetry()
	ctx, cancel, local := state.attemptContext(context.Background(), now)
	defer cancel()
	if !local {
		t.Fatal("retry attempt must use the local budget deadline")
	}
	if deadline, ok := ctx.Deadline(); !ok || !deadline.Equal(state.deadline) {
		t.Fatalf("attempt deadline=%v ok=%v want %v", deadline, ok, state.deadline)
	}
}

func TestEmptyToolTurnRetryUnbindsStickyAccountAndBindsRecoveredAccount(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		if calls == 1 {
			return emptyToolTurnEmptyFixture(t, 10, 2, 0, 0.1)
		}
		return emptyToolTurnFixture(t, true, 20, 5, 2, 0.2)
	}, &captured)
	store := newBindStore()
	sess := session.New(session.Config{
		TTL:       time.Minute,
		Store:     store,
		Available: func() []string { return []string{"sticky-a", "sticky-b"} },
	})
	p := testPoolWith(
		&auth.Auth{UID: "sticky-a", AccessToken: "sticky-at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "sticky-b", AccessToken: "sticky-at-b", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{
		Pool: p, Upstream: up, Session: sess,
		EmptyToolTurnRetryModels: []string{"*"},
		EmptyToolTurnMaxRetries:  1,
		EmptyToolTurnRetryBudget: 75 * time.Second,
	})
	sess.Bind("conv-empty", "sticky-a")
	body := `{"model":"deepseek-v4.1-flash","stream":false,"messages":[{"role":"user","content":"run"}],"metadata":{"conversation_id":"conv-empty"},"tools":[{"type":"function","function":{"name":"run","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	assertEmptyTurnRequestsSwitchedAccount(t, captured)
	if uid, ok := store.lastUID("conv-empty"); !ok || uid != "sticky-b" {
		t.Fatalf("sticky binding=%q ok=%v want sticky-b (binds=%v)", uid, ok, store.binds)
	}
}

func TestCodexCompactionSummarySkipsEmptyToolTurnRetry(t *testing.T) {
	prompt := codexCompactionPromptPrefix + " Create a handoff summary."
	for _, route := range []string{"/v1/chat/completions", "/v1/responses"} {
		t.Run(route, func(t *testing.T) {
			var captured []emptyToolTurnCapturedRequest
			calls := 0
			up := newEmptyToolTurnTestUpstream(t, func(int) string {
				calls++
				return codexCompactionSummaryFixture(t, "compaction summary")
			}, &captured)
			p := testPoolWith(&auth.Auth{UID: "compaction", AccessToken: "compaction-at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
			h := NewHandler(Config{
				Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true,
				EmptyToolTurnRetryModels: []string{"*"},
			})
			var payload map[string]any
			if route == "/v1/responses" {
				payload = map[string]any{
					"model": "deepseek-v4.1-flash", "input": prompt, "stream": true, "store": false,
					"tools": []any{map[string]any{
						"type": "function", "name": "run", "parameters": map[string]any{"type": "object"},
					}},
					"tool_choice": "auto",
				}
			} else {
				payload = map[string]any{
					"model": "deepseek-v4.1-flash", "stream": true,
					"messages": []any{map[string]any{"role": "user", "content": prompt}},
					"tools": []any{map[string]any{
						"type": "function",
						"function": map[string]any{
							"name": "run", "parameters": map[string]any{"type": "object"},
						},
					}},
					"tool_choice": "auto",
				}
			}
			body, err := json.Marshal(payload)
			if err != nil {
				t.Fatal(err)
			}
			r := httptest.NewRequest(http.MethodPost, route, bytes.NewReader(body))
			r.Header.Set("Authorization", "Bearer key")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, r)
			if rec.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
			}
			if calls != 1 || len(captured) != 1 {
				t.Fatalf("upstream calls=%d captured=%d want 1", calls, len(captured))
			}
			if strings.Contains(rec.Body.String(), "upstream_empty_tool_turn") || !strings.Contains(rec.Body.String(), "compaction summary") {
				t.Fatalf("compaction summary was not returned normally: %s", rec.Body.String())
			}
			if route == "/v1/responses" && !strings.Contains(rec.Body.String(), "event: response.completed") {
				t.Fatalf("Responses compaction did not complete: %s", rec.Body.String())
			}
		})
	}
}

func TestEmptyToolTurnNoRetryWhenFeatureDisabled(t *testing.T) {
	var captured []emptyToolTurnCapturedRequest
	calls := 0
	up := newEmptyToolTurnTestUpstream(t, func(int) string {
		calls++
		return emptyToolTurnFixture(t, false, 1, 1, 1, 0)
	}, &captured)
	p := testPoolWith(&auth.Auth{UID: "disabled", AccessToken: "disabled-at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	h := NewHandler(Config{Pool: p, Upstream: up})
	body := `{"model":"deepseek-v4.1-flash","stream":false,"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"run"}}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK || calls != 1 {
		t.Fatalf("status=%d calls=%d body=%s; default behavior must remain unchanged", rec.Code, calls, rec.Body.String())
	}
}
