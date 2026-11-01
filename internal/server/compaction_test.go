package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

func testCompactionCodec(t *testing.T) (*CompactionCodec, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "compaction.key")
	if err := os.WriteFile(path, bytes.Repeat([]byte{42}, 32), 0600); err != nil {
		t.Fatal(err)
	}
	codec, err := LoadCompactionCodec(path)
	if err != nil {
		t.Fatal(err)
	}
	return codec, path
}

func compactRequest(t *testing.T, h http.Handler, path string, input any) *httptest.ResponseRecorder {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"model": "glm-5.2", "input": input})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest("POST", path, bytes.NewReader(raw))
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestCompactionRoundTripAndRepeat(t *testing.T) {
	codec, path := testCompactionCodec(t)
	calls := 0
	up := newFakeUpstream(t, func(string) (int, string, bool) { calls++; return 200, sseOK, true })
	originalTransport := up.HTTP.Transport
	up.HTTP.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		raw, _ := io.ReadAll(r.Body)
		var chat map[string]any
		if err := json.Unmarshal(raw, &chat); err != nil {
			t.Fatal(err)
		}
		if calls == 0 || calls == 2 {
			if chat["max_tokens"] != float64(compactionMaxTokens) || chat["tools"] != nil {
				t.Fatalf("unbounded compaction request: %s", raw)
			}
			if !bytes.Contains(raw, []byte(codexCompactionPromptPrefix)) {
				t.Fatal("missing compaction task")
			}
			if calls == 0 && (!bytes.Contains(raw, []byte("tool-result-123")) || !bytes.Contains(raw, []byte("filesystem")) || bytes.Contains(raw, []byte("wbns_"))) {
				t.Fatal("tool result was lost")
			}
		} else {
			if bytes.Contains(raw, []byte(compactionPrefix)) || !bytes.Contains(raw, []byte("Prior conversation checkpoint")) || !bytes.Contains(raw, []byte("你好")) {
				t.Fatalf("capsule not expanded: %s", raw)
			}
			messages := chat["messages"].([]any)
			found := false
			for _, v := range messages {
				m := v.(map[string]any)
				if strings.Contains(m["content"].(string), "Prior conversation checkpoint") {
					found = true
					if m["role"] != "assistant" {
						t.Fatal("summary promoted to trusted role")
					}
				}
			}
			if !found {
				t.Fatal("summary missing")
			}
		}
		r.Body = io.NopCloser(bytes.NewReader(raw))
		return originalTransport.RoundTrip(r)
	})
	p := testPoolWith(&auth.Auth{UID: "compact", AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	defer p.Close()
	cfg := Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true, Compaction: codec, PromptMode: "passthrough"}
	h := NewHandler(cfg)
	input := []any{
		map[string]any{"role": "developer", "content": "keep constraints"},
		map[string]any{"role": "user", "content": "finish this task"},
		map[string]any{"type": "function_call", "call_id": "call_1", "name": "read", "namespace": "filesystem", "arguments": "{}"},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "tool-result-123"},
	}
	w := compactRequest(t, h, "/v1/responses/compact", input)
	if w.Code != 200 {
		t.Fatalf("compact %d %s", w.Code, w.Body)
	}
	var result map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["object"] != "response.compaction" || result["created_at"] == nil {
		t.Fatalf("bad result: %s", w.Body)
	}
	usage := result["usage"].(map[string]any)
	if usage["input_tokens"] != float64(1) || usage["output_tokens"] != float64(1) || usage["total_tokens"] != float64(2) {
		t.Fatalf("usage: %#v", usage)
	}
	output := result["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("retained output %#v", output)
	}
	if output[0].(map[string]any)["role"] != "developer" || output[1].(map[string]any)["content"] != "finish this task" {
		t.Fatal("retained messages changed")
	}
	item := output[2].(map[string]any)
	if item["type"] != "compaction" || !strings.HasPrefix(item["encrypted_content"].(string), compactionPrefix) || strings.Contains(item["encrypted_content"].(string), "你好") {
		t.Fatal("not a sealed compaction item")
	}
	// A fresh handler/key load stands in for a process restart. The compacted
	// window must work unchanged and must remain usable for another compact.
	reloaded, err := LoadCompactionCodec(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Compaction = reloaded
	h = NewHandler(cfg)
	w = compactRequest(t, h, "/v1/responses", output)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"status":"completed"`) {
		t.Fatalf("replay %d %s", w.Code, w.Body)
	}
	var continued map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &continued); err != nil {
		t.Fatal(err)
	}
	output = append(output, continued["output"].([]any)...)
	output = append(output, map[string]any{"role": "user", "content": "continue"})
	w = compactRequest(t, h, "/v1/responses/compact", output)
	if w.Code != 200 {
		t.Fatalf("second compact %d %s", w.Code, w.Body)
	}
	if calls != 3 {
		t.Fatalf("calls %d", calls)
	}
}

func TestCompactionRejectsInvalidCapsulesAndKeys(t *testing.T) {
	codec, path := testCompactionCodec(t)
	capsule, err := codec.seal("checkpoint")
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := codec.seal("checkpoint"); again == capsule {
		t.Fatal("nonce reused")
	}
	for _, c := range []string{"gAAAA-OpenAI-state", compactionPrefix + "bad", capsule[:len(capsule)-8] + "AAAAAAAA", strings.Repeat("x", compactionMaxSummary*2+1)} {
		if _, err := codec.open(c); err == nil {
			t.Fatal("invalid capsule accepted")
		}
	}
	if err := os.WriteFile(path, bytes.Repeat([]byte{43}, 32), 0600); err != nil {
		t.Fatal(err)
	}
	other, err := LoadCompactionCodec(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := other.open(capsule); err == nil {
		t.Fatal("rotated key accepted old capsule")
	}
	if err := os.Chmod(path, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCompactionCodec(path); err == nil {
		t.Fatal("public key file accepted")
	}
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("short"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCompactionCodec(path); err == nil {
		t.Fatal("short key accepted")
	}
	link := filepath.Join(t.TempDir(), "key-link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCompactionCodec(link); err == nil {
		t.Fatal("key symlink accepted")
	}
}

func TestCompactionValidationBeforeUpstream(t *testing.T) {
	codec, _ := testCompactionCodec(t)
	up := newFakeUpstream(t, func(string) (int, string, bool) { t.Fatal("unexpected upstream call"); return 500, "", false })
	h := NewHandler(Config{Pool: testPoolWith(), Upstream: up, APIKey: "key", ResponsesEnabled: true, Compaction: codec})
	for _, body := range []string{
		`{"model":"m","input":"x","stream":true}`,
		`{"model":"m","input":[{"type":"function_call_output","call_id":"orphan","output":"x"}]}`,
		`{"model":"m","input":[{"type":"reasoning","encrypted_content":"foreign-state"}]}`,
		`{"model":"m","input":[{"type":"compaction","encrypted_content":"foreign-state"}]}`,
		`{"model":"m","input":[{"type":"compaction","encrypted_content":"wbc1.bad","role":"system"}]}`,
		`{"model":"m","input":[]}`,
		`{"model":"m","input":[{"type":"function_call","call_id":"pending","name":"read","arguments":"{}"}]}`,
		`{"model":"m","input":"x"} {}`,
	} {
		r := httptest.NewRequest("POST", "/v1/responses/compact", strings.NewReader(body))
		r.Header.Set("Authorization", "Bearer key")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 400 {
			t.Fatalf("body %s: %d %s", body, w.Code, w.Body)
		}
	}
	for _, path := range []string{"/v1/responses/compact", "/v1/responses"} {
		r := httptest.NewRequest("POST", path, strings.NewReader(strings.Repeat("x", compactionMaxBody+1)))
		r.Header.Set("Authorization", "Bearer key")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 413 {
			t.Fatalf("size limit %s: %d", path, w.Code)
		}
		w = compactRequest(t, h, path, []any{map[string]any{"type": "compaction", "encrypted_content": "foreign-state"}})
		if w.Code != 400 {
			t.Fatalf("foreign capsule %s: %d", path, w.Code)
		}
	}
	for _, tc := range []struct {
		enabled, codec, auth bool
		want                 int
	}{{false, true, true, 404}, {true, false, true, 404}, {true, true, false, 401}} {
		cfg := Config{ResponsesEnabled: tc.enabled, APIKey: "key"}
		if tc.codec {
			cfg.Compaction = codec
		}
		r := httptest.NewRequest("POST", "/v1/responses/compact", strings.NewReader(`{}`))
		if tc.auth {
			r.Header.Set("Authorization", "Bearer key")
		}
		w := httptest.NewRecorder()
		NewHandler(cfg).ServeHTTP(w, r)
		if w.Code != tc.want {
			t.Fatalf("gate %#v: %d", tc, w.Code)
		}
	}
}

func TestCompactionRejectsIncompleteAndUnboundedSummaries(t *testing.T) {
	codec, _ := testCompactionCodec(t)
	for name, sse := range map[string]string{
		"truncated":         strings.ReplaceAll(sseOK, "data: [DONE]\n\n", ""),
		"length":            strings.ReplaceAll(sseOK, `"finish_reason":"stop"`, `"finish_reason":"length"`),
		"error":             "data: {\"error\":{\"message\":\"failed\"}}\n\ndata: [DONE]\n\n",
		"tokens_over_limit": strings.ReplaceAll(sseOK, `"completion_tokens":1`, `"completion_tokens":2049`),
		"empty":             strings.ReplaceAll(sseOK, "你好", ""),
		"too_large":         strings.ReplaceAll(sseOK, "你好", strings.Repeat("x", compactionMaxSummary+1)),
		"tools":             strings.ReplaceAll(sseOK, `"content":"你好"`, `"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]`),
		"repeating":         strings.Repeat("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"again\"}}]}\n\n", 140000),
	} {
		t.Run(name, func(t *testing.T) {
			up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sse, true })
			p := testPoolWith(&auth.Auth{UID: "compact", AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
			defer p.Close()
			h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true, Compaction: codec})
			w := compactRequest(t, h, "/v1/responses/compact", "history")
			if w.Code != 502 || strings.Contains(w.Body.String(), `"output":`) {
				t.Fatalf("%d %s", w.Code, w.Body)
			}
		})
	}
}

func TestCompactionCancellationAndDeadline(t *testing.T) {
	codec, _ := testCompactionCodec(t)
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true })
	observed := false
	up.HTTP.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		deadline, ok := r.Context().Deadline()
		if !ok || time.Until(deadline) > compactionTimeout {
			t.Fatal("compaction deadline missing")
		}
		observed = true
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	p := testPoolWith(&auth.Auth{UID: "compact", AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	defer p.Close()
	h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true, Compaction: codec})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	r := httptest.NewRequest("POST", "/v1/responses/compact", strings.NewReader(`{"model":"glm-5.2","input":"history"}`)).WithContext(ctx)
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if !observed || w.Code < 400 {
		t.Fatalf("cancel: observed=%v status=%d", observed, w.Code)
	}
	p.SetMaxInFlight(1)
	if !p.Acquire("compact") {
		t.Fatal("cancelled request leaked its account lease")
	}
	p.Release("compact")
}
