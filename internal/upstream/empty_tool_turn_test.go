package upstream

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func emptyToolTurnSSE(content string, finish string, done bool) string {
	return emptyToolTurnSSEWithReasoningTokens(content, finish, done, 0)
}

func emptyToolTurnSSEWithReasoningTokens(content string, finish string, done bool, reasoningTokens int) string {
	var b strings.Builder
	b.WriteString("data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":")
	encoded, _ := jsonMarshalString(content)
	b.Write(encoded)
	b.WriteString("}}]}\n\n")
	if finish != "" {
		fmt.Fprintf(&b, "data: {\"id\":\"chatcmpl-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":%q}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5,\"credit\":0.25,\"completion_tokens_details\":{\"reasoning_tokens\":%d}}}\n\n", finish, reasoningTokens)
	}
	if done {
		b.WriteString("data: [DONE]\n\n")
	}
	return b.String()
}

func jsonMarshalString(s string) ([]byte, error) { return json.Marshal(s) }

func TestInspectEmptyToolTurnClassifiesOnlyExplicitStopWithoutCalls(t *testing.T) {
	request := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"tool_choice":"auto"}`)
	cases := []struct {
		name string
		raw  string
		want bool
	}{
		{name: "truly-empty", raw: emptyToolTurnSSE("", "stop", true), want: true},
		{name: "whitespace-only", raw: emptyToolTurnSSE("  \n\t", "stop", true), want: true},
		{name: "reasoning-only", raw: "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", want: false},
		{name: "reasoning-token-only", raw: emptyToolTurnSSEWithReasoningTokens("", "stop", true, 2), want: false},
		{name: "short-content", raw: emptyToolTurnSSE("ok", "stop", true), want: false},
		{name: "long-content", raw: emptyToolTurnSSE(strings.Repeat("x", 250_000), "stop", true), want: false},
		{name: "empty-finish-is-not-terminal", raw: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"\"}]}\n\ndata: [DONE]\n\n", want: false},
		{name: "missing-finish-is-not-terminal", raw: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n", want: false},
		{name: "missing-done-is-not-terminal", raw: emptyToolTurnSSE("x", "stop", false), want: false},
		{name: "length-is-not-retried", raw: emptyToolTurnSSE("x", "length", true), want: false},
		{name: "content-filter-is-not-retried", raw: emptyToolTurnSSE("x", "content_filter", true), want: false},
		{name: "error-frame-is-not-retried", raw: "data: {\"error\":{\"code\":\"temporary\"}}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := InspectEmptyToolTurn([]byte(tc.raw), request)
			if err != nil {
				t.Fatalf("InspectEmptyToolTurn: %v", err)
			}
			if got.EmptyToolTurnCandidate() != tc.want {
				t.Fatalf("candidate=%v want %v (observation=%+v)", got.EmptyToolTurnCandidate(), tc.want, got)
			}
		})
	}
}

func TestInspectEmptyToolTurnCountsRepairedAndStructuredCalls(t *testing.T) {
	request := []byte(`{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"tool_choice":"auto"}`)
	markup := `Here<` + markupPipe + `DSML` + markupPipe + ` calls>` +
		`<` + markupPipe + `DSML` + markupPipe + ` invoke name="read">` +
		`<` + markupPipe + `DSML` + markupPipe + ` parameter name="path" string="true">/tmp/a` +
		markupParamClose +
		`</` + markupPipe + `DSML` + markupPipe + ` invoke></` + markupPipe + `DSML` + markupPipe + ` calls>`
	observation, err := InspectEmptyToolTurn([]byte(emptyToolTurnSSE(markup, "stop", true)), request)
	if err != nil {
		t.Fatal(err)
	}
	if observation.EmptyToolTurnCandidate() || observation.RepairedToolCalls != 1 || observation.ValidToolCalls != 1 {
		t.Fatalf("DSML call was not treated as a real tool call: %+v", observation)
	}

	structured := "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/a\\\"}\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	observation, err = InspectEmptyToolTurn([]byte(structured), request)
	if err != nil {
		t.Fatal(err)
	}
	if observation.EmptyToolTurnCandidate() || observation.ValidToolCalls != 1 {
		t.Fatalf("structured tool call was not recognized: %+v", observation)
	}
}

func TestEmptyToolTurnTerminalReasonDistinguishesContentAndReasoning(t *testing.T) {
	request := []byte(`{"tools":[{"type":"function","function":{"name":"read"}}]}`)
	content, err := InspectEmptyToolTurn([]byte(emptyToolTurnSSE("final answer", "stop", true)), request)
	if err != nil {
		t.Fatal(err)
	}
	if got := content.EmptyToolTurnTerminalReason(); got != "content_present" {
		t.Fatalf("content reason=%q want content_present", got)
	}
	reasoning, err := InspectEmptyToolTurn([]byte(emptyToolTurnSSEWithReasoningTokens("", "stop", true, 2)), request)
	if err != nil {
		t.Fatal(err)
	}
	if got := reasoning.EmptyToolTurnTerminalReason(); got != "reasoning_present" {
		t.Fatalf("reasoning reason=%q want reasoning_present", got)
	}
}

func TestInspectEmptyToolTurnRetainsAttemptUsage(t *testing.T) {
	observation, err := InspectEmptyToolTurn([]byte(emptyToolTurnSSEWithReasoningTokens("ok", "stop", true, 2)), []byte(`{"tools":[{"type":"function","function":{"name":"f"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	usage, ok := observation.Response["usage"].(map[string]any)
	if !ok || usage["prompt_tokens"] != float64(2) || usage["credit"] != 0.25 {
		t.Fatalf("usage=%#v", usage)
	}
	details := usage["completion_tokens_details"].(map[string]any)
	if details["reasoning_tokens"] != float64(2) {
		t.Fatalf("reasoning usage lost: %#v", usage)
	}
}
