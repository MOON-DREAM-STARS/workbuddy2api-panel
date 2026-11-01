package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/livecfg"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

func TestResponsesStrictSubsetAndHistory(t *testing.T) {
	body, raw, err := responsesToChat([]byte(`{"model":"global:test","input":[{"role":"developer","content":[{"type":"input_text","text":"rules"}]},{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.png"}]},{"type":"function_call","call_id":"call_a","name":"read","arguments":"{}"},{"type":"function_call_output","call_id":"call_a","output":"done"}],"tools":[{"type":"function","name":"read","parameters":{"type":"object"},"strict":true}],"tool_choice":{"type":"function","name":"read"},"reasoning":{"effort":"high"},"prompt_cache_key":"stable","store":false}`))
	if err != nil {
		t.Fatal(err)
	}
	if raw["model"] != "global:test" {
		t.Fatalf("raw request lost model: %#v", raw)
	}
	var chat map[string]any
	if err := json.Unmarshal(body, &chat); err != nil {
		t.Fatal(err)
	}
	if chat["model"] != "global:test" || chat["reasoning_effort"] != "high" || chat["prompt_cache_key"] != "stable" {
		t.Fatalf("unexpected chat conversion: %s", body)
	}
	if _, ok := chat["input"]; ok {
		t.Fatal("Responses input leaked into Chat body")
	}
	if _, ok := chat["store"]; ok {
		t.Fatal("store leaked into Chat body")
	}
	messages := chat["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("messages=%d want 4: %s", len(messages), body)
	}
	image := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if image["image_url"].(map[string]any)["url"] != "https://example.com/a.png" {
		t.Fatalf("image conversion=%#v", image)
	}
	assistant := messages[2].(map[string]any)
	if len(assistant["tool_calls"].([]any)) != 1 {
		t.Fatalf("tool calls=%#v", assistant["tool_calls"])
	}

	for _, raw := range []string{
		`{"model":"m","input":"x","store":true}`,
		`{"model":"m","input":"x","previous_response_id":"resp_1"}`,
		`{"model":"m","input":"x","json_schema":{}}`,
		`{"model":"m","input":"x","text":{"format":{"type":"json_schema"}}}`,
		`{"model":"m","input":"x","unknown":true}`,
		`{"model":"m","input":[{"type":"function_call_output","call_id":"missing","output":"x"}]}`,
	} {
		if _, _, err := responsesToChat([]byte(raw)); err == nil {
			t.Fatalf("expected strict rejection: %s", raw)
		}
	}
}

func TestResponsesNonFunctionToolsAreDropped(t *testing.T) {
	body, _, err := responsesToChat([]byte(`{"model":"m","input":"hello","tools":[{"type":"namespace","name":"shell"},{"type":"function","name":"read","parameters":{"type":"object"}},{"type":"web_search"}]}`))
	if err != nil {
		t.Fatalf("mixed tool request should be accepted: %v", err)
	}
	var chat map[string]any
	if err := json.Unmarshal(body, &chat); err != nil {
		t.Fatal(err)
	}
	tools, ok := chat["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools=%#v want one function tool", chat["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Fatalf("tool=%#v", tool)
	}
	fn := tool["function"].(map[string]any)
	if fn["name"] != "read" {
		t.Fatalf("function=%#v", fn)
	}

	body, _, err = responsesToChat([]byte(`{"model":"m","input":"hello","tools":[{"type":"namespace","name":"shell"},{"type":"web_search"}]}`))
	if err != nil {
		t.Fatalf("all non-function tools should be accepted: %v", err)
	}
	chat = nil
	if err := json.Unmarshal(body, &chat); err != nil {
		t.Fatal(err)
	}
	if _, ok := chat["tools"]; ok {
		t.Fatalf("unsupported tools leaked into ChatBody: %#v", chat["tools"])
	}
}

func TestResponsesToolValidationAndRequiredChoice(t *testing.T) {
	for _, tc := range []struct {
		name  string
		body  string
		param string
	}{
		{name: "non-object tool", body: `{"model":"m","input":"hello","tools":[1]}`, param: "tools[0]"},
		{name: "missing type", body: `{"model":"m","input":"hello","tools":[{"name":"read"}]}`, param: "tools[0].type"},
		{name: "non-string type", body: `{"model":"m","input":"hello","tools":[{"type":1}]}`, param: "tools[0].type"},
		{name: "required without function", body: `{"model":"m","input":"hello","tools":[{"type":"namespace","name":"shell"}],"tool_choice":"required"}`, param: "tool_choice"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := responsesToChat([]byte(tc.body))
			if err == nil {
				t.Fatalf("expected rejection: %s", tc.body)
			}
			validation, ok := err.(*responsesValidationError)
			if !ok || validation.Param != tc.param {
				t.Fatalf("err=%T %#v want param %q", err, err, tc.param)
			}
		})
	}
}

func TestResponsesMessageIDIsAcceptedAndDropped(t *testing.T) {
	body, _, err := responsesToChat([]byte(`{"model":"m","input":[{"type":"message","id":"msg_1","role":"assistant","phase":"commentary","content":"prior"},{"role":"user","content":"continue"}]}`))
	if err != nil {
		t.Fatalf("message id should be accepted: %v", err)
	}
	if strings.Contains(string(body), "\"id\"") {
		t.Fatalf("message id leaked into ChatBody: %s", body)
	}
}

func TestResponsesMessageIDValidation(t *testing.T) {
	_, _, err := responsesToChat([]byte(`{"model":"m","input":[{"type":"message","id":123,"role":"assistant","content":"prior"}]}`))
	if err == nil {
		t.Fatal("non-string message id should be rejected")
	}
	validation, ok := err.(*responsesValidationError)
	if !ok || validation.Param != "input[0].id" {
		t.Fatalf("err=%T %#v", err, err)
	}
}

func TestResponsesAssistantMessagePhaseIsAcceptedAndDropped(t *testing.T) {
	for _, phase := range []string{"commentary", "final_answer"} {
		body, _, err := responsesToChat([]byte(fmt.Sprintf(
			`{"model":"m","input":[{"type":"message","role":"assistant","phase":%q,"content":"prior"},{"role":"user","content":"continue"}]}`, phase)))
		if err != nil {
			t.Fatalf("assistant phase %q should be accepted: %v", phase, err)
		}
		if strings.Contains(string(body), "\"phase\"") {
			t.Fatalf("phase leaked into ChatBody: %s", body)
		}
		var chat map[string]any
		if err := json.Unmarshal(body, &chat); err != nil {
			t.Fatal(err)
		}
		messages := chat["messages"].([]any)
		if messages[0].(map[string]any)["role"] != "assistant" {
			t.Fatalf("messages=%#v", messages)
		}
	}
}

func TestResponsesMessagePhaseValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		body  string
		param string
	}{
		{name: "non-string", body: `{"model":"m","input":[{"type":"message","role":"assistant","phase":1,"content":"prior"}]}`, param: "input[0].phase"},
		{name: "unsupported-value", body: `{"model":"m","input":[{"type":"message","role":"assistant","phase":"analysis","content":"prior"}]}`, param: "input[0].phase"},
		{name: "non-assistant", body: `{"model":"m","input":[{"type":"message","role":"user","phase":"commentary","content":"prior"}]}`, param: "input[0].phase"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := responsesToChat([]byte(tc.body))
			if err == nil {
				t.Fatalf("expected rejection: %s", tc.body)
			}
			validation, ok := err.(*responsesValidationError)
			if !ok || validation.Param != tc.param {
				t.Fatalf("err=%T %#v want param %q", err, err, tc.param)
			}
		})
	}
}

func TestResponsesReasoningContentReplayIsAcceptedAndDropped(t *testing.T) {
	body, _, err := responsesToChat([]byte(`{"model":"m","input":[{"type":"reasoning","id":"rs_1","status":"completed","summary":[],"encrypted_content":"opaque","content":[{"type":"reasoning_text","text":"opaque"}]},{"role":"user","content":"continue"}]}`))
	if err != nil {
		t.Fatalf("reasoning content replay should be accepted: %v", err)
	}
	if strings.Contains(string(body), "opaque") {
		t.Fatalf("reasoning content metadata leaked into ChatBody: %s", body)
	}
}

func TestResponsesFunctionToolIDsAreAcceptedAndDropped(t *testing.T) {
	body, _, err := responsesToChat([]byte(`{"model":"m","input":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{}"},{"type":"function_call_output","id":"fco_1","call_id":"call_1","output":"done"}]}`))
	if err != nil {
		t.Fatalf("function item ids should be accepted: %v", err)
	}
	if strings.Contains(string(body), "fc_1") || strings.Contains(string(body), "fco_1") {
		t.Fatalf("function item ids leaked into ChatBody: %s", body)
	}
}

func TestResponsesFunctionToolContentReplayIsAcceptedAndDropped(t *testing.T) {
	body, _, err := responsesToChat([]byte(`{"model":"m","input":[{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}","content":"opaque"},{"type":"function_call_output","call_id":"call_1","content":"done"}]}`))
	if err != nil {
		t.Fatalf("function tool content replay should be accepted: %v", err)
	}
	if strings.Contains(string(body), "opaque") {
		t.Fatalf("function_call content metadata leaked into ChatBody: %s", body)
	}
	var chat map[string]any
	if err := json.Unmarshal(body, &chat); err != nil {
		t.Fatal(err)
	}
	messages := chat["messages"].([]any)
	if len(messages) != 2 || messages[0].(map[string]any)["role"] != "assistant" || messages[1].(map[string]any)["role"] != "tool" {
		t.Fatalf("messages=%#v", messages)
	}
}

func TestResponsesReasoningEncryptedContentIsAcceptedAndDropped(t *testing.T) {
	for name, encrypted := range map[string]string{
		"string": `"opaque-reasoning"`,
		"null":   "null",
	} {
		t.Run(name, func(t *testing.T) {
			body, _, err := responsesToChat([]byte(fmt.Sprintf(
				`{"model":"m","input":[{"type":"reasoning","id":"rs_1","status":"completed","summary":[{"type":"summary_text","text":"plan"}],"encrypted_content":%s},{"role":"user","content":"continue"}]}`, encrypted)))
			if err != nil {
				t.Fatalf("reasoning encrypted_content should be accepted: %v", err)
			}
			if strings.Contains(string(body), "encrypted_content") {
				t.Fatalf("encrypted_content leaked into ChatBody: %s", body)
			}
			var chat map[string]any
			if err := json.Unmarshal(body, &chat); err != nil {
				t.Fatal(err)
			}
			messages := chat["messages"].([]any)
			if len(messages) != 2 {
				t.Fatalf("messages=%#v", messages)
			}
			assistant := messages[0].(map[string]any)
			if assistant["reasoning_content"] != "plan" {
				t.Fatalf("assistant=%#v", assistant)
			}
		})
	}
}

func TestResponsesReasoningEncryptedContentValidation(t *testing.T) {
	for _, value := range []string{`1`, `{}`, `[]`} {
		_, _, err := responsesToChat([]byte(fmt.Sprintf(
			`{"model":"m","input":[{"type":"reasoning","summary":[],"encrypted_content":%s}]}`, value)))
		if err == nil {
			t.Fatalf("encrypted_content=%s should be rejected", value)
		}
		validation, ok := err.(*responsesValidationError)
		if !ok || validation.Param != "input[0].encrypted_content" {
			t.Fatalf("err=%T %#v", err, err)
		}
	}

	body := `{"model":"m","input":[{"type":"message","role":"user","content":"hello","encrypted_content":"opaque"}]}`
	_, _, err := responsesToChat([]byte(body))
	if err == nil {
		t.Fatal("encrypted_content on a non-reasoning item should be rejected")
	}
	validation, ok := err.(*responsesValidationError)
	if !ok || validation.Param != "input[0].encrypted_content" {
		t.Fatalf("err=%T %#v", err, err)
	}
}
func TestResponsesClientMetadataIsCompatibilityOnly(t *testing.T) {
	req, err := parseResponsesRequest([]byte(`{"model":"m","input":"hello","client_metadata":{"source":"codex"},"include":["reasoning.encrypted_content"]}`))
	if err != nil {
		t.Fatalf("Codex compatibility fields should be accepted: %v", err)
	}
	if req.ClientMetadata == nil {
		t.Fatal("client_metadata was not retained in the parsed request")
	}
	if len(req.Include) != 1 || req.Include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include=%v want [reasoning.encrypted_content]", req.Include)
	}

	var chat map[string]any
	if err := json.Unmarshal(req.ChatBody, &chat); err != nil {
		t.Fatalf("decode ChatBody: %v", err)
	}
	if _, ok := chat["client_metadata"]; ok {
		t.Fatal("client_metadata must not be forwarded to the Chat upstream")
	}
	if _, ok := chat["include"]; ok {
		t.Fatal("include must not be forwarded to the Chat upstream")
	}
}

func TestResponsesCodexCompatibilityFields(t *testing.T) {
	req, err := parseResponsesRequest([]byte(`{"model":"m","input":"hello","instructions":"Follow the task","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"tool_choice":"auto","parallel_tool_calls":true,"reasoning":{"effort":"high","summary":"detailed","context":"all_turns"},"store":false,"stream":true,"stream_options":{"reasoning_summary_delivery":"sequential_cutoff"},"include":["reasoning.encrypted_content"],"prompt_cache_key":"codex-session","text":{"verbosity":"medium"},"client_metadata":{"x-codex-installation-id":"synthetic"},"service_tier":"default"}`))
	if err != nil {
		t.Fatalf("Codex Responses compatibility fields should be accepted: %v", err)
	}
	if req.ReasoningEffort != "high" || req.ReasoningSummary != "detailed" || !req.Stream {
		t.Fatalf("unexpected parsed Codex request: %+v", req)
	}

	var chat map[string]any
	if err := json.Unmarshal(req.ChatBody, &chat); err != nil {
		t.Fatalf("decode ChatBody: %v", err)
	}
	for _, key := range []string{"stream_options", "service_tier", "client_metadata", "include"} {
		if _, exists := chat[key]; exists {
			t.Errorf("Responses compatibility field %q leaked into ChatBody", key)
		}
	}
	if strings.Contains(string(req.ChatBody), `"all_turns"`) {
		t.Fatal("reasoning.context leaked into ChatBody")
	}

	prepared := upstream.PrepareBodyOptWithEffortsAndDefault(req.ChatBody, false, nil, nil)
	var outbound map[string]any
	if err := json.Unmarshal(prepared, &outbound); err != nil {
		t.Fatalf("decode prepared Chat request: %v", err)
	}
	streamOptions, ok := outbound["stream_options"].(map[string]any)
	if !ok || streamOptions["include_usage"] != true {
		t.Fatalf("outbound stream_options=%#v want include_usage=true", outbound["stream_options"])
	}
}

func TestResponsesCodexCompatibilityFieldValidation(t *testing.T) {
	cases := []struct {
		name  string
		body  string
		param string
	}{
		{
			name:  "stream options must be object",
			body:  `{"model":"m","input":"hello","stream_options":true}`,
			param: "stream_options",
		},
		{
			name:  "unsupported stream option",
			body:  `{"model":"m","input":"hello","stream_options":{"include_usage":true}}`,
			param: "stream_options.include_usage",
		},
		{
			name:  "unsupported summary delivery",
			body:  `{"model":"m","input":"hello","stream_options":{"reasoning_summary_delivery":"concurrent"}}`,
			param: "stream_options.reasoning_summary_delivery",
		},
		{
			name:  "non-default service tier",
			body:  `{"model":"m","input":"hello","service_tier":"priority"}`,
			param: "service_tier",
		},
		{
			name:  "unsupported reasoning context",
			body:  `{"model":"m","input":"hello","reasoning":{"context":"current_turn"}}`,
			param: "reasoning.context",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseResponsesRequest([]byte(tc.body))
			validation, ok := err.(*responsesValidationError)
			if !ok || validation.Param != tc.param {
				t.Fatalf("error=%T %#v want param %q", err, err, tc.param)
			}
		})
	}
}

func TestResponsesTextFormatCompatibility(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		wantFormatType string
	}{
		{
			name: "text omitted",
			body: `{"model":"m","input":"hello"}`,
		},
		{
			name: "empty text defaults to plain text",
			body: `{"model":"m","input":"hello","text":{}}`,
		},
		{
			name: "verbosity defaults to plain text",
			body: `{"model":"m","input":"hello","text":{"verbosity":"medium"}}`,
		},
		{
			name:           "explicit text format",
			body:           `{"model":"m","input":"hello","text":{"format":{"type":"text"}}}`,
			wantFormatType: "text",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body, _, err := responsesToChat([]byte(tc.body))
			if err != nil {
				t.Fatalf("responsesToChat() error = %v", err)
			}

			var chat map[string]any
			if err := json.Unmarshal(body, &chat); err != nil {
				t.Fatalf("decode ChatBody: %v", err)
			}
			format, exists := chat["response_format"]
			if tc.wantFormatType == "" {
				if exists {
					t.Fatalf("response_format=%#v, want omitted", format)
				}
				if strings.Contains(string(body), "verbosity") {
					t.Fatalf("verbosity leaked into ChatBody: %s", body)
				}
				return
			}

			responseFormat, ok := format.(map[string]any)
			if !ok || responseFormat["type"] != tc.wantFormatType {
				t.Fatalf("response_format=%#v, want type %q", format, tc.wantFormatType)
			}
		})
	}
}

func TestResponsesTextFormatValidation(t *testing.T) {
	tests := []struct {
		name  string
		body  string
		param string
	}{
		{
			name:  "text is not an object",
			body:  `{"model":"m","input":"hello","text":"plain"}`,
			param: "text",
		},
		{
			name:  "verbosity is not a string",
			body:  `{"model":"m","input":"hello","text":{"verbosity":1}}`,
			param: "text.verbosity",
		},
		{
			name:  "verbosity is unsupported",
			body:  `{"model":"m","input":"hello","text":{"verbosity":"verbose"}}`,
			param: "text.verbosity",
		},
		{
			name:  "format is not an object",
			body:  `{"model":"m","input":"hello","text":{"format":"text"}}`,
			param: "text.format",
		},
		{
			name:  "format type is unsupported",
			body:  `{"model":"m","input":"hello","text":{"format":{"type":"json_schema"}}}`,
			param: "text.format.type",
		},
		{
			name:  "format has unsupported field",
			body:  `{"model":"m","input":"hello","text":{"format":{"type":"text","name":"answer"}}}`,
			param: "text.format.name",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := responsesToChat([]byte(tc.body))
			if err == nil {
				t.Fatalf("expected rejection: %s", tc.body)
			}
			validation, ok := err.(*responsesValidationError)
			if !ok || validation.Param != tc.param {
				t.Fatalf("err=%T %#v want param %q", err, err, tc.param)
			}
		})
	}
}

func TestResponsesIncludeRejectsInvalidValues(t *testing.T) {
	cases := []struct {
		name  string
		body  string
		param string
	}{
		{name: "not an array", body: `{"model":"m","input":"hello","include":"reasoning.encrypted_content"}`, param: "include"},
		{name: "non-string item", body: `{"model":"m","input":"hello","include":[1]}`, param: "include[0]"},
		{name: "unsupported marker", body: `{"model":"m","input":"hello","include":["web_search_call.action.sources"]}`, param: "include[0]"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseResponsesRequest([]byte(tc.body))
			if err == nil {
				t.Fatalf("expected rejection for %s", tc.body)
			}
			validation, ok := err.(*responsesValidationError)
			if !ok {
				t.Fatalf("error=%T %v want *responsesValidationError", err, err)
			}
			if validation.Param != tc.param {
				t.Fatalf("param=%q want %q", validation.Param, tc.param)
			}
		})
	}
}

func TestResponsesReasoningSummaryParserAndChatBody(t *testing.T) {
	for _, summary := range []string{"auto", "concise", "detailed"} {
		t.Run(summary, func(t *testing.T) {
			body, _, err := responsesToChat([]byte(fmt.Sprintf(
				`{"model":"m","input":"hello","reasoning":{"effort":"high","summary":%q}}`, summary)))
			if err != nil {
				t.Fatal(err)
			}
			var chat map[string]any
			if err := json.Unmarshal(body, &chat); err != nil {
				t.Fatal(err)
			}
			if chat["reasoning_effort"] != "high" || chat["reasoning_summary"] != summary {
				t.Fatalf("chat=%#v", chat)
			}
		})
	}
	for _, tc := range []struct {
		body  string
		param string
	}{
		{body: `{"model":"m","input":"hello","reasoning":{"summary":1}}`, param: "reasoning.summary"},
		{body: `{"model":"m","input":"hello","reasoning":{"summary":"brief"}}`, param: "reasoning.summary"},
		{body: `{"model":"m","input":"hello","reasoning":{"mode":"x"}}`, param: "reasoning.mode"},
	} {
		_, _, err := responsesToChat([]byte(tc.body))
		if err == nil {
			t.Fatalf("expected rejection: %s", tc.body)
		}
		validation, ok := err.(*responsesValidationError)
		if !ok || validation.Param != tc.param {
			t.Fatalf("err=%T %#v want param %q", err, err, tc.param)
		}
	}
}

func TestResponsesReasoningSummaryNonStreamingOutput(t *testing.T) {
	wire := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":"think "}}]}`,
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":"more","content":"answer"}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	state := newResponseState(responsesRequest{Model: "m", ReasoningSummary: "auto", ParallelTools: true}, false, nil)
	decoded := decodeChatSSE(strings.NewReader(wire), state.consumeChunk)
	state.finish(decoded.SawDone, decoded.Err)
	if decoded.Err != nil || state.response["status"] != "completed" {
		t.Fatalf("decoded=%#v response=%#v", decoded, state.response)
	}
	output := state.response["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output=%#v", output)
	}
	reasoning := output[0].(map[string]any)
	if reasoning["type"] != "reasoning" || reasoning["status"] != "completed" {
		t.Fatalf("reasoning=%#v", reasoning)
	}
	parts := reasoning["summary"].([]any)
	if parts[0].(map[string]any)["text"] != "think more" {
		t.Fatalf("summary=%#v", parts)
	}
	message := output[1].(map[string]any)
	if message["content"].([]any)[0].(map[string]any)["text"] != "answer" {
		t.Fatalf("message=%#v", message)
	}
}

func TestResponsesReasoningSummaryStreamingEvents(t *testing.T) {
	wire := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":"think","content":""}}]}`,
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":"more","content":"answer"}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	var out strings.Builder
	encoder := &responseSSEEncoder{w: &out}
	state := newResponseState(responsesRequest{Model: "m", ReasoningSummary: "auto", ParallelTools: true}, true, encoder)
	state.start()
	decoded := decodeChatSSE(strings.NewReader(wire), state.consumeChunk)
	state.finish(decoded.SawDone, decoded.Err)
	events := decodeResponsesTestEvents(t, out.String())
	want := []string{
		"response.output_item.added",
		"response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta",
		"response.content_part.added",
		"response.reasoning_summary_text.done",
		"response.reasoning_summary_part.done",
		"response.output_item.done",
		"response.completed",
	}
	for _, typ := range want {
		found := false
		for _, event := range events {
			if event["type"] == typ {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing event %s: %#v", typ, events)
		}
	}
	reasoningAdded := 0
	for _, event := range events {
		if event["type"] == "response.output_item.added" {
			item := event["item"].(map[string]any)
			if item["type"] == "reasoning" {
				reasoningAdded++
			}
		}
	}
	if reasoningAdded != 1 {
		t.Fatalf("reasoning item count=%d events=%#v", reasoningAdded, events)
	}
	for i, event := range events {
		if event["sequence_number"] != float64(i) {
			t.Fatalf("sequence=%#v want %d", event, i)
		}
	}
}

func TestResponsesReasoningContentHiddenWithoutSummary(t *testing.T) {
	state := newResponseState(responsesRequest{Model: "m", ParallelTools: true}, false, nil)
	if err := state.consumeChunk(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "secret", "content": "answer"}}}}); err != nil {
		t.Fatal(err)
	}
	if len(state.items) != 1 || state.items[0]["type"] != "message" {
		t.Fatalf("items=%#v", state.items)
	}
}

func TestResponsesReasoningHistoryToChat(t *testing.T) {
	body, _, err := responsesToChat([]byte(`{"model":"m","input":[{"type":"reasoning","id":"rs_1","status":"completed","summary":[{"type":"summary_text","text":"plan"}]},{"role":"user","content":"continue"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var chat map[string]any
	if err := json.Unmarshal(body, &chat); err != nil {
		t.Fatal(err)
	}
	messages := chat["messages"].([]any)
	assistant := messages[0].(map[string]any)
	if assistant["role"] != "assistant" || assistant["reasoning_content"] != "plan" {
		t.Fatalf("messages=%#v", messages)
	}
}

func TestResponsesHandlerNonStreamingPreservesFinishAndUsage(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "responses", AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true})
	r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello","store":false}`))
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response["object"] != "response" || response["status"] != "completed" {
		t.Fatalf("response=%#v", response)
	}
	output := response["output"].([]any)
	item := output[0].(map[string]any)
	part := item["content"].([]any)[0].(map[string]any)
	if part["text"] != "你好" {
		t.Fatalf("output=%#v", output)
	}
	usage := response["usage"].(map[string]any)
	if usage["input_tokens"] != float64(1) || usage["output_tokens"] != float64(1) || usage["total_tokens"] != float64(2) {
		t.Fatalf("usage=%#v", usage)
	}
}

// TestResponsesEndpointGate 覆盖原生 Responses 接口开关：缺省关闭时（无论是否
// 携带鉴权）鉴权前统一 404 endpoint_disabled；经 livecfg 热开启后进入鉴权路径；
// 关回后恢复 404。上游在任何关闭态都不得被调用。
func TestResponsesEndpointGate(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		t.Fatal("upstream must not be called while /v1/responses is disabled")
		return 0, "", false
	})

	decodeCode := func(t *testing.T, body string) string {
		t.Helper()
		var payload map[string]any
		if err := json.Unmarshal([]byte(body), &payload); err != nil {
			t.Fatalf("unmarshal error body: %v (%s)", err, body)
		}
		errorBody, ok := payload["error"].(map[string]any)
		if !ok {
			t.Fatalf("error object missing: %s", body)
		}
		code, _ := errorBody["code"].(string)
		return code
	}

	// 缺省关闭：无鉴权与带鉴权请求都拿到同一 404（闸门在鉴权之前）。
	h := NewHandler(Config{Pool: testPoolWith(), Upstream: up, APIKey: "key"})
	for name, withAuth := range map[string]bool{"without_auth": false, "with_auth": true} {
		t.Run("disabled_"+name, func(t *testing.T) {
			r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello","store":false}`))
			if withAuth {
				r.Header.Set("Authorization", "Bearer key")
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != http.StatusNotFound {
				t.Fatalf("status=%d body=%s want 404", w.Code, w.Body.String())
			}
			if code := decodeCode(t, w.Body.String()); code != "endpoint_disabled" {
				t.Fatalf("error code=%q want endpoint_disabled", code)
			}
		})
	}

	// 热切换：开启后进入鉴权（错 key → 401），正确 key 进入执行层（空池 →
	// 503 no_healthy_account）；关回后即便带正确 key 也是 404。
	live := livecfg.New(livecfg.Snapshot{APIKey: "key", ResponsesEnabled: false})
	hl := NewHandler(Config{Pool: testPoolWith(), Upstream: up, APIKey: "key", Live: live})
	live.Store(livecfg.Snapshot{APIKey: "key", ResponsesEnabled: true})
	req := func(withAuth bool) *http.Request {
		r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello","store":false}`))
		if withAuth {
			r.Header.Set("Authorization", "Bearer key")
		}
		return r
	}
	w := httptest.NewRecorder()
	hl.ServeHTTP(w, req(false))
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("enabled without auth status=%d body=%s want 401", w.Code, w.Body.String())
	}
	w = httptest.NewRecorder()
	hl.ServeHTTP(w, req(true))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("enabled with auth status=%d body=%s want 503", w.Code, w.Body.String())
	}
	if code := decodeCode(t, w.Body.String()); code != "no_healthy_account" {
		t.Fatalf("enabled error code=%q want no_healthy_account", code)
	}
	live.Store(livecfg.Snapshot{APIKey: "key", ResponsesEnabled: false})
	w = httptest.NewRecorder()
	hl.ServeHTTP(w, req(true))
	if w.Code != http.StatusNotFound {
		t.Fatalf("re-disabled status=%d body=%s want 404", w.Code, w.Body.String())
	}
}

func TestResponsesUnsupportedUpstreamFieldFailsFastWithSafeDiagnostic(t *testing.T) {
	calls := 0
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		calls++
		return http.StatusBadRequest, `{"error":{"message":"field is not supported (upstream-only-canary)","type":"invalid_request_error","param":"max_completion_tokens","code":"unsupported_value"}}`, false
	})
	p := testPoolWith(
		&auth.Auth{UID: "responses-unsupported-a", AccessToken: "at-a", ExpiresAt: time.Now().Add(time.Hour).Unix()},
		&auth.Auth{UID: "responses-unsupported-b", AccessToken: "at-b", ExpiresAt: time.Now().Add(time.Hour).Unix()},
	)
	h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true})
	body := `{"model":"glm-5.2","input":"hello","max_output_tokens":64,"reasoning":{"effort":"medium"},"text":{"format":{"type":"text"}},"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}}],"tool_choice":"auto"}`
	r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s want 400", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	errorBody := payload["error"].(map[string]any)
	if errorBody["code"] != "unsupported_value" || errorBody["param"] != "max_output_tokens" {
		t.Fatalf("error=%#v want unsupported_value with mapped Responses param", errorBody)
	}
	hint, ok := errorBody["gateway_hint"].(string)
	if !ok {
		t.Fatalf("gateway_hint missing: %#v", errorBody)
	}
	for _, section := range []string{"input/history", "reasoning", "text.format", "tools", "tool_choice", "output limits"} {
		if !strings.Contains(hint, section) {
			t.Errorf("gateway_hint=%q missing present section %q", hint, section)
		}
	}
	for _, raw := range []string{"upstream-only-canary", "field is not supported"} {
		if strings.Contains(w.Body.String(), raw) {
			t.Errorf("upstream details leaked in response: %s", w.Body.String())
		}
	}
	if calls != 1 {
		t.Fatalf("upstream calls=%d want 1 (request-level rejection must not rotate accounts)", calls)
	}
	for _, uid := range []string{"responses-unsupported-a", "responses-unsupported-b"} {
		status, ok := p.Status(uid)
		if !ok || status.Cooling || status.Disabled || status.ErrTotal != 0 || status.BreakerFails != 0 {
			t.Errorf("unsupported request must not penalize account %s: %+v", uid, status)
		}
	}
}

func TestResponsesNoHealthyAccountIsNotBlamedOnRequestFields(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		t.Fatal("upstream must not be called with an empty account pool")
		return 0, "", false
	})
	h := NewHandler(Config{Pool: testPoolWith(), Upstream: up, APIKey: "key", ResponsesEnabled: true})
	r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello","reasoning":{"effort":"medium"}}`))
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s want 503", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	errorBody := payload["error"].(map[string]any)
	if errorBody["code"] != "no_healthy_account" {
		t.Fatalf("error=%#v want no_healthy_account", errorBody)
	}
	if _, exists := errorBody["param"]; exists {
		t.Fatalf("account availability error must not blame a request field: %#v", errorBody)
	}
	if strings.Contains(w.Body.String(), "unsupported_value") || strings.Contains(w.Body.String(), "request sections") {
		t.Fatalf("account availability error was mislabeled as a request-field error: %s", w.Body.String())
	}
}

func TestResponsesStreamingStateMachine(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"hello "}}]}`,
		`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"world"}}]}`,
		`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read","arguments":"{\"a\":"}},{"index":1,"id":"call_b","function":{"name":"list","arguments":"{}"}}]}}]}`,
		`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, stream, true })
	p := testPoolWith(&auth.Auth{UID: "responses-stream", AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true})
	r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello","stream":true,"store":false}`))
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	events := decodeResponsesTestEvents(t, w.Body.String())
	if events[len(events)-1]["type"] != "response.completed" {
		t.Fatalf("terminal=%#v", events[len(events)-1])
	}
	term := events[len(events)-1]["response"].(map[string]any)
	if term["status"] != "completed" {
		t.Fatalf("terminal response=%#v", term)
	}
	output := term["output"].([]any)
	if len(output) != 3 {
		t.Fatalf("output=%#v", output)
	}
	text := output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	if text != "hello world" {
		t.Fatalf("text=%v", text)
	}
	if output[1].(map[string]any)["call_id"] != "call_a" || output[2].(map[string]any)["call_id"] != "call_b" {
		t.Fatalf("tool identity=%#v", output)
	}
	if strings.Contains(w.Body.String(), "[DONE]") {
		t.Fatal("Chat [DONE] leaked into Responses stream")
	}
	for i, event := range events {
		if event["sequence_number"] != float64(i) {
			t.Fatalf("sequence=%#v want %d", event, i)
		}
	}
}

func TestResponsesMissingDoneAndTruncatedToolAreNotCompleted(t *testing.T) {
	for name, stream := range map[string]string{
		"missing_done": `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read","arguments":"{\"a\":"}}]},"finish_reason":"tool_calls"}]}` + "\n\n",
		"length":       `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read","arguments":"{\"a\":"}}]},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n`,
	} {
		t.Run(name, func(t *testing.T) {
			stream = strings.ReplaceAll(stream, "\\n", "\n")
			up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, stream, true })
			p := testPoolWith(&auth.Auth{UID: "responses-fail-" + name, AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
			h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "key", ResponsesEnabled: true})
			r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello","stream":false}`))
			r.Header.Set("Authorization", "Bearer key")
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
			}
			var response map[string]any
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if name == "missing_done" && response["status"] != "failed" {
				t.Fatalf("missing DONE status=%v response=%#v", response["status"], response)
			}
			if name == "length" && response["status"] != "incomplete" {
				t.Fatalf("length status=%v response=%#v", response["status"], response)
			}
			item := response["output"].([]any)[0].(map[string]any)
			if item["status"] == "completed" {
				t.Fatalf("truncated tool falsely completed: %#v", item)
			}
		})
	}
}

func decodeResponsesTestEvents(t *testing.T, raw string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range strings.Split(raw, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if len(events) == 0 {
		t.Fatal("no response events")
	}
	return events
}

func TestResponsesDecoderOneByteReadAndCRLF(t *testing.T) {
	wire := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hé\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	state := newResponseState(responsesRequest{Model: "m", ParallelTools: true}, false, nil)
	decoded := decodeChatSSE(&oneByteReader{data: []byte(wire)}, state.consumeChunk)
	state.finish(decoded.SawDone, decoded.Err)
	if decoded.Err != nil || !decoded.SawDone {
		t.Fatalf("decode=%#v", decoded)
	}
	if state.response["status"] != "completed" {
		t.Fatalf("status=%v response=%#v", state.response["status"], state.response)
	}
	text := state.items[0]["content"].([]any)[0].(map[string]any)["text"]
	if text != "hé" {
		t.Fatalf("text=%v", text)
	}
}

type oneByteReader struct{ data []byte }

func (r *oneByteReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	p[0] = r.data[0]
	r.data = r.data[1:]
	return 1, nil
}

func TestResponsesEmptyFinishReasonIsNonTerminal(t *testing.T) {
	wire := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"\"}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	state := newResponseState(responsesRequest{Model: "m", ParallelTools: true}, false, nil)
	decoded := decodeChatSSE(strings.NewReader(wire), state.consumeChunk)
	state.finish(decoded.SawDone, decoded.Err)
	if decoded.Err != nil || !decoded.SawDone {
		t.Fatalf("decode=%#v", decoded)
	}
	if state.response["status"] != "completed" {
		t.Fatalf("status=%v response=%#v", state.response["status"], state.response)
	}
}

func TestResponsesDecoderEmptyAndInvalidFramesFail(t *testing.T) {
	for name, wire := range map[string]string{
		"empty":   "",
		"comment": ": keep-alive\n\n",
		"invalid": "data: {not-json}\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			state := newResponseState(responsesRequest{Model: "m", ParallelTools: true}, false, nil)
			decoded := decodeChatSSE(strings.NewReader(wire), state.consumeChunk)
			state.finish(decoded.SawDone, decoded.Err)
			if state.response["status"] != "failed" {
				t.Fatalf("status=%v response=%#v decoded=%#v", state.response["status"], state.response, decoded)
			}
		})
	}
}

func TestResponsesContentFilterIsIncomplete(t *testing.T) {
	wire := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"content_filter\"}]}\n\n" +
		"data: [DONE]\n\n"
	state := newResponseState(responsesRequest{Model: "m", ParallelTools: true}, false, nil)
	decoded := decodeChatSSE(strings.NewReader(wire), state.consumeChunk)
	state.finish(decoded.SawDone, decoded.Err)
	if state.response["status"] != "incomplete" {
		t.Fatalf("status=%v response=%#v", state.response["status"], state.response)
	}
	if state.response["incomplete_details"].(map[string]any)["reason"] != "content_filter" {
		t.Fatalf("details=%#v", state.response["incomplete_details"])
	}
}

func TestResponsesLateToolIDDoesNotRewritePublicID(t *testing.T) {
	state := newResponseState(responsesRequest{Model: "m", ParallelTools: true}, false, nil)
	firstCall := map[string]any{
		"index":    float64(0),
		"function": map[string]any{"name": "read", "arguments": "{\"a\":"},
	}
	firstChoice := map[string]any{
		"index": float64(0),
		"delta": map[string]any{"tool_calls": []any{firstCall}},
	}
	if err := state.consumeChunk(map[string]any{"choices": []any{firstChoice}}); err != nil {
		t.Fatal(err)
	}
	publicID := state.items[0]["call_id"]
	lateCall := map[string]any{
		"id":       "late-call",
		"function": map[string]any{"arguments": "1}"},
	}
	lateChoice := map[string]any{
		"index": float64(0),
		"delta": map[string]any{"tool_calls": []any{lateCall}},
	}
	if err := state.consumeChunk(map[string]any{"choices": []any{lateChoice}}); err != nil {
		t.Fatal(err)
	}
	if state.items[0]["call_id"] != publicID {
		t.Fatalf("public call_id changed from %v to %v", publicID, state.items[0]["call_id"])
	}
}

func TestResponsesHiddenModelRejectedWithoutUpstreamCall(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) {
		t.Error("hidden model must not reach upstream")
		return 200, sseOK, true
	})
	p := testPoolWith(&auth.Auth{UID: "hidden", AccessToken: "at", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	h := NewHandler(Config{Pool: p, Upstream: up, ResponsesEnabled: true, ModelsHidden: []string{"glm-5.2"}})
	request := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"glm-5.2","input":"hello"}`))
	response := httptest.NewRecorder()
	h.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "model_unavailable") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}
