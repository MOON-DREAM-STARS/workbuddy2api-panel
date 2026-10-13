package server

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestResponsesNamespaceHistoryAndChoice(t *testing.T) {
	raw := `{"model":"m","input":[{"type":"function_call","namespace":"alpha","name":"read","call_id":"c","arguments":"{}","status":"completed"},{"type":"function_call_output","call_id":"c","output":"ok"}],"tools":[{"type":"function","name":"read","parameters":{"type":"object"}},{"type":"namespace","name":"alpha","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]},{"type":"namespace","name":"beta","tools":[{"type":"function","name":"read","parameters":{"type":"object"}}]}],"tool_choice":{"type":"function","namespace":"alpha","name":"read"}}`
	req, err := parseResponsesRequest([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	var chat map[string]any
	json.Unmarshal(req.ChatBody, &chat)
	tools := chat["tools"].([]any)
	names := []string{}
	for _, raw := range tools {
		names = append(names, raw.(map[string]any)["function"].(map[string]any)["name"].(string))
	}
	if len(names) != 3 || names[0] != "read" || names[1] == names[2] || len(names[1]) > 64 {
		t.Fatalf("names=%v", names)
	}
	msg := chat["messages"].([]any)[0].(map[string]any)
	call := msg["tool_calls"].([]any)[0].(map[string]any)
	if call["function"].(map[string]any)["name"] != names[1] || chat["tool_choice"].(map[string]any)["function"].(map[string]any)["name"] != names[1] {
		t.Fatal("history/choice mismatch")
	}
	if strings.Contains(string(req.ChatBody), `"namespace"`) {
		t.Fatal("namespace leaked upstream")
	}
	if req.Tools[1].(map[string]any)["type"] != "namespace" {
		t.Fatal("public tool declaration lost")
	}
	for _, stream := range []bool{false, true} {
		var out strings.Builder
		encoder := &responseSSEEncoder{w: &out}
		state := newResponseState(req, stream, encoder)
		chunk := map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": float64(0), "id": "newcall", "function": map[string]any{"name": names[1], "arguments": "{}"}}}}, "finish_reason": "tool_calls"}}}
		if err := state.consumeChunk(chunk); err != nil {
			t.Fatal(err)
		}
		state.finish(true, nil)
		item := state.items[0]
		if item["namespace"] != "alpha" || item["name"] != "read" || state.response["status"] != "completed" {
			t.Fatalf("output=%v", item)
		}
		if stream {
			events := decodeResponsesTestEvents(t, out.String())
			terminals := 0
			for _, e := range events {
				if e["type"] == "response.completed" {
					terminals++
				}
				if e["type"] == "response.output_item.added" || e["type"] == "response.output_item.done" {
					i := e["item"].(map[string]any)
					if i["namespace"] != "alpha" || i["name"] != "read" {
						t.Fatalf("event lost identity: %v", e)
					}
				}
			}
			if terminals != 1 {
				t.Fatalf("terminals=%d", terminals)
			}
		}
	}
}
func TestResponsesNamespaceHistoryOptional(t *testing.T) {
	for _, field := range []string{``, `,"namespace":null`, `,"namespace":""`, `,"namespace":"functions"`} {
		body := fmt.Sprintf(`{"model":"m","input":[{"type":"function_call","call_id":"c","name":"read","arguments":"{}"%s},{"type":"function_call_output","call_id":"c","output":"ok"}]}`, field)
		if _, _, err := responsesToChat([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
}
func TestResponsesNamespaceValidation(t *testing.T) {
	cases := []struct{ raw, param string }{
		{`{"input":[{"type":"function_call","call_id":"c","name":"read","arguments":"{}","namespace":7}]}`, "input[0].namespace"},
		{`{"input":[{"type":"message","role":"user","content":"x","namespace":"a"}]}`, "input[0].namespace"},
		{`{"input":[{"type":"function_call","call_id":"c","name":"read","arguments":"{}","status":"unknown"}]}`, "input[0].status"},
		{`{"input":"x","tools":[{"type":"namespace","name":"a","tools":[{"type":"function","name":"x","mystery":1}]}]}`, "tools[0].tools[0].mystery"},
		{`{"input":"x","tools":[{"type":"namespace","name":"a","tools":[{"type":"namespace","name":"b","tools":[]}]}]}`, "tools[0].tools[0].type"},
		{`{"input":"x","tools":[{"type":"namespace","name":"a","tools":[{"type":"function","name":"x"},{"type":"function","name":"x"}]}]}`, "tools[0].tools[1].name"},
		{`{"input":"x","tool_choice":{"type":"function","name":"x","namespace":{}}}`, "tool_choice.namespace"},
	}
	for _, c := range cases {
		var body map[string]any
		json.Unmarshal([]byte(c.raw), &body)
		body["model"] = "m"
		raw, _ := json.Marshal(body)
		_, err := parseResponsesRequest(raw)
		v, ok := err.(*responsesValidationError)
		if !ok || v.Param != c.param {
			t.Errorf("param=%s err=%v", c.param, err)
		}
	}
	n := newResponseToolNames()
	wire, err := n.wire("alpha", "read", "tools")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = n.wire("", wire, "tools"); err == nil {
		t.Fatal("alias collision accepted")
	}
}
