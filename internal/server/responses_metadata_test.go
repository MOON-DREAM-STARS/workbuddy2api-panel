package server

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestResponsesHistoryMetadataIsClientOnly(t *testing.T) {
	metadata := map[string]any{"content_item_kinds": []any{"history-meta-sentinel"}}
	items := []any{
		map[string]any{"type": "reasoning", "summary": []any{map[string]any{"type": "summary_text", "text": "previous plan"}}, "internal_chat_message_metadata_passthrough": metadata},
		map[string]any{"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}", "internal_chat_message_metadata_passthrough": metadata},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "done", "internal_chat_message_metadata_passthrough": metadata},
	}
	raw, _ := json.Marshal(map[string]any{"model": "m", "input": items})
	req, err := parseResponsesRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(req.ChatBody), "metadata_passthrough") || strings.Contains(string(req.ChatBody), "history-meta-sentinel") {
		t.Fatalf("metadata leaked into Chat: %s", req.ChatBody)
	}
	codec, _ := testCompactionCodec(t)
	capsule, err := codec.seal("saved checkpoint")
	if err != nil {
		t.Fatal(err)
	}
	for _, typ := range []string{"reasoning", "compaction"} {
		item := map[string]any{"type": typ, "internal_chat_message_metadata_passthrough": metadata}
		if typ == "compaction" {
			item["encrypted_content"] = capsule
		} else {
			item["summary"] = []any{map[string]any{"type": "summary_text", "text": "plan"}}
		}
		raw, _ = json.Marshal(map[string]any{"model": "m", "input": []any{item}})
		if _, err := parseResponsesRequestWithCompaction(raw, codec); err != nil {
			t.Fatalf("%s metadata: %v", typ, err)
		}
		item["internal_chat_message_metadata_passthrough"] = true
		raw, _ = json.Marshal(map[string]any{"model": "m", "input": []any{item}})
		if _, err := parseResponsesRequestWithCompaction(raw, codec); err == nil {
			t.Fatalf("invalid %s metadata accepted", typ)
		}
	}
}
