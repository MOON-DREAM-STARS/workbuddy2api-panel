package server

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"
)

func TestResponsesRejectsOutputAfterFinish(t *testing.T) {
	const sentence = "This is the final sentence."
	const textFrame = `data: {"choices":[{"index":0,"delta":{"content":"` + sentence + `"}}]}` + "\n\n"
	const finish = `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
	for name, late := range map[string]string{
		"replayed text":  textFrame,
		"reasoning":      `data: {"choices":[{"index":0,"delta":{"reasoning_content":"late reasoning"}}]}` + "\n\n",
		"refusal":        `data: {"choices":[{"index":0,"delta":{"refusal":"late refusal"}}]}` + "\n\n",
		"tool arguments": `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"late","function":{"name":"read","arguments":"{}"}}]}}]}` + "\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			var wire strings.Builder
			renderer := &responsesRenderer{req: responsesRequest{Model: "m"}, encoder: &responseSSEEncoder{w: &wire}}
			upstream := io.NopCloser(strings.NewReader(textFrame + finish + late + "data: [DONE]\n\n"))
			out := renderer.RenderStream(context.Background(), upstream, ChatRenderMeta{})
			var text strings.Builder
			for _, event := range decodeResponsesTestEvents(t, wire.String()) {
				if event["type"] == "response.output_text.delta" {
					text.WriteString(event["delta"].(string))
				}
			}
			if text.String() != sentence {
				t.Fatalf("text forwarded after finish: %q", text.String())
			}
			if out.ReadErr == nil || out.RendererErr != nil || out.Response.(map[string]any)["status"] != "failed" {
				t.Fatalf("post-finish output was accepted: %+v", out)
			}
		})
	}
}

func TestResponsesFinishPreservesFinalDeltaAndTrailingUsage(t *testing.T) {
	// A literal '<' can be held by markup repair until [DONE]. It belongs to
	// the final delta and must not be mistaken for output after the finish.
	const raw = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a <\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}}\n\n" +
		"data: [DONE]\n\n"
	renderer := &responsesRenderer{req: responsesRequest{Model: "m"}}
	out := renderer.RenderNonStream(context.Background(), io.NopCloser(strings.NewReader(raw)), ChatRenderMeta{})
	if out.ReadErr != nil || !out.SawDone || out.Tokens != 3 || out.Prompt != 9 {
		t.Fatalf("valid finish or trailing usage was lost: %+v", out)
	}
	response := out.Response.(map[string]any)
	item := response["output"].([]any)[0].(map[string]any)
	part := item["content"].([]any)[0].(map[string]any)
	if part["text"] != "a <" || response["status"] != "completed" {
		t.Fatalf("final text or completion was lost: %#v", response)
	}
}

func TestResponsesPostFinishOutputClosesUpstreamWithoutDone(t *testing.T) {
	reader, writer := io.Pipe()
	body := &stopTestBody{PipeReader: reader, closed: make(chan struct{})}
	defer body.Close()
	defer writer.Close()
	renderer := &responsesRenderer{req: responsesRequest{Model: "m"}, encoder: &responseSSEEncoder{w: io.Discard}}
	result := make(chan ChatOutcome, 1)
	go func() { result <- renderer.RenderStream(context.Background(), body, ChatRenderMeta{}) }()
	const raw = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"unexpected continuation\"}}]}\n\n"
	go func() { _, _ = io.WriteString(writer, raw) }()
	select {
	case out := <-result:
		if out.ReadErr == nil || out.SawDone || out.Response.(map[string]any)["status"] != "failed" {
			t.Fatalf("open malformed stream did not fail: %+v", out)
		}
	case <-time.After(time.Second):
		t.Fatal("renderer waited for [DONE] after post-finish output")
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("post-finish output left the upstream open")
	}
}
