package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

type stopTestBody struct {
	*io.PipeReader
	closed chan struct{}
	once   sync.Once
}

func (b *stopTestBody) Close() error {
	b.once.Do(func() { close(b.closed) })
	return b.PipeReader.Close()
}

type stopTestWriter struct{ err error }

func (w stopTestWriter) Write(p []byte) (int, error) {
	if strings.Contains(string(p), "event: response.output_text.delta\n") {
		return 0, w.err
	}
	return len(p), nil
}

// A repeating upstream never reaches EOF or an idle timeout on its own.
func TestResponsesStopsOnDownstreamWriteFailure(t *testing.T) {
	reader, writer := io.Pipe()
	body := &stopTestBody{PipeReader: reader, closed: make(chan struct{})}
	defer body.Close()
	defer writer.Close()
	errClient := errors.New("client stopped reading")
	renderer := &responsesRenderer{req: responsesRequest{Model: "m"}, encoder: &responseSSEEncoder{w: stopTestWriter{err: errClient}}}
	p := testPoolWith(&auth.Auth{UID: "stop-test", AccessToken: "synthetic", ExpiresAt: time.Now().Add(time.Hour).Unix()})
	p.SetMaxInFlight(1)
	calls := 0
	up := &upstream.Client{ChatBaseCN: "https://fake.example", HTTP: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: body}, nil
	})}}
	h := NewHandler(Config{Pool: p, Upstream: up, MaxRotate: 3})
	done := make(chan ChatOutcome, 1)
	go func() {
		out, _ := h.executeChat(ChatExecutionRequest{Context: context.Background(), Model: "m", Stream: true, Body: []byte(`{"model":"m","stream":true,"messages":[{"role":"user","content":"synthetic"}]}`)}, renderer)
		done <- out
	}()
	producerDone := make(chan struct{})
	go func() {
		defer close(producerDone)
		for {
			if _, err := io.WriteString(writer, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"重复一句话。\"}}]}\n\n"); err != nil {
				return
			}
		}
	}()
	select {
	case out := <-done:
		if !errors.Is(out.RendererErr, errClient) || out.ReadErr != nil {
			t.Fatalf("write failure must be an interruption: %+v", out)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("renderer kept reading upstream after downstream write failed")
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("upstream body was not closed")
	}
	select {
	case <-producerDone:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("repeating upstream producer did not stop")
	}
	status, _ := p.Status("stop-test")
	if calls != 1 || status.InFlight != 0 || status.ErrTotal != 0 || status.SuccessCount != 0 {
		t.Fatalf("interruption retried, leaked a lease, or changed account health: calls=%d status=%+v", calls, status)
	}
}

func TestResponsesRepeatedTextIsNotDuplicatedByAdapter(t *testing.T) {
	chunk := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"重复一句话。\"}}]}\n\n"
	raw := strings.Repeat(chunk, 20) + "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":20,\"total_tokens\":27}}\n\ndata: [DONE]\n\n"
	renderer := &responsesRenderer{req: responsesRequest{Model: "m"}, encoder: &responseSSEEncoder{w: io.Discard}}
	out := renderer.RenderStream(context.Background(), io.NopCloser(strings.NewReader(raw)), ChatRenderMeta{})
	if out.ReadErr != nil || out.RendererErr != nil || !out.SawDone || out.Tokens != 20 || out.Prompt != 7 {
		t.Fatalf("normal repeated stream or usage changed: %+v", out)
	}
	response := out.Response.(map[string]any)
	item := response["output"].([]any)[0].(map[string]any)
	content := item["content"].([]any)[0].(map[string]any)
	if content["text"] != strings.Repeat("重复一句话。", 20) || response["status"] != "completed" {
		t.Fatalf("adapter duplicated or suppressed upstream text: %#v", response)
	}
}

func TestResponsesStopsOnContextCancellation(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "nonstream", true: "stream"}[stream], func(t *testing.T) {
			reader, writer := io.Pipe()
			body := &stopTestBody{PipeReader: reader, closed: make(chan struct{})}
			defer body.Close()
			defer writer.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			renderer := &responsesRenderer{req: responsesRequest{Model: "m"}}
			if stream {
				renderer.encoder = &responseSSEEncoder{w: io.Discard}
			}
			done := make(chan ChatOutcome, 1)
			go func() {
				if stream {
					done <- renderer.RenderStream(ctx, body, ChatRenderMeta{})
				} else {
					done <- renderer.RenderNonStream(ctx, body, ChatRenderMeta{})
				}
			}()
			if _, err := io.WriteString(writer, "data: {\"choices\":[{\"delta\":{\"content\":\"one chunk\"}}]}\n\n"); err != nil {
				t.Fatal(err)
			}
			cancel()
			select {
			case out := <-done:
				if !errors.Is(out.RendererErr, context.Canceled) || out.ReadErr != nil {
					t.Fatalf("cancel not classified as interruption: %+v", out)
				}
			case <-time.After(500 * time.Millisecond):
				t.Fatal("renderer ignored cancellation while upstream stayed open")
			}
			select {
			case <-body.closed:
			default:
				t.Fatal("upstream not closed")
			}
		})
	}
}

func TestResponsesParseFailureClosesUpstream(t *testing.T) {
	reader, writer := io.Pipe()
	body := &stopTestBody{PipeReader: reader, closed: make(chan struct{})}
	defer body.Close()
	defer writer.Close()
	renderer := &responsesRenderer{req: responsesRequest{Model: "m"}, encoder: &responseSSEEncoder{w: io.Discard}}
	done := make(chan ChatOutcome, 1)
	go func() { done <- renderer.RenderStream(context.Background(), body, ChatRenderMeta{}) }()
	if _, err := io.WriteString(writer, "data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"unsupported second choice\"}}]}\n\n"); err != nil {
		t.Fatal(err)
	}
	select {
	case out := <-done:
		if out.ReadErr == nil || out.RendererErr != nil {
			t.Fatalf("parse failure classification: %+v", out)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("renderer did not finish after parse error")
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("parse error left upstream open")
	}
}
