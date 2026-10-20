package upstream

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// EmptyToolTurnObservation contains only response facts needed to decide
// whether an opt-in retry is safe. Response is the ordinary
// aggregate after the existing DSML repair has been applied.
type EmptyToolTurnObservation struct {
	SawDone           bool
	SawFinishReason   bool
	FinishReason      string
	SawErrorFrame     bool
	ValidToolCalls    int
	RepairedToolCalls int
	ContentChars      int
	ReasoningChars    int
	ReasoningTokens   int64
	Response          map[string]any
}

// InspectEmptyToolTurn observes the original SSE terminators before Aggregate
// can normalize them, then applies the same aggregate DSML repair used by the
// Chat non-stream path. A missing [DONE] or finish_reason is never inferred.
func InspectEmptyToolTurn(raw, requestBody []byte) (EmptyToolTurnObservation, error) {
	var observation EmptyToolTurnObservation
	for _, line := range bytes.Split(raw, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if bytes.Equal(payload, []byte("[DONE]")) {
			observation.SawDone = true
			break
		}
		var frame struct {
			Error   json.RawMessage `json:"error"`
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if json.Unmarshal(payload, &frame) != nil {
			continue
		}
		if len(frame.Error) > 0 && !bytes.Equal(bytes.TrimSpace(frame.Error), []byte("null")) {
			observation.SawErrorFrame = true
		}
		for _, choice := range frame.Choices {
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				observation.SawFinishReason = true
				observation.FinishReason = *choice.FinishReason
			}
		}
	}

	response, err := Aggregate(bytes.NewReader(raw))
	if err != nil {
		return observation, err
	}
	observation.Response = response
	repair := RepairAggregatedResponse(response, ToolNameAllowlist(requestBody), true)
	observation.RepairedToolCalls = repair.Converted()
	observation.ValidToolCalls = countValidFunctionToolCalls(response)
	observation.ContentChars, observation.ReasoningChars = EmptyToolTurnTextStats(response)
	observation.ReasoningTokens = emptyToolTurnReasoningTokens(response)
	return observation, nil
}

// EmptyToolTurnTextStats returns only the amount of visible assistant content
// and reasoning text. It never returns the text itself.
func EmptyToolTurnTextStats(response map[string]any) (contentChars, reasoningChars int) {
	choices, _ := response["choices"].([]any)
	for _, rawChoice := range choices {
		choice, _ := rawChoice.(map[string]any)
		message, _ := choice["message"].(map[string]any)
		if content, _ := message["content"].(string); strings.TrimSpace(content) != "" {
			contentChars += len(strings.TrimSpace(content))
		}
		if reasoning, _ := message["reasoning_content"].(string); strings.TrimSpace(reasoning) != "" {
			reasoningChars += len(strings.TrimSpace(reasoning))
		}
	}
	return contentChars, reasoningChars
}

func emptyToolTurnReasoningTokens(response map[string]any) int64 {
	usage, _ := response["usage"].(map[string]any)
	details, _ := usage["completion_tokens_details"].(map[string]any)
	if value, ok := usageNumber(details, "reasoning_tokens"); ok && value > 0 {
		return int64(value)
	}
	return 0
}

func countValidFunctionToolCalls(response map[string]any) int {
	choices, _ := response["choices"].([]any)
	valid := 0
	for _, rawChoice := range choices {
		choice, _ := rawChoice.(map[string]any)
		message, _ := choice["message"].(map[string]any)
		var calls []map[string]any
		switch rawCalls := message["tool_calls"].(type) {
		case []any:
			for _, rawCall := range rawCalls {
				if call, ok := rawCall.(map[string]any); ok {
					calls = append(calls, call)
				}
			}
		case []map[string]any:
			calls = rawCalls
		}
		for _, call := range calls {
			if call == nil || call["type"] != "function" {
				continue
			}
			id, _ := call["id"].(string)
			function, _ := call["function"].(map[string]any)
			name, _ := function["name"].(string)
			arguments, _ := function["arguments"].(string)
			if id == "" || name == "" || (arguments != "" && !json.Valid([]byte(strings.TrimSpace(arguments)))) {
				continue
			}
			valid++
		}
	}
	return valid
}

// EmptyToolTurnCandidate is deliberately stricter than “response has no calls”:
// it requires an explicit upstream stop terminator and no error frame.
func (o EmptyToolTurnObservation) EmptyToolTurnCandidate() bool {
	if !o.SawDone || !o.SawFinishReason || o.FinishReason != "stop" || o.SawErrorFrame || o.ValidToolCalls != 0 {
		return false
	}
	// A normal final text answer can legitimately have finish_reason=stop even
	// when the request declared tools. Retrying it can discard a valid answer
	// and amplify a client tool loop, so only a truly empty assistant turn is
	// eligible.
	if o.ContentChars != 0 || o.ReasoningChars != 0 || o.ReasoningTokens != 0 {
		return false
	}
	if o.Response == nil {
		return false
	}
	choices, _ := o.Response["choices"].([]any)
	if len(choices) != 1 {
		return false
	}
	return true
}

// EmptyToolTurnTerminalReason returns a stable, non-sensitive diagnostic label.
func (o EmptyToolTurnObservation) EmptyToolTurnTerminalReason() string {
	if !o.SawDone {
		return "missing_done"
	}
	if !o.SawFinishReason {
		return "missing_finish_reason"
	}
	if o.FinishReason != "stop" {
		return fmt.Sprintf("finish_%s", o.FinishReason)
	}
	if o.SawErrorFrame {
		return "error_frame"
	}
	if o.ValidToolCalls != 0 {
		return "valid_tool_calls"
	}
	if o.ContentChars != 0 {
		return "content_present"
	}
	if o.ReasoningChars != 0 || o.ReasoningTokens != 0 {
		return "reasoning_present"
	}
	return "empty_tool_turn"
}
