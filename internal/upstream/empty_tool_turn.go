package upstream

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// EmptyToolTurnObservation contains only response facts needed to decide
// whether an opt-in same-account retry is safe. Response is the ordinary
// aggregate after the existing DSML repair has been applied.
type EmptyToolTurnObservation struct {
	SawDone           bool
	SawFinishReason   bool
	FinishReason      string
	SawErrorFrame     bool
	ValidToolCalls    int
	RepairedToolCalls int
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
	return observation, nil
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
	return "empty_tool_turn"
}
