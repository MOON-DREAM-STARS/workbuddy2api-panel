package server

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

type responsesRequest struct {
	Raw              map[string]any
	Model            string
	Stream           bool
	Metadata         map[string]any
	PromptCacheKey   string
	Tools            []any
	ToolChoice       any
	ParallelTools    bool
	ReasoningEffort  string
	ReasoningSummary string
	TextFormat       map[string]any
	// ClientMetadata is a Codex compatibility field. It is intentionally kept
	// out of ChatBody because the WorkBuddy Chat upstream has no contract for it.
	ClientMetadata any
	// Include is a narrow Codex compatibility field. The accepted marker is
	// intentionally kept out of ChatBody because the WorkBuddy Chat upstream
	// has no contract for Responses include semantics.
	Include         []string
	Instructions    any
	MaxOutputTokens any
	Temperature     any
	TopP            any
	ChatBody        []byte
	ToolNames       *responseToolNames
}

type responsesValidationError struct {
	Param   string
	Code    string
	Message string
}

func (e *responsesValidationError) Error() string {
	if e.Param == "" {
		return e.Message
	}
	return e.Param + ": " + e.Message
}

func responseBad(param, message string) error {
	return &responsesValidationError{Param: param, Code: "invalid_request_error", Message: message}
}

func responseUnsupported(param, message string) error {
	return &responsesValidationError{Param: param, Code: "unsupported_value", Message: message}
}

// responsesToChat is kept as a small testable conversion boundary. It returns
// the original decoded request as the second value for compatibility with
// existing package-local tests; the executor only receives Chat-shaped JSON.
func responsesToChat(raw []byte) ([]byte, map[string]any, error) {
	req, err := parseResponsesRequest(raw)
	if err != nil {
		return nil, nil, err
	}
	return req.ChatBody, req.Raw, nil
}

func parseResponsesRequest(raw []byte) (responsesRequest, error) {
	var req map[string]any
	dec := json.NewDecoder(bytes.NewReader(raw))
	if err := dec.Decode(&req); err != nil || req == nil {
		return responsesRequest{}, responseBad("", "request must be a JSON object")
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return responsesRequest{}, responseBad("", "request must contain one JSON value")
	}

	allowed := map[string]bool{
		"model": true, "input": true, "instructions": true, "stream": true,
		"store": true, "previous_response_id": true, "conversation": true,
		"max_output_tokens": true, "temperature": true, "top_p": true,
		"metadata": true, "prompt_cache_key": true, "tools": true,
		"tool_choice": true, "parallel_tool_calls": true, "reasoning": true,
		"text": true, "background": true, "client_metadata": true, "include": true,
		"service_tier": true, "stream_options": true,
	}
	unknown := make([]string, 0)
	for key := range req {
		if !allowed[key] {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return responsesRequest{}, responseUnsupported(unknown[0], "field is not supported by this stateless gateway")
	}

	model, ok := req["model"].(string)
	if !ok || strings.TrimSpace(model) == "" {
		return responsesRequest{}, responseBad("model", "model is required")
	}
	input, exists := req["input"]
	if !exists {
		return responsesRequest{}, responseBad("input", "input is required")
	}

	out := responsesRequest{Raw: req, Model: model, ParallelTools: true, ToolNames: newResponseToolNames()}
	if v, exists := req["client_metadata"]; exists {
		// This is request metadata for Codex clients, not a WorkBuddy Chat
		// control. Accept it explicitly but never copy it into ChatBody.
		out.ClientMetadata = v
	}
	if v, exists := req["service_tier"]; exists && v != nil {
		tier, ok := v.(string)
		if !ok {
			return responsesRequest{}, responseBad("service_tier", "must be a string or null")
		}
		if tier != "" && tier != "default" {
			return responsesRequest{}, responseUnsupported("service_tier", "non-default service tiers are not supported by this gateway")
		}
	}
	if v, exists := req["stream_options"]; exists && v != nil {
		options, ok := v.(map[string]any)
		if !ok {
			return responsesRequest{}, responseBad("stream_options", "must be an object or null")
		}
		unknownOptions := make([]string, 0)
		for key := range options {
			if key != "reasoning_summary_delivery" {
				unknownOptions = append(unknownOptions, key)
			}
		}
		if len(unknownOptions) > 0 {
			sort.Strings(unknownOptions)
			return responsesRequest{}, responseUnsupported("stream_options."+unknownOptions[0], "only reasoning_summary_delivery is supported")
		}
		if v, exists := options["reasoning_summary_delivery"]; exists && v != nil {
			delivery, ok := v.(string)
			if !ok {
				return responsesRequest{}, responseBad("stream_options.reasoning_summary_delivery", "must be sequential_cutoff")
			}
			if delivery != "sequential_cutoff" {
				return responsesRequest{}, responseUnsupported("stream_options.reasoning_summary_delivery", "only sequential_cutoff is supported")
			}
		}
	}
	if v, exists := req["include"]; exists {
		items, ok := v.([]any)
		if !ok {
			return responsesRequest{}, responseBad("include", "must be an array")
		}
		out.Include = make([]string, 0, len(items))
		for i, raw := range items {
			marker, ok := raw.(string)
			if !ok {
				return responsesRequest{}, responseBad(fmt.Sprintf("include[%d]", i), "must be a string")
			}
			if marker != "reasoning.encrypted_content" {
				return responsesRequest{}, responseUnsupported(fmt.Sprintf("include[%d]", i), "only reasoning.encrypted_content is supported as a compatibility marker")
			}
			out.Include = append(out.Include, marker)
		}
	}
	if v, exists := req["stream"]; exists {
		var ok bool
		out.Stream, ok = v.(bool)
		if !ok {
			return responsesRequest{}, responseBad("stream", "must be a boolean")
		}
	}
	if v, exists := req["store"]; exists && v != nil {
		b, ok := v.(bool)
		if !ok || b {
			return responsesRequest{}, responseUnsupported("store", "only false or null is supported; this gateway is stateless")
		}
	}
	if v, exists := req["previous_response_id"]; exists && v != nil {
		if s, ok := v.(string); !ok {
			return responsesRequest{}, responseBad("previous_response_id", "must be null or an empty string")
		} else if s != "" {
			return responsesRequest{}, responseUnsupported("previous_response_id", "stateful response lookup is not supported; send complete input history")
		}
	}
	if _, exists := req["conversation"]; exists {
		return responsesRequest{}, responseUnsupported("conversation", "conversation state is not supported; send complete input history")
	}
	if _, exists := req["background"]; exists {
		return responsesRequest{}, responseUnsupported("background", "background responses are not supported")
	}
	if v, exists := req["instructions"]; exists && v != nil {
		if _, ok := v.(string); !ok {
			return responsesRequest{}, responseBad("instructions", "must be a string or null")
		}
		out.Instructions = v
	}
	if v, exists := req["max_output_tokens"]; exists && v != nil {
		if !positiveInteger(v) {
			return responsesRequest{}, responseBad("max_output_tokens", "must be a positive integer")
		}
		out.MaxOutputTokens = v
	}
	for _, key := range []string{"temperature", "top_p"} {
		if v, exists := req[key]; exists && v != nil {
			if _, ok := v.(float64); !ok {
				return responsesRequest{}, responseBad(key, "must be a number")
			}
			if key == "temperature" {
				out.Temperature = v
			} else {
				out.TopP = v
			}
		}
	}
	if v, exists := req["metadata"]; exists && v != nil {
		m, ok := v.(map[string]any)
		if !ok {
			return responsesRequest{}, responseBad("metadata", "must be an object")
		}
		for key, value := range m {
			if _, ok := value.(string); !ok {
				return responsesRequest{}, responseBad("metadata."+key, "metadata values must be strings")
			}
		}
		out.Metadata = m
	}
	if v, exists := req["prompt_cache_key"]; exists && v != nil {
		s, ok := v.(string)
		if !ok || strings.TrimSpace(s) == "" {
			return responsesRequest{}, responseBad("prompt_cache_key", "must be a non-empty string")
		}
		out.PromptCacheKey = s
	}
	if v, exists := req["parallel_tool_calls"]; exists {
		b, ok := v.(bool)
		if !ok {
			return responsesRequest{}, responseBad("parallel_tool_calls", "must be a boolean")
		}
		out.ParallelTools = b
	}
	if v, exists := req["reasoning"]; exists && v != nil {
		m, ok := v.(map[string]any)
		if !ok {
			return responsesRequest{}, responseBad("reasoning", "must be an object")
		}
		for key := range m {
			if key != "effort" && key != "summary" && key != "context" {
				return responsesRequest{}, responseUnsupported("reasoning."+key, "only effort, summary, and all_turns context are supported")
			}
		}
		if v, exists := m["context"]; exists && v != nil {
			context, ok := v.(string)
			if !ok {
				return responsesRequest{}, responseBad("reasoning.context", "must be all_turns")
			}
			if context != "all_turns" {
				return responsesRequest{}, responseUnsupported("reasoning.context", "only all_turns is supported by this stateless gateway")
			}
		}
		if v, ok := m["effort"]; ok && v != nil {
			effort, ok := v.(string)
			if !ok || !isSupportedReasoningEffort(effort) {
				return responsesRequest{}, responseBad("reasoning.effort", "must be one of none, minimal, low, medium, high, xhigh, or max")
			}
			out.ReasoningEffort = effort
		}
		if v, ok := m["summary"]; ok && v != nil {
			summary, ok := v.(string)
			if !ok || !isSupportedReasoningSummary(summary) {
				return responsesRequest{}, responseBad("reasoning.summary", "must be one of auto, concise, or detailed")
			}
			out.ReasoningSummary = summary
		}
	}
	if v, exists := req["text"]; exists && v != nil {
		m, ok := v.(map[string]any)
		if !ok {
			return responsesRequest{}, responseBad("text", "must be an object")
		}
		if verbosity, exists := m["verbosity"]; exists {
			value, ok := verbosity.(string)
			if !ok || !isSupportedTextVerbosity(value) {
				return responsesRequest{}, responseBad("text.verbosity", "must be one of low, medium, or high")
			}
		}
		if rawFormat, exists := m["format"]; exists {
			format, ok := rawFormat.(map[string]any)
			if !ok {
				return responsesRequest{}, responseBad("text.format", "must specify format.type=text")
			}
			for key := range format {
				if key != "type" {
					return responsesRequest{}, responseUnsupported("text.format."+key, "only text format is supported")
				}
			}
			if format["type"] != "text" {
				return responsesRequest{}, responseUnsupported("text.format.type", "json_schema and other response formats are not supported")
			}
			out.TextFormat = map[string]any{"type": "text"}
		}
	}
	var chatTools []any
	var chatToolChoice any
	if v, exists := req["tools"]; exists {
		tools, publicTools, err := out.ToolNames.convert(v)
		if err != nil {
			return responsesRequest{}, err
		}
		chatTools = tools
		out.Tools = publicTools
	}
	if v, exists := req["tool_choice"]; exists {
		choice, chatChoice, err := out.ToolNames.choice(v)
		if err != nil {
			return responsesRequest{}, err
		}
		if choice == "required" && len(out.Tools) == 0 {
			return responsesRequest{}, responseUnsupported("tool_choice", "required needs at least one supported function tool")
		}
		out.ToolChoice = choice
		chatToolChoice = chatChoice
	}

	messages, err := responsesInputMessagesWithNames(input, out.Instructions, out.ToolNames)
	if err != nil {
		return responsesRequest{}, err
	}
	chat := map[string]any{"model": model, "messages": messages, "stream": out.Stream}
	if out.MaxOutputTokens != nil {
		chat["max_completion_tokens"] = out.MaxOutputTokens
	}
	if out.Temperature != nil {
		chat["temperature"] = out.Temperature
	}
	if out.TopP != nil {
		chat["top_p"] = out.TopP
	}
	if out.Metadata != nil {
		chat["metadata"] = out.Metadata
	}
	if out.PromptCacheKey != "" {
		chat["prompt_cache_key"] = out.PromptCacheKey
	}
	if _, exists := req["parallel_tool_calls"]; exists {
		chat["parallel_tool_calls"] = out.ParallelTools
	}
	if out.ReasoningEffort != "" {
		chat["reasoning_effort"] = out.ReasoningEffort
	}
	if out.ReasoningSummary != "" {
		chat["reasoning_summary"] = out.ReasoningSummary
	}
	if out.TextFormat != nil {
		chat["response_format"] = out.TextFormat
	}
	if chatTools != nil {
		chat["tools"] = chatTools
	}
	if _, exists := req["tool_choice"]; exists {
		chat["tool_choice"] = chatToolChoice
	}
	body, err := json.Marshal(chat)
	if err != nil {
		return responsesRequest{}, fmt.Errorf("marshal chat request: %w", err)
	}
	out.ChatBody = body
	return out, nil
}

func positiveInteger(v any) bool {
	n, ok := v.(float64)
	return ok && n > 0 && n == float64(int64(n))
}

func isSupportedReasoningEffort(v string) bool {
	switch v {
	case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		return true
	default:
		return false
	}
}

func isSupportedTextVerbosity(v string) bool {
	switch v {
	case "low", "medium", "high":
		return true
	default:
		return false
	}
}

func isSupportedReasoningSummary(v string) bool {
	switch v {
	case "auto", "concise", "detailed":
		return true
	default:
		return false
	}
}

func convertFunctionTools(v any) ([]any, error) {
	items, ok := v.([]any)
	if !ok {
		return nil, responseBad("tools", "must be an array")
	}
	var out []any
	for i, raw := range items {
		tool, ok := raw.(map[string]any)
		if !ok {
			return nil, responseBad(fmt.Sprintf("tools[%d]", i), "must be an object")
		}
		rawType, exists := tool["type"]
		if !exists {
			return nil, responseBad(fmt.Sprintf("tools[%d].type", i), "must be a string")
		}
		toolType, ok := rawType.(string)
		if !ok {
			return nil, responseBad(fmt.Sprintf("tools[%d].type", i), "must be a string")
		}
		if toolType != "function" {
			// WorkBuddy Chat cannot execute hosted, namespace, or other opaque
			// Responses tools. Keep the request compatible by dropping only those
			// declarations; function tools continue through the normal conversion.
			continue
		}
		allowed := map[string]bool{"type": true, "name": true, "description": true, "parameters": true, "strict": true}
		for key := range tool {
			if !allowed[key] {
				return nil, responseUnsupported(fmt.Sprintf("tools[%d].%s", i, key), "field is not supported")
			}
		}
		name, ok := tool["name"].(string)
		if !ok || strings.TrimSpace(name) == "" {
			return nil, responseBad(fmt.Sprintf("tools[%d].name", i), "function name is required")
		}
		fn := map[string]any{"name": name}
		if v, ok := tool["description"]; ok {
			if _, ok := v.(string); !ok {
				return nil, responseBad(fmt.Sprintf("tools[%d].description", i), "must be a string")
			}
			fn["description"] = v
		}
		if v, ok := tool["parameters"]; ok {
			if _, ok := v.(map[string]any); !ok {
				return nil, responseBad(fmt.Sprintf("tools[%d].parameters", i), "must be an object")
			}
			fn["parameters"] = v
		}
		if v, ok := tool["strict"]; ok {
			if _, ok := v.(bool); !ok {
				return nil, responseBad(fmt.Sprintf("tools[%d].strict", i), "must be a boolean")
			}
			fn["strict"] = v
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out, nil
}

func convertToolChoice(v any) (any, any, error) {
	switch choice := v.(type) {
	case string:
		if choice != "auto" && choice != "none" && choice != "required" {
			return nil, nil, responseUnsupported("tool_choice", "only auto, none, required, or a function choice is supported")
		}
		return choice, choice, nil
	case map[string]any:
		if choice["type"] != "function" {
			return nil, nil, responseUnsupported("tool_choice.type", "only function tool choice is supported")
		}
		name, ok := choice["name"].(string)
		if !ok || strings.TrimSpace(name) == "" {
			return nil, nil, responseBad("tool_choice.name", "function name is required")
		}
		for key := range choice {
			if key != "type" && key != "name" {
				return nil, nil, responseUnsupported("tool_choice."+key, "field is not supported")
			}
		}
		return choice, map[string]any{"type": "function", "function": map[string]any{"name": name}}, nil
	default:
		return nil, nil, responseBad("tool_choice", "must be a string or function object")
	}
}

func responsesInputMessages(input any, instructions any) ([]any, error) {
	return responsesInputMessagesWithNames(input, instructions, newResponseToolNames())
}

func responsesInputMessagesWithNames(input any, instructions any, names *responseToolNames) ([]any, error) {
	messages := make([]any, 0)
	if instructions != nil {
		if s := instructions.(string); s != "" {
			messages = append(messages, map[string]any{"role": "system", "content": s})
		}
	}
	history := &responseHistory{names: names, messages: messages, calls: map[string]bool{}, outputs: map[string]bool{}}
	switch value := input.(type) {
	case string:
		history.messages = append(history.messages, map[string]any{"role": "user", "content": value})
	case []any:
		for i, raw := range value {
			item, ok := raw.(map[string]any)
			if !ok {
				return nil, responseBad(fmt.Sprintf("input[%d]", i), "must be an object")
			}
			if err := history.add(item, i); err != nil {
				return nil, err
			}
		}
	default:
		return nil, responseBad("input", "must be a string or array")
	}
	if len(history.messages) == 0 {
		return nil, responseBad("input", "must contain at least one supported item")
	}
	return history.messages, nil
}

type responseHistory struct {
	names    *responseToolNames
	messages []any
	calls    map[string]bool
	outputs  map[string]bool
}

func (h *responseHistory) add(item map[string]any, pos int) error {
	typ, _ := item["type"].(string)
	if typ == "" {
		typ = "message"
	}
	allowed := map[string]bool{}
	switch typ {
	case "message":
		allowed = map[string]bool{"type": true, "role": true, "content": true, "phase": true, "id": true}
	case "input_text":
		allowed = map[string]bool{"type": true, "text": true}
	case "input_image":
		allowed = map[string]bool{"type": true, "image_url": true, "detail": true}
	case "function_call":
		allowed = map[string]bool{"type": true, "id": true, "call_id": true, "name": true, "arguments": true, "content": true, "namespace": true, "status": true}
	case "reasoning":
		allowed = map[string]bool{"type": true, "id": true, "status": true, "summary": true, "encrypted_content": true, "content": true}
	case "function_call_output":
		allowed = map[string]bool{"type": true, "id": true, "call_id": true, "output": true, "content": true}
	default:
		return responseUnsupported(fmt.Sprintf("input[%d].type", pos), "supported types are message, input_text, input_image, reasoning, function_call, and function_call_output")
	}
	for key := range item {
		if !allowed[key] {
			return responseUnsupported(fmt.Sprintf("input[%d].%s", pos, key), "field is not supported")
		}
	}
	if id, exists := item["id"]; exists && id != nil {
		if _, ok := id.(string); !ok {
			return responseBad(fmt.Sprintf("input[%d].id", pos), "must be a string or null")
		}
	}
	if encrypted, exists := item["encrypted_content"]; exists && encrypted != nil {
		if _, ok := encrypted.(string); !ok {
			return responseBad(fmt.Sprintf("input[%d].encrypted_content", pos), "must be a string or null")
		}
	}
	switch typ {
	case "message":
		role, ok := item["role"].(string)
		if !ok || (role != "user" && role != "assistant" && role != "system" && role != "developer") {
			return responseBad(fmt.Sprintf("input[%d].role", pos), "unsupported message role")
		}
		if phase, exists := item["phase"]; exists && phase != nil {
			phaseValue, ok := phase.(string)
			if !ok {
				return responseBad(fmt.Sprintf("input[%d].phase", pos), "must be a string or null")
			}
			if role != "assistant" {
				return responseUnsupported(fmt.Sprintf("input[%d].phase", pos), "phase is only supported for assistant messages")
			}
			if phaseValue != "commentary" && phaseValue != "final_answer" {
				return responseUnsupported(fmt.Sprintf("input[%d].phase", pos), "only commentary or final_answer is supported")
			}
		}
		content, err := responsesContent(item["content"])
		if err != nil {
			return responseBad(fmt.Sprintf("input[%d].content", pos), err.Error())
		}
		h.messages = append(h.messages, map[string]any{"role": role, "content": content})
	case "input_text":
		text, ok := item["text"].(string)
		if !ok {
			return responseBad(fmt.Sprintf("input[%d].text", pos), "must be a string")
		}
		h.messages = append(h.messages, map[string]any{"role": "user", "content": text})
	case "input_image":
		content, err := responsesContent([]any{item})
		if err != nil {
			return responseBad(fmt.Sprintf("input[%d]", pos), err.Error())
		}
		h.messages = append(h.messages, map[string]any{"role": "user", "content": content})
	case "reasoning":
		text, err := responsesReasoningSummaryText(item["summary"])
		if err != nil {
			return responseBad(fmt.Sprintf("input[%d].summary", pos), err.Error())
		}
		if text != "" {
			h.messages = append(h.messages, map[string]any{"role": "assistant", "content": nil, "reasoning": text, "reasoning_content": text})
		}
	case "function_call":
		id, _ := item["call_id"].(string)
		name, _ := item["name"].(string)
		args, ok := item["arguments"].(string)
		if id == "" || name == "" || !ok {
			return responseBad(fmt.Sprintf("input[%d]", pos), "function_call requires call_id, name and string arguments")
		}
		ns, err := responseNamespace(item, fmt.Sprintf("input[%d].namespace", pos))
		if err != nil {
			return err
		}
		if status, exists := item["status"]; exists && status != nil {
			if status != "completed" && status != "in_progress" && status != "incomplete" {
				return responseBad(fmt.Sprintf("input[%d].status", pos), "unsupported function call status")
			}
		}
		if h.names == nil {
			h.names = newResponseToolNames()
		}
		name, err = h.names.wire(ns, name, fmt.Sprintf("input[%d].name", pos))
		if err != nil {
			return err
		}
		if h.calls[id] {
			return responseBad(fmt.Sprintf("input[%d].call_id", pos), "duplicate function call id")
		}
		h.calls[id] = true
		call := map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": args}}
		if len(h.messages) > 0 {
			if last, ok := h.messages[len(h.messages)-1].(map[string]any); ok && last["role"] == "assistant" {
				calls, _ := last["tool_calls"].([]any)
				last["tool_calls"] = append(calls, call)
				return nil
			}
		}
		h.messages = append(h.messages, map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{call}})
	case "function_call_output":
		id, _ := item["call_id"].(string)
		if id == "" {
			return responseBad(fmt.Sprintf("input[%d].call_id", pos), "function_call_output requires call_id")
		}
		if !h.calls[id] {
			return responseBad(fmt.Sprintf("input[%d].call_id", pos), "function_call_output has no preceding function call")
		}
		if h.outputs[id] {
			return responseBad(fmt.Sprintf("input[%d].call_id", pos), "duplicate function_call_output")
		}
		h.outputs[id] = true
		outputValue, outputParam := item["output"], "output"
		if _, exists := item["output"]; !exists {
			outputValue, outputParam = item["content"], "content"
		}
		content, err := responsesContent(outputValue)
		if err != nil {
			return responseBad(fmt.Sprintf("input[%d].%s", pos, outputParam), err.Error())
		}
		h.messages = append(h.messages, map[string]any{"role": "tool", "tool_call_id": id, "content": content})
	default:
		return responseUnsupported(fmt.Sprintf("input[%d].type", pos), "supported types are message, input_text, input_image, reasoning, function_call, and function_call_output")
	}
	return nil
}

func responsesReasoningSummaryText(v any) (string, error) {
	if v == nil {
		return "", nil
	}
	parts, ok := v.([]any)
	if !ok {
		return "", errors.New("summary must be an array")
	}
	var b strings.Builder
	for i, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			return "", fmt.Errorf("summary[%d] must be an object", i)
		}
		for key := range part {
			if key != "type" && key != "text" {
				return "", fmt.Errorf("summary[%d].%s is unsupported", i, key)
			}
		}
		typ, _ := part["type"].(string)
		if typ != "summary_text" {
			return "", fmt.Errorf("summary[%d].type must be summary_text", i)
		}
		text, ok := part["text"].(string)
		if !ok {
			return "", fmt.Errorf("summary[%d].text must be a string", i)
		}
		b.WriteString(text)
	}
	return b.String(), nil
}

func responsesContent(v any) (any, error) {
	if s, ok := v.(string); ok {
		return s, nil
	}
	parts, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("content must be a string or array")
	}
	out := make([]any, 0, len(parts))
	for i, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("content part %d must be an object", i)
		}
		typ, _ := part["type"].(string)
		allowed := map[string]bool{"type": true}
		if typ == "input_text" || typ == "output_text" {
			allowed["text"] = true
		} else if typ == "input_image" {
			allowed["image_url"] = true
			allowed["detail"] = true
		}
		for key := range part {
			if !allowed[key] {
				return nil, fmt.Errorf("content part %d field %q is unsupported", i, key)
			}
		}
		switch typ {
		case "input_text", "output_text":
			text, ok := part["text"].(string)
			if !ok {
				return nil, fmt.Errorf("content part %d text must be a string", i)
			}
			out = append(out, map[string]any{"type": "text", "text": text})
		case "input_image":
			url, ok := part["image_url"].(string)
			if !ok || strings.TrimSpace(url) == "" {
				return nil, fmt.Errorf("content part %d requires string image_url; file_id is unsupported", i)
			}
			image := map[string]any{"url": url}
			if detail, ok := part["detail"]; ok {
				if _, ok := detail.(string); !ok {
					return nil, fmt.Errorf("content part %d detail must be a string", i)
				}
				image["detail"] = detail
			}
			out = append(out, map[string]any{"type": "image_url", "image_url": image})
		default:
			return nil, fmt.Errorf("unsupported content part type %q", typ)
		}
	}
	return out, nil
}

type responseSSEEncoder struct {
	w       io.Writer
	flush   func()
	prepare func()
	seq     int
	started bool
	closed  bool
	err     error
}

func (e *responseSSEEncoder) emit(event string, payload map[string]any) error {
	if e == nil || e.closed || e.err != nil {
		if e != nil {
			return e.err
		}
		return nil
	}
	if !e.started {
		e.started = true
		if e.prepare != nil {
			e.prepare()
		}
	}
	if payload == nil {
		payload = map[string]any{}
	}
	payload["type"] = event
	payload["sequence_number"] = e.seq
	e.seq++
	data, err := json.Marshal(payload)
	if err == nil {
		_, err = io.WriteString(e.w, "event: "+event+"\ndata: "+string(data)+"\n\n")
	}
	if err != nil {
		e.err = err
		return err
	}
	if e.flush != nil {
		e.flush()
	}
	return nil
}

type responseToolState struct {
	Key          string
	UpstreamID   string
	PublicCallID string
	Name         string
	Arguments    strings.Builder
	ItemIndex    int
	Index        int
	HasIndex     bool
}

type responseState struct {
	req      responsesRequest
	stream   bool
	encoder  *responseSSEEncoder
	response map[string]any
	items    []map[string]any

	textItem     int
	toolByKey    map[string]*responseToolState
	toolAliases  map[string]string
	toolOrder    []string
	activeTool   string
	anonymousSeq int

	finishSeen      bool
	finishReason    string
	sawDone         bool
	reasoningItem   int
	reasoningText   strings.Builder
	reasoningClosed bool
	terminal        bool
	started         bool
	validEvents     int
	upstreamErr     map[string]any
	usage           map[string]any
	priorUsage      map[string]any
	decodeErr       error
}

func newResponseState(req responsesRequest, stream bool, encoder *responseSSEEncoder) *responseState {
	resp := map[string]any{
		"id":                   responseID("resp_"),
		"object":               "response",
		"created_at":           time.Now().Unix(),
		"status":               "in_progress",
		"model":                req.Model,
		"output":               []any{},
		"error":                nil,
		"incomplete_details":   nil,
		"usage":                nil,
		"store":                false,
		"previous_response_id": nil,
		"metadata":             map[string]any{},
		"tools":                []any{},
		"tool_choice":          "auto",
		"parallel_tool_calls":  req.ParallelTools,
		"instructions":         req.Instructions,
		"max_output_tokens":    req.MaxOutputTokens,
		"text":                 map[string]any{"format": map[string]any{"type": "text"}},
	}
	if req.Metadata != nil {
		resp["metadata"] = req.Metadata
	}
	if req.Tools != nil {
		resp["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		resp["tool_choice"] = req.ToolChoice
	}
	if req.TextFormat != nil {
		resp["text"] = map[string]any{"format": req.TextFormat}
	}
	if req.Temperature != nil {
		resp["temperature"] = req.Temperature
	}
	if req.TopP != nil {
		resp["top_p"] = req.TopP
	}
	if req.ReasoningEffort != "" || req.ReasoningSummary != "" {
		reasoning := map[string]any{}
		if req.ReasoningEffort != "" {
			reasoning["effort"] = req.ReasoningEffort
		}
		if req.ReasoningSummary != "" {
			reasoning["summary"] = req.ReasoningSummary
		}
		resp["reasoning"] = reasoning
	}
	return &responseState{
		req:           req,
		stream:        stream,
		encoder:       encoder,
		response:      resp,
		textItem:      -1,
		reasoningItem: -1,
		toolByKey:     map[string]*responseToolState{},
		toolAliases:   map[string]string{},
	}
}

func responseID(prefix string) string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return fmt.Sprintf("%s%x", prefix, time.Now().UnixNano())
	}
	return prefix + hex.EncodeToString(buf[:])
}

func (s *responseState) emit(event string, payload map[string]any) {
	if s.encoder == nil || s.terminal {
		return
	}
	if err := s.encoder.emit(event, payload); err != nil {
		s.decodeErr = err
	}
}

func (s *responseState) start() {
	if s.started {
		return
	}
	s.started = true
	s.emit("response.created", map[string]any{"response": s.response})
	s.emit("response.in_progress", map[string]any{"response": s.response})
}

func (s *responseState) ensureReasoningItem() int {
	if s.reasoningItem >= 0 && !s.reasoningClosed {
		return s.reasoningItem
	}
	if s.reasoningClosed {
		s.reasoningItem = -1
		s.reasoningText.Reset()
		s.reasoningClosed = false
	}
	item := map[string]any{
		"id":      responseID("rs_"),
		"type":    "reasoning",
		"status":  "in_progress",
		"summary": []any{},
	}
	s.reasoningItem = s.addItem(item)
	s.emit("response.reasoning_summary_part.added", map[string]any{
		"item_id":       item["id"],
		"output_index":  s.reasoningItem,
		"summary_index": 0,
		"part":          map[string]any{"type": "summary_text"},
	})
	return s.reasoningItem
}

func (s *responseState) addReasoning(text string) {
	if s.req.ReasoningSummary == "" || text == "" {
		return
	}
	idx := s.ensureReasoningItem()
	item := s.items[idx]
	parts := item["summary"].([]any)
	if len(parts) == 0 {
		parts = append(parts, map[string]any{"type": "summary_text", "text": ""})
		item["summary"] = parts
	}
	part := parts[0].(map[string]any)
	part["text"] = part["text"].(string) + text
	s.reasoningText.WriteString(text)
	s.emit("response.reasoning_summary_text.delta", map[string]any{
		"item_id":       item["id"],
		"output_index":  idx,
		"summary_index": 0,
		"delta":         text,
	})
}

func (s *responseState) markReasoningClosed() {
	if s.reasoningItem >= 0 {
		s.reasoningClosed = true
	}
}

func (s *responseState) addItem(item map[string]any) int {
	index := len(s.items)
	s.items = append(s.items, item)
	s.emit("response.output_item.added", map[string]any{
		"output_index": index,
		"item":         item,
	})
	return index
}

func (s *responseState) ensureTextItem() int {
	if s.textItem >= 0 {
		return s.textItem
	}
	item := map[string]any{
		"id":     responseID("msg_"),
		"type":   "message",
		"role":   "assistant",
		"status": "in_progress",
		"content": []any{map[string]any{
			"type":        "output_text",
			"text":        "",
			"annotations": []any{},
		}},
	}
	s.textItem = s.addItem(item)
	s.emit("response.content_part.added", map[string]any{
		"item_id":       item["id"],
		"output_index":  s.textItem,
		"content_index": 0,
		"part":          item["content"].([]any)[0],
	})
	return s.textItem
}

func (s *responseState) addText(text string) {
	if text == "" {
		return
	}
	idx := s.ensureTextItem()
	item := s.items[idx]
	parts := item["content"].([]any)
	part := parts[0].(map[string]any)
	part["text"] = part["text"].(string) + text
	s.emit("response.output_text.delta", map[string]any{
		"item_id":       item["id"],
		"output_index":  idx,
		"content_index": 0,
		"delta":         text,
	})
}

func (s *responseState) addRefusal(text string) {
	if text == "" {
		return
	}
	idx := s.ensureTextItem()
	item := s.items[idx]
	parts := item["content"].([]any)
	part := parts[0].(map[string]any)
	previous := ""
	if value, ok := part["refusal"].(string); ok {
		previous = value
	}
	part["type"] = "refusal"
	part["refusal"] = previous + text
	delete(part, "text")
	s.emit("response.refusal.delta", map[string]any{
		"item_id":       item["id"],
		"output_index":  idx,
		"content_index": 0,
		"delta":         text,
	})
}

func (s *responseState) resolveTool(tc map[string]any) (*responseToolState, error) {
	fn, _ := tc["function"].(map[string]any)
	if fn == nil {
		return nil, errors.New("tool call is missing function object")
	}
	id, _ := tc["id"].(string)
	name, _ := fn["name"].(string)
	if rawIndex, exists := tc["index"]; exists {
		if _, ok := toolIndex(rawIndex); !ok {
			return nil, errors.New("tool call index must be a non-negative integer")
		}
	}
	idx, hasIndex := toolIndex(tc["index"])
	key := ""
	if id != "" {
		key = s.toolAliases["id:"+id]
		if key == "" && !hasIndex && s.activeTool != "" {
			if len(s.toolByKey) > 1 {
				return nil, errors.New("ambiguous late tool call id without index")
			}
			if active := s.toolByKey[s.activeTool]; active != nil && active.UpstreamID == "" {
				key = s.activeTool
			}
		}
		if key == "" {
			key = "id:" + id
		}
	}
	if hasIndex {
		indexKey := fmt.Sprintf("index:%d", idx)
		if existing := s.toolAliases[indexKey]; existing != "" && key != "" && existing != key {
			return nil, errors.New("tool call index and id identify different calls")
		}
		if key == "" {
			key = existingOr(indexKey, s.toolAliases[indexKey])
		}
		s.toolAliases[indexKey] = key
	}
	if key == "" {
		if !hasIndex && id == "" && len(s.toolByKey) > 1 {
			return nil, errors.New("ambiguous tool call without index or id")
		}
		if s.activeTool == "" {
			s.anonymousSeq++
			key = fmt.Sprintf("anonymous:%d", s.anonymousSeq)
		} else {
			active := s.toolByKey[s.activeTool]
			if name != "" && active != nil && active.Name != "" && active.Arguments.Len() > 0 {
				return nil, errors.New("ambiguous tool call without index or id")
			}
			key = s.activeTool
		}
	}
	if id != "" {
		s.toolAliases["id:"+id] = key
	}
	state := s.toolByKey[key]
	if state == nil {
		publicID := id
		if publicID == "" {
			publicID = responseID("call_")
		}
		item := map[string]any{
			"id":        responseID("fc_"),
			"type":      "function_call",
			"status":    "in_progress",
			"call_id":   publicID,
			"name":      name,
			"arguments": "",
		}
		s.req.ToolNames.restore(item, name)
		state = &responseToolState{Key: key, UpstreamID: id, PublicCallID: publicID, Name: name, ItemIndex: s.addItem(item), Index: idx, HasIndex: hasIndex}
		s.toolByKey[key] = state
		s.toolOrder = append(s.toolOrder, key)
	} else {
		if hasIndex && state.HasIndex && state.Index != idx {
			return nil, errors.New("tool call index changed after publication")
		}
		if id != "" && state.UpstreamID != "" && id != state.UpstreamID {
			return nil, errors.New("tool call id changed after publication")
		}
		if name != "" && state.Name != "" && name != state.Name {
			return nil, errors.New("tool call name changed after publication")
		}
		if state.UpstreamID == "" {
			state.UpstreamID = id
		}
		if state.Name == "" {
			state.Name = name
		}
	}
	s.activeTool = key
	return state, nil
}

func existingOr(fallback, value string) string {
	if value != "" {
		return value
	}
	return fallback
}

func toolIndex(v any) (int, bool) {
	n, ok := v.(float64)
	if !ok || n != float64(int(n)) || n < 0 {
		return 0, false
	}
	return int(n), true
}

func (s *responseState) addToolArguments(tc map[string]any) error {
	state, err := s.resolveTool(tc)
	if err != nil {
		return err
	}
	fn, _ := tc["function"].(map[string]any)
	args, _ := fn["arguments"].(string)
	if name, _ := fn["name"].(string); name != "" {
		item := s.items[state.ItemIndex]
		s.req.ToolNames.restore(item, name)
	}
	if args == "" {
		return nil
	}
	item := s.items[state.ItemIndex]
	state.Arguments.WriteString(args)
	item["arguments"] = state.Arguments.String()
	s.emit("response.function_call_arguments.delta", map[string]any{
		"item_id":      item["id"],
		"output_index": state.ItemIndex,
		"call_id":      state.PublicCallID,
		"delta":        args,
	})
	return nil
}

func (s *responseState) consumeChunk(chunk map[string]any) error {
	s.validEvents++
	if raw, ok := chunk["error"].(map[string]any); ok {
		s.upstreamErr = raw
		return nil
	}
	if usage, ok := chunk["usage"].(map[string]any); ok {
		s.usage = usage
		s.response["usage"] = responsesUsage(usage)
	}
	choices, ok := chunk["choices"].([]any)
	if !ok {
		return nil
	}
	if len(choices) == 0 {
		return nil
	}
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			return errors.New("choice must be an object")
		}
		if index, ok := toolIndex(choice["index"]); ok && index != 0 {
			return errors.New("multiple chat choices are not supported")
		}
		if rawFinish, exists := choice["finish_reason"]; exists && rawFinish != nil {
			finish, ok := rawFinish.(string)
			if !ok {
				return errors.New("finish_reason must be a string when present")
			}
			// Some OpenAI-compatible upstreams emit an empty finish_reason on
			// intermediate chunks. It is not a terminal fact; keep decoding so a
			// later non-empty finish_reason can determine the final state. If the
			// stream ends without one, finishSeen remains false and finish() fails.
			if finish != "" {
				s.finishSeen = true
				s.finishReason = finish
			}
		}
		message, _ := choice["delta"].(map[string]any)
		if message == nil {
			message, _ = choice["message"].(map[string]any)
		}
		if message == nil {
			return nil
		}
		if reasoning, ok := message["reasoning_content"].(string); ok {
			s.addReasoning(reasoning)
		}
		if content, ok := message["content"].(string); ok && content != "" {
			s.markReasoningClosed()
			s.addText(content)
		}
		if refusal, ok := message["refusal"].(string); ok && refusal != "" {
			s.markReasoningClosed()
			s.addRefusal(refusal)
		}
		if calls, ok := message["tool_calls"].([]any); ok && len(calls) > 0 {
			s.markReasoningClosed()
			for _, rawCall := range calls {
				call, ok := rawCall.(map[string]any)
				if !ok {
					return errors.New("tool_call must be an object")
				}
				if err := s.addToolArguments(call); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (s *responseState) finish(sawDone bool, decodeErr error) {
	if s.terminal {
		return
	}
	s.sawDone = sawDone
	if decodeErr != nil {
		s.decodeErr = decodeErr
	}
	failed := s.decodeErr != nil || s.upstreamErr != nil || !s.sawDone || !s.finishSeen
	incomplete := false
	incompleteReason := ""
	if !failed {
		switch s.finishReason {
		case "stop", "tool_calls":
			for _, key := range s.toolOrder {
				state := s.toolByKey[key]
				if state != nil && !completeToolArguments(state.Arguments.String()) {
					incomplete = true
					incompleteReason = "tool_call_arguments"
				}
			}
		case "length":
			incomplete = true
			incompleteReason = "max_output_tokens"
		case "content_filter":
			incomplete = true
			incompleteReason = "content_filter"
		default:
			failed = true
		}
	}
	if failed {
		s.response["status"] = "failed"
		message := "upstream response ended without a valid completion"
		code := "upstream_parse"
		if s.upstreamErr != nil {
			if v, ok := s.upstreamErr["message"].(string); ok && v != "" {
				message = v
			}
			if v, ok := s.upstreamErr["code"].(string); ok && v != "" {
				code = v
			}
		}
		if s.decodeErr != nil {
			message = s.decodeErr.Error()
			code = "upstream_parse"
		} else if !s.sawDone {
			message = "upstream stream ended before [DONE]"
		} else if !s.finishSeen {
			message = "upstream ended without a finish reason"
		}
		s.response["error"] = map[string]any{"code": code, "message": message, "type": "server_error"}
	} else if incomplete {
		s.response["status"] = "incomplete"
		s.response["incomplete_details"] = map[string]any{"reason": incompleteReason}
	} else {
		s.response["status"] = "completed"
	}
	for _, item := range s.items {
		if item["type"] == "function_call" {
			if s.response["status"] == "completed" && completeToolArguments(asString(item["arguments"])) {
				item["status"] = "completed"
			} else {
				item["status"] = "incomplete"
			}
		} else {
			if s.response["status"] == "completed" {
				item["status"] = "completed"
			} else {
				item["status"] = "incomplete"
			}
		}
	}
	output := make([]any, 0, len(s.items))
	for _, item := range s.items {
		output = append(output, item)
	}
	s.response["output"] = output
	if s.priorUsage != nil {
		s.response["usage"] = responsesUsage(mergeUsageMaps(s.priorUsage, s.usage))
	}
	if s.encoder != nil {
		s.emitItemDoneEvents()
		terminalEvent := "response.completed"
		if failed {
			terminalEvent = "response.failed"
		} else if incomplete {
			terminalEvent = "response.incomplete"
		}
		s.emit(terminalEvent, map[string]any{"response": s.response})
		s.encoder.closed = true
	}
	s.terminal = true
}

func (s *responseState) emitItemDoneEvents() {
	for index, item := range s.items {
		switch item["type"] {
		case "reasoning":
			parts, _ := item["summary"].([]any)
			text := ""
			if len(parts) > 0 {
				if part, ok := parts[0].(map[string]any); ok {
					text, _ = part["text"].(string)
				}
			}
			s.emit("response.reasoning_summary_text.done", map[string]any{"item_id": item["id"], "output_index": index, "summary_index": 0, "text": text})
			s.emit("response.reasoning_summary_part.done", map[string]any{"item_id": item["id"], "output_index": index, "summary_index": 0, "part": map[string]any{"type": "summary_text", "text": text}})
		case "message":
			parts, _ := item["content"].([]any)
			if len(parts) > 0 {
				part, _ := parts[0].(map[string]any)
				if part["type"] == "refusal" {
					s.emit("response.refusal.done", map[string]any{"item_id": item["id"], "output_index": index, "content_index": 0, "refusal": part["refusal"]})
				} else {
					s.emit("response.output_text.done", map[string]any{"item_id": item["id"], "output_index": index, "content_index": 0, "text": part["text"]})
				}
				s.emit("response.content_part.done", map[string]any{"item_id": item["id"], "output_index": index, "content_index": 0, "part": part})
			}
		case "function_call":
			s.emit("response.function_call_arguments.done", map[string]any{"item_id": item["id"], "output_index": index, "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]})
		}
		s.emit("response.output_item.done", map[string]any{"output_index": index, "item": item})
	}
}

func completeToolArguments(args string) bool {
	trimmed := strings.TrimSpace(args)
	if trimmed == "" {
		return true
	}
	return json.Valid([]byte(trimmed))
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func responsesUsage(usage map[string]any) map[string]any {
	out := map[string]any{}
	copyNumber := func(dst, src string) {
		if v, ok := usage[src]; ok {
			out[dst] = v
		}
	}
	copyNumber("input_tokens", "prompt_tokens")
	copyNumber("output_tokens", "completion_tokens")
	copyNumber("total_tokens", "total_tokens")
	if details, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		if cached, exists := details["cached_tokens"]; exists {
			out["input_tokens_details"] = map[string]any{"cached_tokens": cached}
		}
	}
	if details, ok := usage["completion_tokens_details"].(map[string]any); ok {
		if reasoning, exists := details["reasoning_tokens"]; exists {
			out["output_tokens_details"] = map[string]any{"reasoning_tokens": reasoning}
		}
	}
	if credit, ok := usage["credit"]; ok {
		out["credit"] = credit
	}
	return out
}

func (s *responseState) outcome() ChatOutcome {
	out := ChatOutcome{
		Status:          http.StatusOK,
		Response:        s.response,
		SawDone:         s.sawDone,
		OutputCommitted: s.started || s.validEvents > 0,
		ReadErr:         s.decodeErr,
		HasUsage:        s.usage != nil,
	}
	if s.finishSeen {
		finish := s.finishReason
		out.FinishReason = &finish
	}
	if s.upstreamErr != nil {
		out.UpstreamErr = errors.New("upstream error frame")
	}
	if s.usage != nil {
		if v, ok := s.usage["completion_tokens"].(float64); ok {
			out.Tokens = int(v)
		}
		if _, ok := s.usage["completion_tokens"]; !ok {
			out.Tokens = -1
		}
		if v, ok := s.usage["prompt_tokens"].(float64); ok {
			out.Prompt = int(v)
		}
		if details, ok := s.usage["prompt_tokens_details"].(map[string]any); ok {
			out.CacheHit = intNumber(details["cached_tokens"])
		}
		if v, ok := s.usage["prompt_cache_hit_tokens"].(float64); ok {
			out.CacheHit = int(v)
		}
		if v, ok := s.usage["prompt_cache_miss_tokens"].(float64); ok {
			out.CacheMiss = int(v)
		}
		if v, ok := s.usage["prompt_cache_write_tokens"].(float64); ok {
			out.CacheWrite = int(v)
		}
		if v, ok := s.usage["credit"].(float64); ok {
			out.Credit, out.HasCredit = v, true
		}
	}
	return out
}

func intNumber(v any) int {
	n, _ := v.(float64)
	return int(n)
}

type chatSSEDecodeResult struct {
	SawDone     bool
	ValidEvents int
	Err         error
}

func decodeChatSSE(r io.Reader, onChunk func(map[string]any) error) chatSSEDecodeResult {
	br := bufio.NewReaderSize(r, 64*1024)
	result := chatSSEDecodeResult{}
	for {
		line, err := br.ReadString('\n')
		trimmed := strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(trimmed, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if payload == "[DONE]" {
				result.SawDone = true
				break
			}
			if payload != "" {
				var chunk map[string]any
				if jsonErr := json.Unmarshal([]byte(payload), &chunk); jsonErr != nil {
					result.Err = fmt.Errorf("invalid upstream SSE JSON: %w", jsonErr)
					break
				}
				result.ValidEvents++
				if onChunk != nil {
					if callbackErr := onChunk(chunk); callbackErr != nil {
						result.Err = callbackErr
						break
					}
				}
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				result.Err = err
			}
			break
		}
	}
	if result.Err == nil && result.ValidEvents == 0 {
		result.Err = errors.New("upstream stream contained no valid data events")
	}
	return result
}

type responsesRenderer struct {
	req     responsesRequest
	encoder *responseSSEEncoder
}

func (r *responsesRenderer) RenderStream(_ context.Context, rc io.ReadCloser, meta ChatRenderMeta) ChatOutcome {
	state, stats, decoded := decodeResponsesUpstream(r.req, rc, meta.Started, r.encoder)
	state.priorUsage = meta.PriorUsage
	state.finish(decoded.SawDone, decoded.Err)
	out := state.outcome()
	out.TTFB = stats.TTFB()
	if decoded.Err != nil {
		out.ReadErr = decoded.Err
	}
	return out
}

func (r *responsesRenderer) RenderNonStream(_ context.Context, rc io.ReadCloser, meta ChatRenderMeta) ChatOutcome {
	state, _, decoded := decodeResponsesUpstream(r.req, rc, time.Time{}, nil)
	state.priorUsage = meta.PriorUsage
	state.finish(decoded.SawDone, decoded.Err)
	out := state.outcome()
	if decoded.Err != nil {
		out.ReadErr = decoded.Err
	}
	return out
}

// responseDoneReader records whether the upstream itself sent [DONE]. The
// Panel's StreamHint helper adds a defensive [DONE] when an upstream truncates
// a stream; Responses must retain the original signal so a truncated response
// cannot be reported as completed.
type responseDoneReader struct {
	r       io.Reader
	tail    []byte
	sawDone bool
}

func (r *responseDoneReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	if n > 0 {
		data := append(r.tail, p[:n]...)
		if bytes.Contains(data, []byte("data: [DONE]")) {
			r.sawDone = true
		}
		if len(data) > 32 {
			r.tail = append(r.tail[:0], data[len(data)-32:]...)
		} else {
			r.tail = append(r.tail[:0], data...)
		}
	}
	return n, err
}

type responsePipeWriter struct {
	pipe   *io.PipeWriter
	header http.Header
}

func (w *responsePipeWriter) Header() http.Header { return w.header }
func (w *responsePipeWriter) WriteHeader(int)     {}
func (w *responsePipeWriter) Write(p []byte) (int, error) {
	return w.pipe.Write(p)
}
func (w *responsePipeWriter) Flush() {}

func decodeResponsesUpstream(req responsesRequest, rc io.Reader, started time.Time, encoder *responseSSEEncoder) (*responseState, *chatStatsReader, chatSSEDecodeResult) {
	tracked := &responseDoneReader{r: rc}
	statsStart := started
	if statsStart.IsZero() {
		statsStart = time.Now()
	}
	stats := newChatStatsReaderSince(tracked, statsStart)
	state := newResponseState(req, encoder != nil, encoder)
	state.start()
	reader, writer := io.Pipe()
	go func() {
		repair := upstream.NewMarkupRepair(upstream.ToolNameAllowlist(req.ChatBody), true)
		err := upstream.StreamHint(&responsePipeWriter{pipe: writer, header: make(http.Header)}, stats, nil, upstream.WithMarkupRepair(repair))
		_ = writer.CloseWithError(err)
	}()
	decoded := decodeChatSSE(reader, state.consumeChunk)
	_ = reader.CloseWithError(decoded.Err)
	decoded.SawDone = tracked.sawDone
	if !tracked.sawDone && decoded.Err == nil {
		decoded.Err = errors.New("upstream stream ended before [DONE]")
	}
	return state, stats, decoded
}

func (h *Handler) responses(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeResponsesValidationError(w, responseBad("", "read body: "+err.Error()))
		return
	}
	req, err := parseResponsesRequest(raw)
	if err != nil {
		writeResponsesValidationError(w, err)
		return
	}
	encoder := &responseSSEEncoder{}
	if req.Stream {
		encoder.w = w
		encoder.flush = func() {
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
		encoder.prepare = func() {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
		}
	}
	clientIP := ""
	if h.cfg.Upstream != nil && h.cfg.Upstream.PassthroughIP {
		clientIP = upstream.ExtractClientIP(r)
	}
	out, failure := h.executeChat(ChatExecutionRequest{
		Context:                    r.Context(),
		Body:                       req.ChatBody,
		Model:                      req.Model,
		Stream:                     req.Stream,
		Headers:                    r.Header,
		ClientIP:                   clientIP,
		ResponsesDiagnostics:       true,
		ResponsesCandidateSections: responsesCandidateSections(req),
	}, &responsesRenderer{req: req, encoder: encoderIf(req.Stream, encoder)})
	if failure != nil {
		writeResponsesExecutionError(w, failure)
		return
	}
	if req.Stream {
		return
	}
	if out.Response != nil {
		writeJSON(w, http.StatusOK, out.Response)
		return
	}
	if out.ReadErr != nil {
		writeOpenAIError(w, http.StatusBadGateway, "upstream_parse", out.ReadErr.Error())
		return
	}
	writeOpenAIError(w, http.StatusBadGateway, "upstream_parse", "upstream response was empty")
}

func responsesCandidateSections(req responsesRequest) []string {
	var body map[string]json.RawMessage
	if json.Unmarshal(req.ChatBody, &body) != nil || body == nil {
		return []string{"model selection", "input/history"}
	}
	sections := make([]string, 0, 12)
	add := func(key, label string) {
		if _, ok := body[key]; ok {
			sections = append(sections, label)
		}
	}
	add("model", "model selection")
	add("messages", "input/history")
	add("tools", "tools")
	add("tool_choice", "tool_choice")
	if _, effort := body["reasoning_effort"]; effort {
		sections = append(sections, "reasoning")
	} else if _, summary := body["reasoning_summary"]; summary {
		sections = append(sections, "reasoning")
	}
	add("response_format", "text.format")
	add("max_completion_tokens", "output limits")
	add("temperature", "temperature")
	add("top_p", "top_p")
	add("metadata", "metadata")
	add("prompt_cache_key", "prompt caching")
	add("parallel_tool_calls", "parallel tool calls")
	return sections
}

func responsesUnsupportedFieldHint(sections []string) string {
	if len(sections) == 0 {
		return "Review the request fields sent to the upstream. Request values and message content are omitted."
	}
	return "Review these present request sections: " + strings.Join(sections, ", ") + ". Request values and message content are omitted."
}

func responsesUnsupportedUpstreamField(status int, body []byte, ue *upstream.Error) (string, bool) {
	if status != http.StatusBadRequest {
		return "", false
	}
	raw := string(body)
	if len(body) == 0 && ue != nil {
		raw = ue.Msg
	}
	var envelope map[string]any
	_ = json.Unmarshal([]byte(raw), &envelope)
	errorObject, _ := envelope["error"].(map[string]any)
	code := firstStringValue(errorObject["code"], envelope["code"])
	message := strings.ToLower(firstStringValue(errorObject["message"], envelope["message"], envelope["msg"]))
	lower := strings.ToLower(raw)
	unsupported := strings.EqualFold(strings.TrimSpace(code), "unsupported_value") ||
		strings.Contains(message, "field is not supported") ||
		strings.Contains(message, "unsupported field") ||
		strings.Contains(message, "field is unsupported") ||
		strings.Contains(lower, "field is not supported") ||
		strings.Contains(lower, "unsupported field") ||
		strings.Contains(lower, "field is unsupported")
	if !unsupported {
		return "", false
	}
	param := firstStringValue(errorObject["param"], envelope["param"])
	return mapUpstreamResponsesParam(param), true
}

func firstStringValue(values ...any) string {
	for _, value := range values {
		if s, ok := value.(string); ok && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s)
		}
	}
	return ""
}

func mapUpstreamResponsesParam(param string) string {
	param = strings.TrimSpace(param)
	if param == "" || len(param) > 160 {
		return ""
	}
	pathIs := func(root string) bool {
		return param == root || strings.HasPrefix(param, root+".") || strings.HasPrefix(param, root+"[")
	}
	switch {
	case pathIs("input") || pathIs("messages"):
		return "input"
	case pathIs("tools"):
		return "tools"
	case pathIs("tool_choice"):
		return "tool_choice"
	case pathIs("reasoning_effort"):
		return "reasoning.effort"
	case pathIs("reasoning_summary"):
		return "reasoning.summary"
	case pathIs("reasoning.effort"):
		return "reasoning.effort"
	case pathIs("reasoning.summary"):
		return "reasoning.summary"
	case pathIs("reasoning"):
		return "reasoning"
	case pathIs("response_format"):
		return "text.format"
	case pathIs("max_completion_tokens") || pathIs("max_tokens"):
		return "max_output_tokens"
	case pathIs("metadata"):
		return "metadata"
	case pathIs("prompt_cache_key"):
		return "prompt_cache_key"
	case pathIs("parallel_tool_calls"):
		return "parallel_tool_calls"
	case pathIs("text.format"):
		return "text.format"
	case pathIs("text.verbosity"):
		return "text.verbosity"
	case pathIs("instructions"), pathIs("model"), pathIs("stream"), pathIs("temperature"), pathIs("top_p"),
		pathIs("max_output_tokens"), pathIs("text"):
		return strings.SplitN(param, ".", 2)[0]
	default:
		return ""
	}
}

func writeResponsesExecutionError(w http.ResponseWriter, failure *chatFailure) {
	errorBody := map[string]any{
		"message": failure.Message,
		"type":    "api_error",
		"code":    failure.Code,
	}
	if failure.Param != "" {
		errorBody["param"] = failure.Param
	}
	if failure.Hint != "" {
		errorBody["gateway_hint"] = failure.Hint
	}
	writeJSON(w, failure.Status, map[string]any{"error": errorBody})
}

func encoderIf(stream bool, encoder *responseSSEEncoder) *responseSSEEncoder {
	if !stream {
		return nil
	}
	return encoder
}

func writeResponsesValidationError(w http.ResponseWriter, err error) {
	status := http.StatusBadRequest
	code := "invalid_request_error"
	param := ""
	message := err.Error()
	var validation *responsesValidationError
	if errors.As(err, &validation) {
		code = validation.Code
		param = validation.Param
		message = validation.Message
	}
	payload := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "invalid_request_error",
			"code":    code,
		},
	}
	if param != "" {
		payload["error"].(map[string]any)["param"] = param
	}
	writeJSON(w, status, payload)
}
