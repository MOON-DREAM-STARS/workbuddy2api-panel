package server

import (
	"crypto/sha256"
	"fmt"
)

// Names are request-scoped and deterministic across stateless history replay.
// The Chat upstream only sees ordinary function names, never namespace objects.
type responseToolIdentity struct{ Namespace, Name string }
type responseToolNames struct {
	byWire map[string]responseToolIdentity
}

func newResponseToolNames() *responseToolNames {
	return &responseToolNames{byWire: map[string]responseToolIdentity{}}
}
func responseNamespace(item map[string]any, param string) (string, error) {
	v := item["namespace"]
	if v == nil {
		return "", nil
	}
	ns, ok := v.(string)
	if !ok {
		return "", responseBad(param, "must be a string or null")
	}
	return ns, nil
}
func (n *responseToolNames) wire(ns, name, param string) (string, error) {
	identity := responseToolIdentity{ns, name}
	wire := name
	if ns != "" {
		sum := sha256.Sum256([]byte(ns + "\x00" + name))
		wire = fmt.Sprintf("wbns_%x", sum[:24])
	}
	if old, ok := n.byWire[wire]; ok && old != identity {
		return "", responseBad(param, "tool name conflicts with another tool")
	}
	n.byWire[wire] = identity
	return wire, nil
}
func (n *responseToolNames) restore(item map[string]any, wire string) {
	item["name"] = wire
	if n == nil {
		return
	}
	identity, ok := n.byWire[wire]
	if !ok {
		return
	}
	item["name"] = identity.Name
	if identity.Namespace != "" {
		item["namespace"] = identity.Namespace
	}
}
func (n *responseToolNames) convert(v any) ([]any, []any, error) {
	items, ok := v.([]any)
	if !ok {
		return nil, nil, responseBad("tools", "must be an array")
	}
	var chat, public []any
	declared := map[string]bool{}
	add := func(tool map[string]any, ns, path string) error {
		converted, err := convertFunctionTools([]any{tool})
		if err != nil {
			if ve, ok := err.(*responsesValidationError); ok {
				copy := *ve
				copy.Param = path + copy.Param[len("tools[0]"):]
				return &copy
			}
			return err
		}
		if len(converted) == 0 {
			return responseUnsupported(path+".type", "namespace members must be function tools")
		}
		fn := converted[0].(map[string]any)["function"].(map[string]any)
		wire, err := n.wire(ns, fn["name"].(string), path+".name")
		if err != nil {
			return err
		}
		if declared[wire] {
			return responseBad(path+".name", "duplicate function declaration")
		}
		declared[wire] = true
		if ns != "" {
			description, _ := fn["description"].(string)
			fn["description"] = fmt.Sprintf("Namespace: %s; function: %s.\n%s", ns, fn["name"], description)
		}
		fn["name"] = wire
		chat = append(chat, converted[0])
		return nil
	}
	for i, raw := range items {
		path := fmt.Sprintf("tools[%d]", i)
		tool, ok := raw.(map[string]any)
		if !ok {
			return nil, nil, responseBad(path, "must be an object")
		}
		if tool["type"] != "namespace" {
			if tool["type"] == "function" {
				if err := add(tool, "", path); err != nil {
					return nil, nil, err
				}
				public = append(public, tool)
			} else if _, err := convertFunctionTools([]any{tool}); err != nil {
				if ve, ok := err.(*responsesValidationError); ok {
					copy := *ve
					copy.Param = path + copy.Param[len("tools[0]"):]
					return nil, nil, &copy
				}
				return nil, nil, err
			}
			continue
		}
		for key := range tool {
			if key != "type" && key != "name" && key != "description" && key != "tools" {
				return nil, nil, responseUnsupported(path+"."+key, "field is not supported")
			}
		}
		ns, ok := tool["name"].(string)
		if !ok || ns == "" {
			return nil, nil, responseBad(path+".name", "namespace name is required")
		}
		if d, exists := tool["description"]; exists {
			if _, ok := d.(string); !ok {
				return nil, nil, responseBad(path+".description", "must be a string")
			}
		}
		// Preserve compatibility with empty namespace placeholders from older clients.
		if _, exists := tool["tools"]; !exists {
			continue
		}
		members, ok := tool["tools"].([]any)
		if !ok {
			return nil, nil, responseBad(path+".tools", "must be an array")
		}
		for j, rawMember := range members {
			memberPath := fmt.Sprintf("%s.tools[%d]", path, j)
			member, ok := rawMember.(map[string]any)
			if !ok {
				return nil, nil, responseBad(memberPath, "must be an object")
			}
			if err := add(member, ns, memberPath); err != nil {
				return nil, nil, err
			}
		}
		if len(members) > 0 {
			public = append(public, tool)
		}
	}
	return chat, public, nil
}
func (n *responseToolNames) choice(v any) (any, any, error) {
	choice, ok := v.(map[string]any)
	if !ok {
		return convertToolChoice(v)
	}
	ns, err := responseNamespace(choice, "tool_choice.namespace")
	if err != nil {
		return nil, nil, err
	}
	flat := map[string]any{}
	for k, v := range choice {
		if k != "namespace" {
			flat[k] = v
		}
	}
	_, chat, err := convertToolChoice(flat)
	if err != nil {
		return nil, nil, err
	}
	wire, err := n.wire(ns, flat["name"].(string), "tool_choice.name")
	if err != nil {
		return nil, nil, err
	}
	chat.(map[string]any)["function"].(map[string]any)["name"] = wire
	return choice, chat, nil
}
