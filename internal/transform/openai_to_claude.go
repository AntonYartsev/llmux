package transform

import (
	"encoding/json"
	"strings"

	"llmux/internal/config"
)

// prepended system prompt when targeting Claude
const ClaudeSystemPrefix = "You are Claude Code, Anthropic's official CLI for Claude."

// Anthropic Messages API endpoint
const ClaudeAPIURL = "https://api.anthropic.com/v1/messages"

// returns the HTTP headers required to call the Anthropic Messages API with OAuth bearer auth and the appropriate beta features
func ClaudeHeaders(token string) map[string]string {
	return map[string]string{
		"Authorization":     "Bearer " + token,
		"anthropic-version": "2023-06-01",
		"anthropic-beta":    "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
		"user-agent":        "claude-cli/2.1.75",
		"x-app":             "cli",
		"content-type":      "application/json",
		"anthropic-dangerous-direct-browser-access": "true",
	}
}

// splits a data URI ("data:image/png;base64,xxxx") into (mediaType, base64Data)
// returns ("", "") if the URI is not a valid data URI
func parseDataURI(uri string) (string, string) {
	if !strings.HasPrefix(uri, "data:") {
		return "", ""
	}
	rest := uri[len("data:"):]
	headerAndData := strings.SplitN(rest, ",", 2)
	if len(headerAndData) != 2 {
		return "", ""
	}
	header := headerAndData[0] // e.g. "image/png;base64"
	data := headerAndData[1]
	mediaType := strings.SplitN(header, ";", 2)[0]
	return mediaType, data
}

// converts an OpenAI content-parts array into Claude content blocks
func claudeContentFromParts(parts []interface{}) []map[string]any {
	var blocks []map[string]any
	for _, p := range parts {
		pm, ok := p.(map[string]interface{})
		if !ok {
			continue
		}
		switch pm["type"] {
		case "text":
			text, _ := pm["text"].(string)
			blocks = append(blocks, map[string]any{"type": "text", "text": text})
		case "image_url":
			imageURL, _ := pm["image_url"].(map[string]interface{})
			if imageURL == nil {
				continue
			}
			rawURL, _ := imageURL["url"].(string)
			if rawURL == "" {
				continue
			}
			mediaType, b64 := parseDataURI(rawURL)
			if mediaType == "" {
				// not a data URI – skip (Claude cannot fetch arbitrary URLs here)
				continue
			}
			if mediaType == "" {
				mediaType = "image/png"
			}
			blocks = append(blocks, map[string]any{
				"type": "image",
				"source": map[string]any{
					"type":       "base64",
					"media_type": mediaType,
					"data":       b64,
				},
			})
		}
	}
	return blocks
}

// converts ChatRequest into an Anthropic Messages API request body (as map[string]any, ready for JSON serialisation)
func OpenAIRequestToClaude(req *ChatRequest) map[string]any {
	result := make(map[string]any)

	// --- System messages ---
	var systemParts []string
	var messages []map[string]any

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "developer":
			text := messageContentText(msg.Content)
			if text != "" {
				systemParts = append(systemParts, text)
			}

		case "tool":
			// tool result -> user message with tool_result content block
			text := messageContentText(msg.Content)
			block := map[string]any{
				"type":    "tool_result",
				"content": text,
			}
			if msg.ToolCallID != nil {
				block["tool_use_id"] = *msg.ToolCallID
			}
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": []any{block},
			})

		case "assistant":
			var content []any

			// any text in the message body
			text := messageContentText(msg.Content)
			if text != "" {
				content = append(content, map[string]any{"type": "text", "text": text})
			}

			// tool_calls -> tool_use blocks
			for _, tc := range msg.ToolCalls {
				id, _ := tc["id"].(string)
				fn, _ := tc["function"].(map[string]interface{})
				if fn == nil {
					fn = map[string]interface{}{}
				}
				name, _ := fn["name"].(string)

				// parse arguments JSON string into a map.
				var input map[string]interface{}
				switch a := fn["arguments"].(type) {
				case string:
					json.Unmarshal([]byte(a), &input) //nolint:errcheck
				case map[string]interface{}:
					input = a
				}
				if input == nil {
					input = map[string]interface{}{}
				}

				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    id,
					"name":  name,
					"input": input,
				})
			}

			// if content is an array of parts (not already consumed via text)
			if len(msg.ToolCalls) == 0 {
				if parts, ok := msg.Content.([]interface{}); ok {
					blocks := claudeContentFromParts(parts)
					for _, b := range blocks {
						content = append(content, b)
					}
				}
			}

			if len(content) == 0 {
				content = []any{map[string]any{"type": "text", "text": ""}}
			}
			messages = append(messages, map[string]any{
				"role":    "assistant",
				"content": content,
			})

		default: // "user"
			var content []any
			switch c := msg.Content.(type) {
			case string:
				content = []any{map[string]any{"type": "text", "text": c}}
			case []interface{}:
				blocks := claudeContentFromParts(c)
				for _, b := range blocks {
					content = append(content, b)
				}
				if len(content) == 0 {
					content = []any{map[string]any{"type": "text", "text": ""}}
				}
			default:
				content = []any{map[string]any{"type": "text", "text": ""}}
			}
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": content,
			})
		}
	}

	// Claude API accepts system as an array of content blocks
	var systemBlocks []map[string]any
	systemBlocks = append(systemBlocks, map[string]any{
		"type": "text",
		"text": ClaudeSystemPrefix,
	})
	if len(systemParts) > 0 {
		systemBlocks = append(systemBlocks, map[string]any{
			"type": "text",
			"text": strings.Join(systemParts, "\n\n"),
		})
	}
	result["system"] = systemBlocks

	result["messages"] = messages

	// strip vendor prefix (e.g. "claude/claude-sonnet-4-6" -> "claude-sonnet-4-6")
	_, bareModel := config.ParsePrefixedModel(req.Model)
	result["model"] = bareModel

	if mt := req.EffectiveMaxTokens(); mt != nil && *mt > 0 {
		result["max_tokens"] = *mt
	} else {
		result["max_tokens"] = 8096
	}

	if req.Temperature != nil && *req.Temperature > 0 {
		result["temperature"] = *req.Temperature
	}
	if req.TopP != nil && *req.TopP > 0 {
		result["top_p"] = *req.TopP
	}

	// stop sequences — Stop is interface{} (string or []string / []interface{})
	if req.Stop != nil {
		switch s := req.Stop.(type) {
		case string:
			if s != "" {
				result["stop_sequences"] = []string{s}
			}
		case []string:
			if len(s) > 0 {
				result["stop_sequences"] = s
			}
		case []interface{}:
			var seqs []string
			for _, sv := range s {
				if ss, ok := sv.(string); ok {
					seqs = append(seqs, ss)
				}
			}
			if len(seqs) > 0 {
				result["stop_sequences"] = seqs
			}
		}
	}

	if req.ReasoningEffort != nil && *req.ReasoningEffort != "" {
		effort := *req.ReasoningEffort
		// normalise to one of the known Claude effort values
		switch effort {
		case "low", "medium", "high", "max":
			// valid as-is
		case "none", "minimal":
			effort = "low"
		case "xhigh":
			effort = "max"
		default:
			effort = "medium"
		}
		result["thinking"] = map[string]any{"type": "adaptive"}
		result["output_config"] = map[string]any{"effort": effort}
	}

	if len(req.Tools) > 0 {
		var claudeTools []map[string]any
		for _, tool := range req.Tools {
			if tool["type"] != "function" {
				continue
			}
			fn, _ := tool["function"].(map[string]interface{})
			if fn == nil {
				continue
			}
			name, _ := fn["name"].(string)
			if name == "" {
				continue
			}
			t := map[string]any{"name": name}
			if desc, ok := fn["description"].(string); ok {
				t["description"] = desc
			}
			if params, ok := fn["parameters"].(map[string]interface{}); ok {
				t["input_schema"] = params
			}
			claudeTools = append(claudeTools, t)
		}
		if len(claudeTools) > 0 {
			result["tools"] = claudeTools
		}
	}

	if req.User != nil && *req.User != "" {
		meta, _ := result["metadata"].(map[string]any)
		if meta == nil {
			meta = make(map[string]any)
		}
		meta["user_id"] = *req.User
		result["metadata"] = meta
	}

	return result
}
