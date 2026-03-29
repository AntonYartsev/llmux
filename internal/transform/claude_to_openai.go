package transform

import (
	"encoding/json"
	"fmt"
	"time"
)

// tracks state during Claude SSE streaming
type ClaudeStreamState struct {
	CurrentBlockType  string // "text", "thinking", "tool_use"
	CurrentBlockIndex int
	CurrentToolID     string
	CurrentToolName   string
	InputJSONBuffer   string
}

// converts a Claude stop_reason to an OpenAI finish_reason
func mapClaudeStopReason(stopReason string) string {
	switch stopReason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	default:
		return "stop"
	}
}

// generates a unique chat completion ID
func newResponseID() string {
	return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}

// converts a non-streaming Claude response to OpenAI format
//
// input Claude format:
//
//	{"id": "...", "type": "message", "role": "assistant", "content": [...],
//	 "stop_reason": "...", "usage": {...}}
//
// content block types handled:
//   - {"type": "text", "text": "..."}           -> content string
//   - {"type": "thinking", "thinking": "..."}   -> reasoning_content string
//   - {"type": "tool_use", "id": "...", "name": "...", "input": {...}} -> tool_calls entry
func ClaudeResponseToOpenAI(claudeResp map[string]any, model string) map[string]any {
	var contentParts []string
	var reasoningContent string
	var toolCalls []map[string]any

	if blocks, ok := claudeResp["content"].([]any); ok {
		for _, b := range blocks {
			block, ok := b.(map[string]any)
			if !ok {
				continue
			}
			blockType, _ := block["type"].(string)
			switch blockType {
			case "text":
				if text, ok := block["text"].(string); ok {
					contentParts = append(contentParts, text)
				}
			case "thinking":
				if thinking, ok := block["thinking"].(string); ok {
					reasoningContent += thinking
				}
			case "tool_use":
				id, _ := block["id"].(string)
				name, _ := block["name"].(string)
				input := block["input"]

				var argsJSON string
				if inputBytes, err := json.Marshal(input); err == nil {
					argsJSON = string(inputBytes)
				} else {
					argsJSON = "{}"
				}

				toolCalls = append(toolCalls, map[string]any{
					"id":   id,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": argsJSON,
					},
				})
			}
		}
	}

	// join text parts
	content := ""
	for i, p := range contentParts {
		if i > 0 {
			content += "\n\n"
		}
		content += p
	}

	stopReason, _ := claudeResp["stop_reason"].(string)
	finishReason := mapClaudeStopReason(stopReason)

	// build message object
	message := map[string]any{
		"role":    "assistant",
		"refusal": nil,
	}
	// OpenAI returns content: null when only tool_calls are present
	if len(toolCalls) > 0 && content == "" {
		message["content"] = nil
	} else {
		message["content"] = content
	}
	if reasoningContent != "" {
		message["reasoning_content"] = reasoningContent
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	// build usage object
	inputTokens := 0
	outputTokens := 0
	cachedTokens := 0
	if usageIn, ok := claudeResp["usage"].(map[string]any); ok {
		inputTokens = toInt(usageIn["input_tokens"])
		outputTokens = toInt(usageIn["output_tokens"])
		cachedTokens = toInt(usageIn["cache_read_input_tokens"])
	}

	usageOut := map[string]any{
		"prompt_tokens":     inputTokens,
		"completion_tokens": outputTokens,
		"total_tokens":      inputTokens + outputTokens,
		"prompt_tokens_details": map[string]any{
			"cached_tokens": cachedTokens,
			"audio_tokens":  0,
		},
		"completion_tokens_details": map[string]any{
			"reasoning_tokens":           0,
			"audio_tokens":               0,
			"accepted_prediction_tokens": 0,
			"rejected_prediction_tokens": 0,
		},
	}

	return map[string]any{
		"id":      newResponseID(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"logprobs":      nil,
				"finish_reason": finishReason,
			},
		},
		"usage":              usageOut,
		"system_fingerprint": nil,
		"service_tier":       "default",
	}
}

// converts a single Claude SSE event into zero or more
// OpenAI streaming chunk maps.
//
// eventType is the value from the "event: <type>" SSE line
// state is mutated in place to track cross-event streaming context
func ClaudeStreamEventToOpenAI(
	event map[string]any,
	eventType string,
	model string,
	responseID string,
	state *ClaudeStreamState,
) []map[string]any {
	ts := time.Now().Unix()

	makeChunk := func(delta map[string]any, finishReason any) map[string]any {
		return map[string]any{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": ts,
			"model":   model,
			"choices": []any{
				map[string]any{
					"index":         0,
					"delta":         delta,
					"logprobs":      nil,
					"finish_reason": finishReason,
				},
			},
			"system_fingerprint": nil,
		}
	}

	switch eventType {
	case "message_start":
		// emit role announcement chunk
		return []map[string]any{
			makeChunk(map[string]any{"role": "assistant", "content": ""}, nil),
		}

	case "content_block_start":
		contentBlock, _ := event["content_block"].(map[string]any)
		if contentBlock == nil {
			return nil
		}
		blockType, _ := contentBlock["type"].(string)
		index := toInt(event["index"])
		state.CurrentBlockIndex = index
		state.CurrentBlockType = blockType

		switch blockType {
		case "tool_use":
			state.CurrentToolID, _ = contentBlock["id"].(string)
			state.CurrentToolName, _ = contentBlock["name"].(string)
			state.InputJSONBuffer = ""

			delta := map[string]any{
				"tool_calls": []any{
					map[string]any{
						"index": index,
						"id":    state.CurrentToolID,
						"type":  "function",
						"function": map[string]any{
							"name":      state.CurrentToolName,
							"arguments": "",
						},
					},
				},
			}
			return []map[string]any{makeChunk(delta, nil)}

		default:
			// "text" or "thinking": emit empty delta to signal block start
			return []map[string]any{makeChunk(map[string]any{}, nil)}
		}

	case "content_block_delta":
		deltaObj, _ := event["delta"].(map[string]any)
		if deltaObj == nil {
			return nil
		}
		deltaType, _ := deltaObj["type"].(string)
		index := toInt(event["index"])

		switch deltaType {
		case "text_delta":
			text, _ := deltaObj["text"].(string)
			return []map[string]any{
				makeChunk(map[string]any{"content": text}, nil),
			}

		case "thinking_delta":
			thinking, _ := deltaObj["thinking"].(string)
			return []map[string]any{
				makeChunk(map[string]any{"reasoning_content": thinking}, nil),
			}

		case "input_json_delta":
			partialJSON, _ := deltaObj["partial_json"].(string)
			state.InputJSONBuffer += partialJSON

			delta := map[string]any{
				"tool_calls": []any{
					map[string]any{
						"index": index,
						"function": map[string]any{
							"arguments": partialJSON,
						},
					},
				},
			}
			return []map[string]any{makeChunk(delta, nil)}
		}
		return nil

	case "content_block_stop":
		return nil

	case "message_delta":
		deltaObj, _ := event["delta"].(map[string]any)
		stopReason := ""
		if deltaObj != nil {
			stopReason, _ = deltaObj["stop_reason"].(string)
		}
		finishReason := mapClaudeStopReason(stopReason)

		chunk := makeChunk(map[string]any{}, finishReason)

		// include usage if present in the event
		if usage, ok := event["usage"].(map[string]any); ok {
			outputTokens := toInt(usage["output_tokens"])
			chunk["usage"] = map[string]any{
				"prompt_tokens":     0,
				"completion_tokens": outputTokens,
				"total_tokens":      outputTokens,
				"prompt_tokens_details": map[string]any{
					"cached_tokens": 0,
					"audio_tokens":  0,
				},
				"completion_tokens_details": map[string]any{
					"reasoning_tokens":           0,
					"audio_tokens":               0,
					"accepted_prediction_tokens": 0,
					"rejected_prediction_tokens": 0,
				},
			}
		}

		return []map[string]any{chunk}

	case "message_stop":
		return nil

	default:
		return nil
	}
}

// toInt converts a JSON numeric value (float64 from json.Unmarshal) or int to int
func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case float32:
		return int(n)
	}
	return 0
}
