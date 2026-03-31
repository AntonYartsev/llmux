package transform

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// converts a Gemini finishReason string to an OpenAI finish_reason
func mapFinishReason(geminiReason string) *string {
	var s string
	switch geminiReason {
	case "STOP":
		s = "stop"
	case "MAX_TOKENS":
		s = "length"
	case "SAFETY", "RECITATION":
		s = "content_filter"
	case "":
		return nil
	default:
		s = "stop"
	}
	return &s
}

// processes the parts slice from a Gemini candidate content object and
// returns the joined text content, accumulated reasoning_content, and any tool_calls
// streaming parameter controls whether tool_call entries include an "index" field
func extractParts(parts []any, streaming bool) (content string, reasoningContent string, toolCalls []map[string]any) {
	var contentParts []string

	for i, p := range parts {
		part, ok := p.(map[string]any)
		if !ok {
			continue
		}

		// text / thought parts
		if textVal, hasText := part["text"]; hasText {
			text, _ := textVal.(string)
			thought, _ := part["thought"].(bool)
			if thought {
				reasoningContent += text
			} else {
				contentParts = append(contentParts, text)
			}
			continue
		}

		// function call parts
		if fcVal, hasFc := part["functionCall"]; hasFc {
			fc, ok := fcVal.(map[string]any)
			if !ok {
				continue
			}
			name, _ := fc["name"].(string)
			args := fc["args"]
			argsJSON := "{}"
			if b, err := json.Marshal(args); err == nil {
				argsJSON = string(b)
			}

			entry := map[string]any{
				"id":   fmt.Sprintf("call_%d", time.Now().UnixNano()),
				"type": "function",
				"function": map[string]any{
					"name":      name,
					"arguments": argsJSON,
				},
			}
			if streaming {
				entry["index"] = i
			}
			toolCalls = append(toolCalls, entry)
			continue
		}

		// inline image data parts
		if inlineVal, hasInline := part["inlineData"]; hasInline {
			inline, ok := inlineVal.(map[string]any)
			if !ok {
				continue
			}
			data, _ := inline["data"].(string)
			if data == "" {
				continue
			}
			mime, _ := inline["mimeType"].(string)
			if mime == "" {
				mime = "image/png"
			}
			if strings.HasPrefix(mime, "image/") {
				contentParts = append(contentParts, fmt.Sprintf("![image](data:%s;base64,%s)", mime, data))
			}
		}
	}

	// filter out empty strings before joining (mirrors Python list comprehension)
	var nonEmpty []string
	for _, p := range contentParts {
		if p != "" {
			nonEmpty = append(nonEmpty, p)
		}
	}
	content = strings.Join(nonEmpty, "\n\n")
	return content, reasoningContent, toolCalls
}

// transforms a non-streaming Gemini API response into
// OpenAI chat.completion format
//
// input is the parsed Gemini response map (after any envelope unwrapping)
// model is passed through to the output
func GeminiResponseToOpenAI(geminiResp map[string]any, model string) map[string]any {
	ts := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	var choices []any

	candidates, _ := geminiResp["candidates"].([]any)
	for _, cand := range candidates {
		candidate, ok := cand.(map[string]any)
		if !ok {
			continue
		}

		candidateContent, _ := candidate["content"].(map[string]any)
		var parts []any
		if candidateContent != nil {
			parts, _ = candidateContent["parts"].([]any)
		}

		content, reasoningContent, toolCalls := extractParts(parts, false)

		finishReasonStr, _ := candidate["finishReason"].(string)
		var finishReason any
		if len(toolCalls) > 0 {
			fr := "tool_calls"
			finishReason = fr
		} else {
			finishReason = mapFinishReason(finishReasonStr)
		}

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

		idx := toInt(candidate["index"])
		choices = append(choices, map[string]any{
			"index":         idx,
			"message":       message,
			"logprobs":      nil,
			"finish_reason": finishReason,
		})
	}

	// build usage from usageMetadata
	prompt := 0
	completion := 0
	total := 0
	cachedTokens := 0
	if meta, ok := geminiResp["usageMetadata"].(map[string]any); ok {
		prompt = toInt(meta["promptTokenCount"])
		completion = toInt(meta["candidatesTokenCount"])
		total = toInt(meta["totalTokenCount"])
		cachedTokens = toInt(meta["cachedContentTokenCount"])
		if total == 0 {
			total = prompt + completion
		}
	}

	usageOut := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"total_tokens":      total,
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
		"id":                 id,
		"object":             "chat.completion",
		"created":            ts,
		"model":              model,
		"choices":            choices,
		"usage":              usageOut,
		"service_tier":       "default",
		"system_fingerprint": nil,
	}
}

// transforms a single Gemini streaming response chunk
// into OpenAI chat.completion.chunk format
//
// responseID must be consistent across all chunks for a single streaming response
func GeminiStreamChunkToOpenAI(chunk map[string]any, model string, responseID string) map[string]any {
	ts := time.Now().Unix()

	var choices []any

	candidates, _ := chunk["candidates"].([]any)
	for _, cand := range candidates {
		candidate, ok := cand.(map[string]any)
		if !ok {
			continue
		}

		candidateContent, _ := candidate["content"].(map[string]any)
		var parts []any
		if candidateContent != nil {
			parts, _ = candidateContent["parts"].([]any)
		}

		content, reasoningContent, toolCalls := extractParts(parts, true)

		finishReasonStr, _ := candidate["finishReason"].(string)
		var finishReason any
		if len(toolCalls) > 0 {
			fr := "tool_calls"
			finishReason = fr
		} else {
			finishReason = mapFinishReason(finishReasonStr)
		}

		delta := map[string]any{}
		if content != "" {
			delta["content"] = content
		}
		if reasoningContent != "" {
			delta["reasoning_content"] = reasoningContent
		}
		if len(toolCalls) > 0 {
			delta["tool_calls"] = toolCalls
		}

		idx := toInt(candidate["index"])
		choices = append(choices, map[string]any{
			"index":         idx,
			"delta":         delta,
			"logprobs":      nil,
			"finish_reason": finishReason,
		})
	}

	return map[string]any{
		"id":                 responseID,
		"object":             "chat.completion.chunk",
		"created":            ts,
		"model":              model,
		"choices":            choices,
		"service_tier":       "default",
		"system_fingerprint": nil,
	}
}
