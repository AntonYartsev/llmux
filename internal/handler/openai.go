package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"llmux/internal/config"
	"llmux/internal/transform"

	"github.com/gin-gonic/gin"
)

const defaultModel = "gemini-2.5-flash"

// returns the "owned_by" string for a model ID (supports vendor-prefixed IDs)
func ownerFor(id string) string {
	prefix, bare := config.ParsePrefixedModel(id)
	if prefix == "claude" {
		return "anthropic"
	}
	if prefix == "gemini" {
		return "google"
	}
	// fallback: inspect bare name
	lower := strings.ToLower(bare)
	if strings.Contains(lower, "claude") {
		return "anthropic"
	}
	return "google"
}

// builds the minimal OpenAI model object for a given model ID
func openaiModelObject(id string) gin.H {
	return gin.H{
		"id":       id,
		"object":   "model",
		"created":  0,
		"owned_by": ownerFor(id),
	}
}

// writes a standard OpenAI error JSON response
func writeOpenAIError(c *gin.Context, status int, err error) {
	errType := "internal_error"
	switch status {
	case http.StatusBadRequest:
		errType = "invalid_request_error"
	case http.StatusUnauthorized:
		errType = "authentication_error"
	case http.StatusForbidden:
		errType = "permission_error"
	case http.StatusNotFound:
		errType = "not_found_error"
	case http.StatusTooManyRequests:
		errType = "rate_limit_error"
	case http.StatusServiceUnavailable:
		errType = "service_unavailable_error"
	}
	c.JSON(status, gin.H{
		"error": gin.H{
			"message": err.Error(),
			"type":    errType,
			"param":   nil,
			"code":    nil,
		},
	})
}

// handles POST /v1/chat/completions
func ChatCompletions(r *Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req transform.ChatRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			writeOpenAIError(c, http.StatusBadRequest, err)
			return
		}

		model := req.Model
		if model == "" {
			model = defaultModel
		}

		if req.Stream {
			status, ch, actualModel, err := r.Stream(c.Request.Context(), model, &req)
			if err != nil {
				code := status
				if code == 0 {
					code = http.StatusInternalServerError
				}
				writeOpenAIError(c, code, err)
				return
			}

			c.Header("Content-Type", "text/event-stream")
			c.Header("Cache-Control", "no-cache")
			c.Header("Connection", "keep-alive")
			c.Header("X-Accel-Buffering", "no")
			if actualModel != model {
				c.Header("X-Fallback-Model", actualModel)
			}

			flusher, canFlush := c.Writer.(http.Flusher)
			for data := range ch {
				c.Writer.Write(data) //nolint:errcheck
				if canFlush {
					flusher.Flush()
				}
			}
			// streamWorker already appends "data: [DONE]\n\n"; no extra write needed
			return
		}

		// Non-streaming path.
		body, status, actualModel, err := r.Send(c.Request.Context(), model, &req)
		if err != nil {
			code := status
			if code == 0 {
				code = http.StatusInternalServerError
			}
			writeOpenAIError(c, code, err)
			return
		}
		if actualModel != model {
			c.Header("X-Fallback-Model", actualModel)
		}
		c.Data(status, "application/json; charset=utf-8", body)
	}
}

// handles GET /v1/models
func ListModels(r *Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		models := r.AllModels()
		data := make([]gin.H, 0, len(models))
		for _, m := range models {
			data = append(data, openaiModelObject(m.ID))
		}
		c.JSON(http.StatusOK, gin.H{
			"object": "list",
			"data":   data,
		})
	}
}

// handles GET /v1/models/:id
// accepts both prefixed ("gemini/gemini-2.5-pro") and bare ("gemini-2.5-pro") IDs
func GetModel(r *Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("id")
		// Gin wildcard params may carry a leading slash
		id = strings.TrimPrefix(id, "/")

		for _, m := range r.AllModels() {
			// exact match on prefixed ID, or bare-name fallback
			_, bare := config.ParsePrefixedModel(m.ID)
			if m.ID == id || bare == id {
				c.JSON(http.StatusOK, openaiModelObject(m.ID))
				return
			}
		}
		c.JSON(http.StatusNotFound, gin.H{
			"error": gin.H{
				"message": fmt.Sprintf("The model '%s' does not exist", id),
				"type":    "invalid_request_error",
				"param":   nil,
				"code":    "model_not_found",
			},
		})
	}
}

// converts Responses API input + instructions into ChatMessages
func responsesInputToMessages(input any, instructions string, tools []any) ([]transform.ChatMessage, string) {
	var messages []transform.ChatMessage

	if instructions != "" {
		messages = append(messages, transform.ChatMessage{Role: "system", Content: instructions})
	}

	switch rawInput := input.(type) {
	case string:
		messages = append(messages, transform.ChatMessage{Role: "user", Content: rawInput})
	case []any:
		for _, item := range rawInput {
			switch v := item.(type) {
			case string:
				messages = append(messages, transform.ChatMessage{Role: "user", Content: v})
			case map[string]any:
				itemType, _ := v["type"].(string)

				if itemType == "function_call_output" {
					callID, _ := v["call_id"].(string)
					output, _ := v["output"].(string)
					messages = append(messages, transform.ChatMessage{
						Role:       "tool",
						Content:    output,
						ToolCallID: &callID,
					})
					continue
				}

				role, _ := v["role"].(string)
				if role == "" {
					role = "user"
				}
				if content, ok := v["content"]; ok {
					messages = append(messages, transform.ChatMessage{Role: role, Content: content})
				} else if text, ok := v["text"].(string); ok {
					messages = append(messages, transform.ChatMessage{Role: role, Content: text})
				}
			}
		}
	default:
		if rawInput != nil {
			messages = append(messages, transform.ChatMessage{Role: "user", Content: fmt.Sprintf("%v", rawInput)})
		}
	}

	return messages, instructions
}

// converts Responses API tools to Chat Completions format
func responsesToolsToOpenAI(tools []any) []map[string]interface{} {
	var result []map[string]interface{}
	for _, t := range tools {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		toolType, _ := tm["type"].(string)
		if toolType != "function" {
			continue
		}
		fn := map[string]interface{}{}
		if name, ok := tm["name"].(string); ok {
			fn["name"] = name
		}
		if desc, ok := tm["description"].(string); ok {
			fn["description"] = desc
		}
		if params, ok := tm["parameters"]; ok {
			fn["parameters"] = params
		}
		if strict, ok := tm["strict"]; ok {
			fn["strict"] = strict
		}
		result = append(result, map[string]interface{}{
			"type":     "function",
			"function": fn,
		})
	}
	return result
}

// converts Chat Completions usage to Responses API usage format
func buildResponsesUsage(chatUsage map[string]any) map[string]any {
	if chatUsage == nil {
		return map[string]any{
			"input_tokens":          0,
			"output_tokens":         0,
			"total_tokens":          0,
			"input_tokens_details":  map[string]any{"cached_tokens": 0},
			"output_tokens_details": map[string]any{"reasoning_tokens": 0},
		}
	}

	result := map[string]any{
		"input_tokens":  toIntHandler(chatUsage["prompt_tokens"]),
		"output_tokens": toIntHandler(chatUsage["completion_tokens"]),
		"total_tokens":  toIntHandler(chatUsage["total_tokens"]),
	}

	if ptd, ok := chatUsage["prompt_tokens_details"].(map[string]any); ok {
		result["input_tokens_details"] = map[string]any{
			"cached_tokens": toIntHandler(ptd["cached_tokens"]),
		}
	} else {
		result["input_tokens_details"] = map[string]any{"cached_tokens": 0}
	}

	if ctd, ok := chatUsage["completion_tokens_details"].(map[string]any); ok {
		result["output_tokens_details"] = map[string]any{
			"reasoning_tokens": toIntHandler(ctd["reasoning_tokens"]),
		}
	} else {
		result["output_tokens_details"] = map[string]any{"reasoning_tokens": 0}
	}

	return result
}

// converts OpenAI tool_calls to Responses API function_call output items
func chatToolCallsToOutputItems(toolCalls []any) []any {
	var items []any
	for _, tc := range toolCalls {
		tcMap, ok := tc.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := tcMap["function"].(map[string]any)
		if fn == nil {
			continue
		}
		callID, _ := tcMap["id"].(string)
		name, _ := fn["name"].(string)
		args, _ := fn["arguments"].(string)
		items = append(items, map[string]any{
			"id":        callID,
			"type":      "function_call",
			"call_id":   callID,
			"name":      name,
			"arguments": args,
			"status":    "completed",
		})
	}
	return items
}

func toIntHandler(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

func nilIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

// handles POST /v1/responses (OpenAI Responses API)
func ResponsesAPI(r *Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		var data map[string]any
		if err := c.ShouldBindJSON(&data); err != nil {
			writeOpenAIError(c, http.StatusBadRequest, err)
			return
		}

		model, _ := data["model"].(string)
		if model == "" {
			model = defaultModel
		}

		stream, _ := data["stream"].(bool)
		instructions, _ := data["instructions"].(string)
		input := data["input"]

		// convert Responses API tools to Chat Completions format
		var chatTools []map[string]interface{}
		if rawTools, ok := data["tools"].([]any); ok {
			chatTools = responsesToolsToOpenAI(rawTools)
		}

		messages, _ := responsesInputToMessages(input, instructions, nil)

		req := transform.ChatRequest{
			Model:    model,
			Messages: messages,
			Stream:   stream,
		}

		if temp, ok := data["temperature"].(float64); ok {
			req.Temperature = &temp
		}
		if topP, ok := data["top_p"].(float64); ok {
			req.TopP = &topP
		}
		if maxOut, ok := data["max_output_tokens"].(float64); ok {
			v := int(maxOut)
			req.MaxCompletionTokens = &v
		}
		if stop, ok := data["stop"]; ok {
			req.Stop = stop
		}
		if fp, ok := data["frequency_penalty"].(float64); ok {
			req.FrequencyPenalty = &fp
		}
		if pp, ok := data["presence_penalty"].(float64); ok {
			req.PresencePenalty = &pp
		}
		if len(chatTools) > 0 {
			req.Tools = chatTools
		}
		if tc, ok := data["tool_choice"]; ok {
			req.ToolChoice = tc
		}
		if ptc, ok := data["parallel_tool_calls"].(bool); ok {
			req.ParallelToolCalls = &ptc
		}
		if user, ok := data["user"].(string); ok {
			req.User = &user
		}
		if st, ok := data["service_tier"].(string); ok {
			req.ServiceTier = &st
		}

		// reasoning: { effort: "..." }
		if reasoning, ok := data["reasoning"].(map[string]any); ok {
			if effort, ok := reasoning["effort"].(string); ok && effort != "" {
				req.ReasoningEffort = &effort
			}
		}

		// text.format -> response_format
		if text, ok := data["text"].(map[string]any); ok {
			if format, ok := text["format"].(map[string]any); ok {
				req.ResponseFormat = format
			}
		}

		if stream {
			handleResponsesStream(c, r, model, &req, data)
			return
		}

		body, status, actualModel, err := r.Send(c.Request.Context(), model, &req)
		if err != nil {
			code := status
			if code == 0 {
				code = http.StatusInternalServerError
			}
			writeOpenAIError(c, code, err)
			return
		}
		if actualModel != model {
			c.Header("X-Fallback-Model", actualModel)
		}

		var openaiResp map[string]any
		if err := json.Unmarshal(body, &openaiResp); err != nil {
			writeOpenAIError(c, http.StatusInternalServerError, err)
			return
		}

		outputText := ""
		var outputItems []any
		var toolCallItems []any

		if choices, ok := openaiResp["choices"].([]any); ok && len(choices) > 0 {
			if cm, ok := choices[0].(map[string]any); ok {
				if msg, ok := cm["message"].(map[string]any); ok {
					outputText, _ = msg["content"].(string)
					if tcs, ok := msg["tool_calls"].([]any); ok && len(tcs) > 0 {
						toolCallItems = chatToolCallsToOutputItems(tcs)
					}
				}
			}
		}

		responseID := "resp_" + strings.ReplaceAll(fmt.Sprintf("%x", time.Now().UnixNano()), "-", "")
		msgID := fmt.Sprintf("msg_%x", time.Now().UnixNano())

		msgItem := map[string]any{
			"type": "message",
			"id":   msgID,
			"role": "assistant",
			"content": []any{map[string]any{
				"type":        "output_text",
				"text":        outputText,
				"annotations": []any{},
			}},
			"status": "completed",
		}
		outputItems = append(outputItems, msgItem)
		outputItems = append(outputItems, toolCallItems...)

		chatUsage, _ := openaiResp["usage"].(map[string]any)
		responsesUsage := buildResponsesUsage(chatUsage)

		respBody := map[string]any{
			"id":                 responseID,
			"object":             "response",
			"created_at":         float64(time.Now().Unix()),
			"model":              actualModel,
			"status":             "completed",
			"error":              nil,
			"incomplete_details": nil,
			"instructions":       nilIfEmpty(instructions),
			"metadata":           data["metadata"],
			"output":             outputItems,
			"output_text":        outputText,
			"usage":              responsesUsage,
			"service_tier":       "default",
		}

		out, err := json.Marshal(respBody)
		if err != nil {
			writeOpenAIError(c, http.StatusInternalServerError, err)
			return
		}
		c.Data(status, "application/json; charset=utf-8", out)
	}
}

// handles streaming for the Responses API
func handleResponsesStream(c *gin.Context, r *Router, model string, req *transform.ChatRequest, data map[string]any) {
	status, ch, actualModel, err := r.Stream(c.Request.Context(), model, req)
	if err != nil {
		code := status
		if code == 0 {
			code = http.StatusInternalServerError
		}
		writeOpenAIError(c, code, err)
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")
	if actualModel != model {
		c.Header("X-Fallback-Model", actualModel)
	}

	responseID := "resp_" + strings.ReplaceAll(fmt.Sprintf("%x", time.Now().UnixNano()), "-", "")
	createdAt := float64(time.Now().Unix())
	msgID := fmt.Sprintf("msg_%x", time.Now().UnixNano())
	instructions, _ := data["instructions"].(string)

	flusher, canFlush := c.Writer.(http.Flusher)

	writeSSE := func(eventType string, payload any) {
		b, err := json.Marshal(payload)
		if err != nil {
			return
		}
		fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", eventType, b)
		if canFlush {
			flusher.Flush()
		}
	}

	// lifecycle events
	writeSSE("response.created", map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id":                 responseID,
			"object":             "response",
			"model":              actualModel,
			"status":             "in_progress",
			"created_at":         createdAt,
			"error":              nil,
			"incomplete_details": nil,
			"instructions":       nilIfEmpty(instructions),
			"metadata":           data["metadata"],
			"output":             []any{},
			"usage":              nil,
			"service_tier":       "default",
		},
	})
	writeSSE("response.in_progress", map[string]any{
		"type": "response.in_progress",
		"response": map[string]any{
			"id":         responseID,
			"object":     "response",
			"model":      actualModel,
			"status":     "in_progress",
			"created_at": createdAt,
			"output":     []any{},
		},
	})
	writeSSE("response.output_item.added", map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"type":    "message",
			"id":      msgID,
			"role":    "assistant",
			"content": []any{},
			"status":  "in_progress",
		},
	})
	writeSSE("response.content_part.added", map[string]any{
		"type":          "response.content_part.added",
		"output_index":  0,
		"content_index": 0,
		"part":          map[string]any{"type": "output_text", "text": ""},
	})

	// stream deltas
	var fullText strings.Builder
	var toolCallItems []any

	for raw := range ch {
		line := strings.TrimSpace(string(raw))
		if line == "data: [DONE]" || line == "" {
			continue
		}
		jsonPart := strings.TrimPrefix(line, "data: ")
		var chunk map[string]any
		if err := json.Unmarshal([]byte(jsonPart), &chunk); err != nil {
			continue
		}
		choices, _ := chunk["choices"].([]any)
		for _, ch := range choices {
			cm, _ := ch.(map[string]any)
			if cm == nil {
				continue
			}
			delta, _ := cm["delta"].(map[string]any)
			if delta == nil {
				continue
			}

			if text, _ := delta["content"].(string); text != "" {
				fullText.WriteString(text)
				writeSSE("response.output_text.delta", map[string]any{
					"type":          "response.output_text.delta",
					"output_index":  0,
					"content_index": 0,
					"delta":         text,
				})
			}

			if tcs, ok := delta["tool_calls"].([]any); ok {
				for _, tc := range tcs {
					tcMap, _ := tc.(map[string]any)
					if tcMap == nil {
						continue
					}
					fn, _ := tcMap["function"].(map[string]any)
					if fn == nil {
						continue
					}
					if id, ok := tcMap["id"].(string); ok && id != "" {
						name, _ := fn["name"].(string)
						toolCallItems = append(toolCallItems, map[string]any{
							"id":        id,
							"type":      "function_call",
							"call_id":   id,
							"name":      name,
							"arguments": "",
							"status":    "completed",
						})
					}
					if args, ok := fn["arguments"].(string); ok && args != "" && len(toolCallItems) > 0 {
						last := toolCallItems[len(toolCallItems)-1].(map[string]any)
						last["arguments"] = last["arguments"].(string) + args
					}

					writeSSE("response.function_call_arguments.delta", map[string]any{
						"type":         "response.function_call_arguments.delta",
						"output_index": len(toolCallItems) - 1,
						"delta":        fn["arguments"],
					})
				}
			}
		}
	}

	accumulated := fullText.String()

	// completion events
	writeSSE("response.output_text.done", map[string]any{
		"type":          "response.output_text.done",
		"output_index":  0,
		"content_index": 0,
		"text":          accumulated,
	})
	writeSSE("response.content_part.done", map[string]any{
		"type":          "response.content_part.done",
		"output_index":  0,
		"content_index": 0,
		"part":          map[string]any{"type": "output_text", "text": accumulated},
	})
	writeSSE("response.output_item.done", map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":    "message",
			"id":      msgID,
			"role":    "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": accumulated, "annotations": []any{}}},
			"status":  "completed",
		},
	})

	for i, item := range toolCallItems {
		writeSSE("response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"output_index": i + 1,
			"item":         item,
		})
	}

	var allOutput []any
	allOutput = append(allOutput, map[string]any{
		"type":    "message",
		"id":      msgID,
		"role":    "assistant",
		"content": []any{map[string]any{"type": "output_text", "text": accumulated, "annotations": []any{}}},
		"status":  "completed",
	})
	allOutput = append(allOutput, toolCallItems...)

	writeSSE("response.completed", map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":                 responseID,
			"object":             "response",
			"model":              actualModel,
			"status":             "completed",
			"created_at":         createdAt,
			"error":              nil,
			"incomplete_details": nil,
			"instructions":       nilIfEmpty(instructions),
			"metadata":           data["metadata"],
			"output":             allOutput,
			"output_text":        accumulated,
			"service_tier":       "default",
			"usage": map[string]any{
				"input_tokens":          0,
				"output_tokens":         0,
				"total_tokens":          0,
				"input_tokens_details":  map[string]any{"cached_tokens": 0},
				"output_tokens_details": map[string]any{"reasoning_tokens": 0},
			},
		},
	})
}
