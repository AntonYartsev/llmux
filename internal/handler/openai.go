package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"llmux/internal/transform"

	"github.com/gin-gonic/gin"
)

const defaultModel = "gemini-2.5-flash"

// returns the "owned_by" string for a model ID
func ownerFor(id string) string {
	lower := strings.ToLower(id)
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
func GetModel(r *Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("id")
		// Gin wildcard params may carry a leading slash
		id = strings.TrimPrefix(id, "/")

		for _, m := range r.AllModels() {
			if m.ID == id {
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

// handles POST /v1/responses (OpenAI Responses API, 2025)
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

		var messages []transform.ChatMessage
		switch rawInput := data["input"].(type) {
		case string:
			messages = []transform.ChatMessage{{Role: "user", Content: rawInput}}
		case []any:
			for _, item := range rawInput {
				switch v := item.(type) {
				case string:
					messages = append(messages, transform.ChatMessage{Role: "user", Content: v})
				case map[string]any:
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
				messages = []transform.ChatMessage{{Role: "user", Content: fmt.Sprintf("%v", rawInput)}}
			}
		}

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
			req.MaxTokens = &v
		}
		if stop, ok := data["stop"]; ok {
			req.Stop = stop
		}
		if tools, ok := data["tools"].([]any); ok {
			for _, t := range tools {
				if tm, ok := t.(map[string]any); ok {
					req.Tools = append(req.Tools, tm)
				}
			}
		}

		// reasoning: { effort: "low" | "medium" | "high" | "minimal" }
		if reasoning, ok := data["reasoning"].(map[string]any); ok {
			if effort, ok := reasoning["effort"].(string); ok && effort != "" {
				req.ReasoningEffort = &effort
			}
		}
		// also accept top-level reasoning_effort string
		if re, ok := data["reasoning_effort"].(string); ok && re != "" {
			req.ReasoningEffort = &re
		}

		if stream {
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
			c.Header("X-Accel-Buffering", "no")
			if actualModel != model {
				c.Header("X-Fallback-Model", actualModel)
			}

			responseID := "resp_" + strings.ReplaceAll(fmt.Sprintf("%x", time.Now().UnixNano()), "-", "")
			createdAt := time.Now().Unix()
			msgID := fmt.Sprintf("msg_%x", time.Now().UnixNano())

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

			// emit lifecycle events
			writeSSE("response.created", map[string]any{
				"type": "response.created",
				"response": map[string]any{
					"id":         responseID,
					"object":     "response",
					"model":      model,
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

			// stream deltas: parse OpenAI SSE chunks from the router channel
			var fullText strings.Builder
			for raw := range ch {
				line := strings.TrimSpace(string(raw))
				// skip [DONE] and empty lines
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
					text, _ := delta["content"].(string)
					if text == "" {
						continue
					}
					fullText.WriteString(text)
					writeSSE("response.output_text.delta", map[string]any{
						"type":          "response.output_text.delta",
						"output_index":  0,
						"content_index": 0,
						"delta":         text,
					})
				}
			}

			accumulated := fullText.String()
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
					"content": []any{map[string]any{"type": "output_text", "text": accumulated}},
					"status":  "completed",
				},
			})
			writeSSE("response.completed", map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":         responseID,
					"object":     "response",
					"model":      model,
					"status":     "completed",
					"created_at": createdAt,
					"output": []any{map[string]any{
						"type":    "message",
						"id":      msgID,
						"role":    "assistant",
						"content": []any{map[string]any{"type": "output_text", "text": accumulated}},
						"status":  "completed",
					}},
				},
			})
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
		if choices, ok := openaiResp["choices"].([]any); ok && len(choices) > 0 {
			if cm, ok := choices[0].(map[string]any); ok {
				if msg, ok := cm["message"].(map[string]any); ok {
					outputText, _ = msg["content"].(string)
				}
			}
		}

		responseID := "resp_" + strings.ReplaceAll(fmt.Sprintf("%x", time.Now().UnixNano()), "-", "")
		msgID := fmt.Sprintf("msg_%x", time.Now().UnixNano())

		respBody := map[string]any{
			"id":         responseID,
			"object":     "response",
			"created_at": time.Now().Unix(),
			"model":      model,
			"status":     "completed",
			"output": []any{map[string]any{
				"type": "message",
				"id":   msgID,
				"role": "assistant",
				"content": []any{map[string]any{
					"type":        "output_text",
					"text":        outputText,
					"annotations": []any{},
				}},
				"status": "completed",
			}},
			"usage": openaiResp["usage"],
		}

		out, err := json.Marshal(respBody)
		if err != nil {
			writeOpenAIError(c, http.StatusInternalServerError, err)
			return
		}
		c.Data(status, "application/json; charset=utf-8", out)
	}
}
