package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"llmux/internal/backend"
	"llmux/internal/config"

	"github.com/gin-gonic/gin"
)

// proxies requests to the Anthropic Messages API
func ClaudeMessages(cb *backend.ClaudeBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("read request body: %s", err)})
			return
		}

		// detect streaming by parsing the "stream" field from the JSON body
		streaming := false
		var req map[string]any
		if jsonErr := json.Unmarshal(body, &req); jsonErr == nil {
			if v, ok := req["stream"]; ok {
				if b, ok := v.(bool); ok && b {
					streaming = true
				}
			}
		} else {
			// fallback to a fast byte-level check if parsing fails
			streaming = bytes.Contains(body, []byte(`"stream":true`)) ||
				bytes.Contains(body, []byte(`"stream": true`))
		}

		if !streaming {
			respBody, status, err := cb.SendRaw(c.Request.Context(), "", body)
			if err != nil {
				if be, ok := err.(*backend.BackendError); ok {
					c.Data(be.StatusCode, "application/json", respBody)
					return
				}
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			c.Data(status, "application/json", respBody)
			return
		}

		// streaming path
		status, ch, err := cb.StreamRaw(c.Request.Context(), "", body)
		if err != nil {
			if be, ok := err.(*backend.BackendError); ok {
				c.JSON(be.StatusCode, gin.H{"error": be.Error()})
				return
			}
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		_ = status

		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Transfer-Encoding", "chunked")
		c.Status(http.StatusOK)

		flusher, canFlush := c.Writer.(http.Flusher)

		for chunk := range ch {
			if chunk.Error != nil {
				break
			}
			fmt.Fprintf(c.Writer, "data: %s\n\n", chunk.Data)
			if canFlush {
				flusher.Flush()
			}
		}
	}
}

// Anthropic-format model object returned by ClaudeListModels
type claudeModelEntry struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	Type        string `json:"type"`
	CreatedAt   string `json:"created_at"`
}

// returns the list of supported Claude models in Anthropic API format
func ClaudeListModels(cb *backend.ClaudeBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		models := make([]claudeModelEntry, 0, len(config.ClaudeModels))
		for _, m := range config.ClaudeModels {
			models = append(models, claudeModelEntry{
				ID:          m.Name,
				DisplayName: m.DisplayName,
				Type:        "model",
				CreatedAt:   "2025-01-01T00:00:00Z",
			})
		}
		c.JSON(http.StatusOK, gin.H{"models": models})
	}
}
