package handler

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"llmux/internal/backend"
	"llmux/internal/config"

	"github.com/gin-gonic/gin"
)

// native Gemini API representation of a single model
type geminiModelResponse struct {
	Name                       string   `json:"name"`
	Version                    string   `json:"version"`
	DisplayName                string   `json:"displayName"`
	Description                string   `json:"description"`
	InputTokenLimit            int      `json:"inputTokenLimit"`
	OutputTokenLimit           int      `json:"outputTokenLimit"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
	Temperature                float64  `json:"temperature"`
	MaxTemperature             float64  `json:"maxTemperature"`
	TopP                       float64  `json:"topP"`
	TopK                       int      `json:"topK"`
}

func configModelToResponse(m config.ModelInfo) geminiModelResponse {
	return geminiModelResponse{
		Name:                       m.Name,
		Version:                    m.Version,
		DisplayName:                m.DisplayName,
		Description:                m.Description,
		InputTokenLimit:            m.InputTokenLimit,
		OutputTokenLimit:           m.OutputTokenLimit,
		SupportedGenerationMethods: m.SupportedGenerationMethods,
		Temperature:                m.Temperature,
		MaxTemperature:             m.MaxTemperature,
		TopP:                       m.TopP,
		TopK:                       m.TopK,
	}
}

// returns all available Gemini models in native Gemini API format
func GeminiListModels(gb *backend.GeminiBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		models := config.GeminiBaseModels
		resp := make([]geminiModelResponse, len(models))
		for i, m := range models {
			resp[i] = configModelToResponse(m)
		}
		c.JSON(http.StatusOK, gin.H{"models": resp})
	}
}

// returns a single Gemini model by name in native Gemini API format
func GeminiGetModel(gb *backend.GeminiBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		modelParam := c.Param("model")
		// strip leading slash from gin wildcard captures
		modelParam = strings.TrimPrefix(modelParam, "/")

		// normalise to bare model name (without "models/" prefix) for matching
		bare := strings.TrimPrefix(modelParam, "models/")

		models := config.GeminiBaseModels
		for _, m := range models {
			if m.Name == bare || m.Name == modelParam {
				c.JSON(http.StatusOK, configModelToResponse(m))
				return
			}
		}

		c.JSON(http.StatusNotFound, gin.H{
			"error": gin.H{
				"code":    http.StatusNotFound,
				"message": fmt.Sprintf("model not found: %s", modelParam),
				"status":  "NOT_FOUND",
			},
		})
	}
}

// proxies native Gemini generateContent / streamGenerateContent
func GeminiProxy(gb *backend.GeminiBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		// read the raw request body
		rawBody, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": gin.H{
					"code":    http.StatusBadRequest,
					"message": fmt.Sprintf("failed to read request body: %v", err),
					"status":  "INVALID_ARGUMENT",
				},
			})
			return
		}
		// default to empty JSON object if no body was supplied
		if len(rawBody) == 0 {
			rawBody = []byte("{}")
		}

		// validate that the body is well-formed JSON
		if !json.Valid(rawBody) {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": gin.H{
					"code":    http.StatusBadRequest,
					"message": "invalid JSON in request body",
					"status":  "INVALID_ARGUMENT",
				},
			})
			return
		}

		// extract and normalise model name (strip "models/" prefix if present)
		modelParam := c.Param("model")
		model := strings.TrimPrefix(modelParam, "models/")

		// determine action from the "action" param or fall back to path inspection
		action := c.Param("action")
		action = strings.TrimPrefix(action, "/")
		// some routers encode the action after a colon in the model param
		if action == "" && strings.Contains(model, ":") {
			parts := strings.SplitN(model, ":", 2)
			model = parts[0]
			action = parts[1]
		}

		isStreaming := strings.EqualFold(action, "streamGenerateContent")

		if isStreaming {
			statusCode, ch, err := gb.StreamRaw(c.Request.Context(), model, rawBody)
			if err != nil {
				httpStatus := statusCode
				if httpStatus == 0 {
					httpStatus = http.StatusInternalServerError
				}
				c.JSON(httpStatus, gin.H{
					"error": gin.H{
						"code":    httpStatus,
						"message": err.Error(),
						"status":  "INTERNAL",
					},
				})
				return
			}

			c.Header("Content-Type", "text/event-stream")
			c.Header("Cache-Control", "no-cache")
			c.Header("X-Accel-Buffering", "no")

			flusher, canFlush := c.Writer.(http.Flusher)

			for chunk := range ch {
				if chunk.Error != nil {
					// write the error as a final SSE event and stop
					fmt.Fprintf(c.Writer, "data: {\"error\":{\"message\":%q}}\n\n",
						chunk.Error.Error())
					if canFlush {
						flusher.Flush()
					}
					return
				}
				fmt.Fprintf(c.Writer, "data: %s\n\n", chunk.Data)
				if canFlush {
					flusher.Flush()
				}
			}
			return
		}

		// non-streaming path
		respBody, statusCode, err := gb.SendRaw(c.Request.Context(), model, rawBody)
		if err != nil {
			httpStatus := statusCode
			if httpStatus == 0 {
				httpStatus = http.StatusInternalServerError
			}
			// try to forward the upstream error body as-is when it is valid JSON
			if len(respBody) > 0 && json.Valid(respBody) {
				c.Data(httpStatus, "application/json; charset=utf-8", respBody)
				return
			}
			c.JSON(httpStatus, gin.H{
				"error": gin.H{
					"code":    httpStatus,
					"message": err.Error(),
					"status":  "INTERNAL",
				},
			})
			return
		}

		c.Data(http.StatusOK, "application/json; charset=utf-8", respBody)
	}
}
