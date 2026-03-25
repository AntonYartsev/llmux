package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"llmux/internal/backend"
	"llmux/internal/config"
	"llmux/internal/transform"
)

// dispatch OpenAI-format requests to the appropriate backend
type Router struct {
	backends       map[string]backend.Backend
	fallbackChains map[string][]string
	backendMap     map[string]string // model -> backend name override
}

// constructs a Router wired to the backends and configured from cfg
func NewRouter(gemini, claude backend.Backend, cfg config.AppConfig) *Router {
	r := &Router{
		backends:       make(map[string]backend.Backend),
		fallbackChains: config.ParseFallbackChains(cfg.FallbackChains),
		backendMap:     make(map[string]string),
	}

	if gemini != nil {
		r.backends["gemini"] = gemini
	}
	if claude != nil {
		r.backends["claude"] = claude
	}

	// parse MODEL_BACKEND_MAP: "model1:backend1,model2:backend2,..."
	if cfg.ModelBackendMap != "" {
		for _, entry := range strings.Split(cfg.ModelBackendMap, ",") {
			entry = strings.TrimSpace(entry)
			if entry == "" {
				continue
			}
			parts := strings.SplitN(entry, ":", 2)
			if len(parts) == 2 {
				model := strings.TrimSpace(parts[0])
				bname := strings.TrimSpace(parts[1])
				if model != "" && bname != "" {
					r.backendMap[model] = bname
				}
			}
		}
	}

	return r
}

// returns the best available backend for the given model
func (r *Router) resolveBackend(model string) backend.Backend {
	name := config.ResolveBackendName(model, r.backendMap)
	b := r.backends[name]
	if b != nil && b.IsAvailable() {
		return b
	}
	// primary unavailable — try the other backend
	for k, candidate := range r.backends {
		if k != name && candidate.IsAvailable() {
			return candidate
		}
	}

	return b
}

// reports whether an error or HTTP status warrants trying the next entry in the fallback chain
func shouldFallback(statusCode int, err error) bool {
	if err != nil {
		return true // network / transport error
	}
	if statusCode >= 500 {
		return true
	}
	if statusCode == 429 {
		return true
	}
	return false
}


// converts an OpenAI ChatRequest to the payload map expected by the named backend
func transformPayload(b backend.Backend, openaiReq *transform.ChatRequest, model string) (map[string]any, error) {
	if b == nil {
		return nil, fmt.Errorf("nil backend")
	}
	switch b.Name() {
	case "gemini":
		return transform.OpenAIRequestToGemini(openaiReq), nil
	case "claude":
		return transform.OpenAIRequestToClaude(openaiReq), nil
	default:
		return nil, fmt.Errorf("unknown backend %q", b.Name())
	}
}

// converts a raw backend response body to an OpenAI response map using the appropriate converter for the named backend
func transformResponse(bname string, body []byte, model string) (map[string]any, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	switch bname {
	case "gemini":
		return transform.GeminiResponseToOpenAI(raw, model), nil
	case "claude":
		return transform.ClaudeResponseToOpenAI(raw, model), nil
	default:
		return nil, fmt.Errorf("unknown backend %q", bname)
	}
}

// executes a non-streaming request, optionally walking the fallback chain, and returns (openaiResponseBytes, httpStatus, actualModel, error)
func (r *Router) Send(
	ctx context.Context,
	model string,
	openaiReq *transform.ChatRequest,
) ([]byte, int, string, error) {
	model = strings.TrimPrefix(model, "models/")
	b := r.resolveBackend(model)
	if b == nil {
		return nil, 503, model, fmt.Errorf("no backend available for model %q", model)
	}

	payload, err := transformPayload(b, openaiReq, model)
	if err != nil {
		return nil, 500, model, err
	}

	body, status, err := b.Send(ctx, model, payload)

	if shouldFallback(status, err) {
		if chain, ok := r.fallbackChains[model]; ok {
			for _, fbModel := range chain {
				fbBackend := r.resolveBackend(fbModel)
				if fbBackend == nil {
					continue
				}
				fbPayload, ferr := transformPayload(fbBackend, openaiReq, fbModel)
				if ferr != nil {
					continue
				}
				fbBody, fbStatus, fbErr := fbBackend.Send(ctx, fbModel, fbPayload)
				if !shouldFallback(fbStatus, fbErr) && fbErr == nil {
					result, terr := transformResponse(fbBackend.Name(), fbBody, fbModel)
					if terr != nil {
						return nil, 500, fbModel, terr
					}
					out, merr := json.Marshal(result)
					if merr != nil {
						return nil, 500, fbModel, merr
					}
					return out, fbStatus, fbModel, nil
				}
			}
		}
	}

	if err != nil {
		return nil, status, model, err
	}

	result, terr := transformResponse(b.Name(), body, model)
	if terr != nil {
		return nil, 500, model, terr
	}
	out, merr := json.Marshal(result)
	if merr != nil {
		return nil, 500, model, merr
	}
	return out, status, model, nil
}

// executes a streaming request, returning an SSE byte channel
func (r *Router) Stream(
	ctx context.Context,
	model string,
	openaiReq *transform.ChatRequest,
) (int, <-chan []byte, string, error) {
	model = strings.TrimPrefix(model, "models/")
	b := r.resolveBackend(model)
	if b == nil {
		return 503, nil, model, fmt.Errorf("no backend available for model %q", model)
	}

	payload, err := transformPayload(b, openaiReq, model)
	if err != nil {
		return 500, nil, model, err
	}

	status, chunkCh, err := b.Stream(ctx, model, payload)

	// on initial error, try fallback chain
	if shouldFallback(status, err) {
		if chain, ok := r.fallbackChains[model]; ok {
			for _, fbModel := range chain {
				fbBackend := r.resolveBackend(fbModel)
				if fbBackend == nil {
					continue
				}
				fbPayload, ferr := transformPayload(fbBackend, openaiReq, fbModel)
				if ferr != nil {
					continue
				}
				fbStatus, fbCh, fbErr := fbBackend.Stream(ctx, fbModel, fbPayload)
				if !shouldFallback(fbStatus, fbErr) && fbErr == nil {
					outCh := make(chan []byte, 32)
					go streamWorker(ctx, fbBackend.Name(), fbModel, fbCh, outCh)
					return fbStatus, outCh, fbModel, nil
				}
			}
		}
	}

	if err != nil {
		return status, nil, model, err
	}

	outCh := make(chan []byte, 32)
	go streamWorker(ctx, b.Name(), model, chunkCh, outCh)
	return status, outCh, model, nil
}

// reads backend StreamChunk events, converts each to an OpenAI SSE line, and sends it on outCh. It closes outCh when done
func streamWorker(
	ctx context.Context,
	bname string,
	model string,
	chunkCh <-chan backend.StreamChunk,
	outCh chan<- []byte,
) {
	defer close(outCh)

	responseID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	ts := time.Now().Unix()
	var claudeState transform.ClaudeStreamState

	send := func(data []byte) {
		line := make([]byte, 0, len(data)+8)
		line = append(line, []byte("data: ")...)
		line = append(line, data...)
		line = append(line, '\n', '\n')
		select {
		case outCh <- line:
		case <-ctx.Done():
		}
	}

	// for Gemini backend, emit the initial role announcement chunk
	// (OpenAI streaming always starts with a chunk containing role: "assistant")
	if bname == "gemini" {
		roleChunk := map[string]any{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": ts,
			"model":   model,
			"choices": []any{
				map[string]any{
					"index":         0,
					"delta":         map[string]any{"role": "assistant", "content": ""},
					"logprobs":      nil,
					"finish_reason": nil,
				},
			},
			"system_fingerprint": nil,
		}
		if b, err := json.Marshal(roleChunk); err == nil {
			send(b)
		}
	}

	for chunk := range chunkCh {
		if chunk.Error != nil {
			// send an error SSE event and stop
			errObj := map[string]any{
				"error": map[string]any{
					"message": chunk.Error.Error(),
					"type":    "stream_error",
				},
			}
			if b, err := json.Marshal(errObj); err == nil {
				send(b)
			}
			return
		}

		switch bname {
		case "gemini":
			var raw map[string]any
			if err := json.Unmarshal(chunk.Data, &raw); err != nil {
				continue
			}
			result := transform.GeminiStreamChunkToOpenAI(raw, model, responseID)
			if result == nil {
				continue
			}
			if b, err := json.Marshal(result); err == nil {
				send(b)
			}

		case "claude":
			// each Claude chunk carries a JSON object whose "type" field acts as
			// the SSE event type (Claude sends "event: <type>" + "data: <json>"
			// pairs; ClaudeBackend delivers only the data portion)
			var raw map[string]any
			if err := json.Unmarshal(chunk.Data, &raw); err != nil {
				continue
			}
			eventType, _ := raw["type"].(string)
			results := transform.ClaudeStreamEventToOpenAI(raw, eventType, model, responseID, &claudeState)
			for _, result := range results {
				if b, err := json.Marshal(result); err == nil {
					send(b)
				}
			}
		}
	}

	// send the SSE stream terminator
	select {
	case outCh <- []byte("data: [DONE]\n\n"):
	case <-ctx.Done():
	}
}

// returns the combined model list from all available backends
func (r *Router) AllModels() []backend.ModelInfo {
	var all []backend.ModelInfo
	for _, b := range r.backends {
		if b.IsAvailable() {
			all = append(all, b.ListModels()...)
		}
	}
	return all
}
