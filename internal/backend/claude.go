package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"llmux/internal/auth"
	"llmux/internal/config"
	"llmux/internal/transform"
)

// backend for the Anthropic Messages API
type ClaudeBackend struct {
	store *auth.CredentialStore
}

// creates a ClaudeBackend backed by given credential store
func NewClaudeBackend(store *auth.CredentialStore) *ClaudeBackend {
	return &ClaudeBackend{store: store}
}

// returns the provider identifier
func (c *ClaudeBackend) Name() string { return "claude" }

// reports whether Claude credentials are present in the store
func (c *ClaudeBackend) IsAvailable() bool { return c.store.GetClaude() != nil }

// returns the static list of supported Claude models
func (c *ClaudeBackend) ListModels() []ModelInfo {
	result := make([]ModelInfo, len(config.ClaudeModels))
	for i, m := range config.ClaudeModels {
		result[i] = ModelInfo{
			ID:          m.Name,
			Provider:    "claude",
			DisplayName: m.DisplayName,
			Description: m.Description,
		}
	}
	return result
}

// send structured payload to the Claude Messages API and returns the full response body, HTTP status code, and any error
func (c *ClaudeBackend) Send(ctx context.Context, model string, payload map[string]any) ([]byte, int, error) {
	token, err := auth.EnsureClaudeToken(c.store)
	if err != nil {
		return nil, 0, err
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal payload: %w", err)
	}

	return c.sendRawWithToken(ctx, token, raw)
}

// send pre-serialised bytes to the Claude Messages API, injecting the auth headers without unmarshalling or re-marshalling the body
func (c *ClaudeBackend) SendRaw(ctx context.Context, model string, body []byte) ([]byte, int, error) {
	token, err := auth.EnsureClaudeToken(c.store)
	if err != nil {
		return nil, 0, err
	}

	return c.sendRawWithToken(ctx, token, body)
}

// shared implementation for Send and SendRaw
func (c *ClaudeBackend) sendRawWithToken(ctx context.Context, token string, body []byte) ([]byte, int, error) {
	headers := transform.ClaudeHeaders(token)
	slog.Debug("claude send", "url", transform.ClaudeAPIURL, "body", string(body))

	resp, err := doWithRetry(ctx, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, transform.ClaudeAPIURL, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		return req, nil
	})
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode >= 400 {
		return respBody, resp.StatusCode, &BackendError{
			StatusCode: resp.StatusCode,
			Message:    string(respBody),
		}
	}

	return respBody, resp.StatusCode, nil
}

// send a structured payload with streaming enabled and returns a channel of SSE chunks
// the HTTP status code is also returned so callers can detect early errors before reading channel
func (c *ClaudeBackend) Stream(ctx context.Context, model string, payload map[string]any) (int, <-chan StreamChunk, error) {
	token, err := auth.EnsureClaudeToken(c.store)
	if err != nil {
		return 0, nil, err
	}

	// copy payload and force streaming
	p := make(map[string]any, len(payload)+1)
	for k, v := range payload {
		p[k] = v
	}
	p["stream"] = true

	raw, err := json.Marshal(p)
	if err != nil {
		return 0, nil, fmt.Errorf("marshal payload: %w", err)
	}

	return c.streamRawWithToken(ctx, token, raw)
}

// send pre-serialised bytes with streaming, injecting auth headers without unmarshalling or re-marshalling the body
func (c *ClaudeBackend) StreamRaw(ctx context.Context, model string, body []byte) (int, <-chan StreamChunk, error) {
	token, err := auth.EnsureClaudeToken(c.store)
	if err != nil {
		return 0, nil, err
	}

	return c.streamRawWithToken(ctx, token, body)
}

// shared implementation for Stream and StreamRaw
func (c *ClaudeBackend) streamRawWithToken(ctx context.Context, token string, body []byte) (int, <-chan StreamChunk, error) {
	headers := transform.ClaudeHeaders(token)
	slog.Debug("claude stream", "url", transform.ClaudeAPIURL, "body", string(body))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, transform.ClaudeAPIURL, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, nil, err
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, nil, &BackendError{
			StatusCode: resp.StatusCode,
			Message:    string(respBody),
		}
	}

	ch := make(chan StreamChunk, 16)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := line[6:] // strip "data: " prefix

			if data == "[DONE]" {
				return
			}

			ch <- StreamChunk{Data: []byte(data)}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamChunk{Error: err}
		}
	}()

	return resp.StatusCode, ch, nil
}

// executes makeReq, retrying on network errors and HTTP 429
// responses with delays of 3 s and 8 s between attempts (3 attempts total!)
func doWithRetry(ctx context.Context, makeReq func() (*http.Request, error)) (*http.Response, error) {
	delays := []time.Duration{3 * time.Second, 8 * time.Second}
	var lastErr error

	for attempt := 0; attempt <= len(delays); attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delays[attempt-1]):
			}
		}

		req, err := makeReq()
		if err != nil {
			return nil, err
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			resp.Body.Close()
			lastErr = fmt.Errorf("rate limited (429)")
			continue
		}

		return resp, nil
	}

	return nil, lastErr
}
