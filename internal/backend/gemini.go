package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"llmux/internal/auth"
	"llmux/internal/config"
	"llmux/internal/transform"
)

// backend interface using Google's Code Assist API
type GeminiBackend struct {
	store          *auth.CredentialStore
	onboardingDone atomic.Bool
	projectID      string
	projectIDMu    sync.Mutex
}

// create new GeminiBackend backed by the given credential store
func NewGeminiBackend(store *auth.CredentialStore) *GeminiBackend {
	return &GeminiBackend{store: store}
}

// returns the backend identifier
func (g *GeminiBackend) Name() string { return "gemini" }

// reports whether Gemini credentials are present in the store
func (g *GeminiBackend) IsAvailable() bool { return g.store.GetGemini() != nil }

// returns the full set of Gemini models (base + all variants)
func (g *GeminiBackend) ListModels() []ModelInfo {
	geminiModels := config.GenerateAllGeminiModels()
	result := make([]ModelInfo, len(geminiModels))
	for i, m := range geminiModels {
		result[i] = ModelInfo{
			ID:          m.Name,
			Provider:    "gemini",
			DisplayName: m.DisplayName,
			Description: m.Description,
		}
	}
	return result
}

// resolve the Google Cloud project ID using the following
// priority order:
//  1. GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT env var (via config.Cfg)
//  2. in-memory cache on this backend instance
//  3. ProjectID stored in the credential file
//  4. loadCodeAssist API call (discovers and then caches the result)
func (g *GeminiBackend) DiscoverProjectID() (string, error) {
	// priority 1: env var
	if config.Cfg.GoogleCloudProject != "" {
		return config.Cfg.GoogleCloudProject, nil
	}

	// priority 2: in-memory cache
	g.projectIDMu.Lock()
	cached := g.projectID
	g.projectIDMu.Unlock()
	if cached != "" {
		return cached, nil
	}

	// priority 3: credential file
	if creds := g.store.GetGemini(); creds != nil && creds.ProjectID != "" {
		g.projectIDMu.Lock()
		g.projectID = creds.ProjectID
		g.projectIDMu.Unlock()
		return creds.ProjectID, nil
	}

	// priority 4: API discovery
	token, err := auth.EnsureGeminiToken(g.store)
	if err != nil {
		return "", fmt.Errorf("discoverProjectID: %w", err)
	}

	probePayload := map[string]any{
		"metadata": config.GetClientMetadata(""),
	}
	probeBody, err := json.Marshal(probePayload)
	if err != nil {
		return "", fmt.Errorf("discoverProjectID: marshal probe: %w", err)
	}

	url := config.CodeAssistEndpoint + "/v1internal:loadCodeAssist"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(probeBody))
	if err != nil {
		return "", fmt.Errorf("discoverProjectID: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", config.GetUserAgent())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("discoverProjectID: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("discoverProjectID: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("discoverProjectID: loadCodeAssist returned HTTP %d: %s", resp.StatusCode, respBody)
	}

	var data map[string]any
	if err := json.Unmarshal(respBody, &data); err != nil {
		return "", fmt.Errorf("discoverProjectID: parse response: %w", err)
	}

	projectID, _ := data["cloudaicompanionProject"].(string)
	if projectID == "" {
		return "", fmt.Errorf("discoverProjectID: 'cloudaicompanionProject' not found in loadCodeAssist response")
	}

	// cache in memory and persist to the credential store
	g.projectIDMu.Lock()
	g.projectID = projectID
	g.projectIDMu.Unlock()

	if creds := g.store.GetGemini(); creds != nil {
		creds.ProjectID = projectID
		if err := g.store.UpdateGemini(creds); err != nil {
			log.Printf("gemini: failed to persist projectID: %v", err)
		}
	}

	return projectID, nil
}

// builds an authenticated HTTP request and executes it
// it adds Authorization, Content-Type, User-Agent, and (when known) x-goog-request-params headers
func (g *GeminiBackend) doRequest(ctx context.Context, method, urlStr string, body []byte) (*http.Response, error) {
	token, err := auth.EnsureGeminiToken(g.store)
	if err != nil {
		return nil, fmt.Errorf("doRequest: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, urlStr, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("doRequest: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", config.GetUserAgent())

	// add project param if we have one cached already
	g.projectIDMu.Lock()
	pid := g.projectID
	g.projectIDMu.Unlock()
	if pid == "" {
		pid = config.Cfg.GoogleCloudProject
	}
	if pid != "" {
		req.Header.Set("x-goog-request-params", "project="+pid)
	}

	return http.DefaultClient.Do(req)
}

// wait durations between retries for 429 / network errors
var retryDelays = []time.Duration{3 * time.Second, 8 * time.Second}

// send a non-streaming request to Gemini Code Assist API
func (g *GeminiBackend) Send(ctx context.Context, model string, payload map[string]any) ([]byte, int, error) {
	token, err := auth.EnsureGeminiToken(g.store)
	if err != nil {
		return nil, 0, fmt.Errorf("gemini Send: %w", err)
	}
	_ = token // token is used inside doRequest; call EnsureGeminiToken first to fail early

	projectID, err := g.DiscoverProjectID()
	if err != nil {
		return nil, 0, fmt.Errorf("gemini Send: %w", err)
	}

	// run onboarding if not yet done
	if !g.onboardingDone.Load() {
		if err := g.RunOnboarding(); err != nil {
			log.Printf("gemini: onboarding error (continuing): %v", err)
		}
	}

	wrapped := transform.BuildGeminiPayload(payload, model, projectID)
	reqBody, err := json.Marshal(wrapped)
	if err != nil {
		return nil, 0, fmt.Errorf("gemini Send: marshal payload: %w", err)
	}

	url := config.CodeAssistEndpoint + "/v1internal:generateContent"
	slog.Debug("gemini send", "url", url, "body", string(reqBody))

	var (
		resp     *http.Response
		respBody []byte
	)
	for attempt := 0; attempt <= len(retryDelays); attempt++ {
		resp, err = g.doRequest(ctx, http.MethodPost, url, reqBody)
		if err != nil {
			// network error: retry if attempts remain
			if attempt < len(retryDelays) {
				log.Printf("gemini Send: request failed (attempt %d), retrying in %s: %v",
					attempt+1, retryDelays[attempt], err)
				select {
				case <-ctx.Done():
					return nil, 0, ctx.Err()
				case <-time.After(retryDelays[attempt]):
				}
				continue
			}
			return nil, 0, fmt.Errorf("gemini Send: request failed: %w", err)
		}

		respBody, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, 0, fmt.Errorf("gemini Send: read response: %w", err)
		}

		// retry on 429
		if resp.StatusCode == http.StatusTooManyRequests && attempt < len(retryDelays) {
			log.Printf("gemini Send: rate limited (429), retrying in %s (attempt %d)",
				retryDelays[attempt], attempt+1)
			select {
			case <-ctx.Done():
				return nil, 0, ctx.Err()
			case <-time.After(retryDelays[attempt]):
			}
			continue
		}

		break
	}

	if resp.StatusCode >= 400 {
		return respBody, resp.StatusCode, &BackendError{StatusCode: resp.StatusCode, Message: string(respBody)}
	}

	// unwrap Code Assist envelope: {"response": <actual Gemini response>}
	inner, err := unwrapEnvelope(respBody)
	if err != nil {
		// if we can't unwrap, return raw body as-is (best effort)
		log.Printf("gemini Send: envelope unwrap failed: %v", err)
		return respBody, resp.StatusCode, nil
	}
	return inner, http.StatusOK, nil
}

// send a streaming request to the Gemini Code Assist API (SSE)
func (g *GeminiBackend) Stream(ctx context.Context, model string, payload map[string]any) (int, <-chan StreamChunk, error) {
	token, err := auth.EnsureGeminiToken(g.store)
	if err != nil {
		return 0, nil, fmt.Errorf("gemini Stream: %w", err)
	}
	_ = token

	projectID, err := g.DiscoverProjectID()
	if err != nil {
		return 0, nil, fmt.Errorf("gemini Stream: %w", err)
	}

	// run onboarding if not yet done
	if !g.onboardingDone.Load() {
		if err := g.RunOnboarding(); err != nil {
			log.Printf("gemini: onboarding error (continuing): %v", err)
		}
	}

	wrapped := transform.BuildGeminiPayload(payload, model, projectID)
	reqBody, err := json.Marshal(wrapped)
	if err != nil {
		return 0, nil, fmt.Errorf("gemini Stream: marshal payload: %w", err)
	}

	url := config.CodeAssistEndpoint + "/v1internal:streamGenerateContent?alt=sse"

	resp, err := g.doRequest(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return 0, nil, fmt.Errorf("gemini Stream: request failed: %w", err)
	}

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return resp.StatusCode, nil, &BackendError{StatusCode: resp.StatusCode, Message: string(body)}
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
			data := line[len("data: "):]

			inner, err := unwrapEnvelope([]byte(data))
			if err != nil {
				// if unwrapping fails, forward the raw JSON data
				log.Printf("gemini Stream: envelope unwrap failed: %v", err)
				inner = []byte(data)
			}

			select {
			case ch <- StreamChunk{Data: inner}:
			case <-ctx.Done():
				return
			}
		}

		if err := scanner.Err(); err != nil {
			select {
			case ch <- StreamChunk{Error: err}:
			case <-ctx.Done():
			}
		}
	}()

	return http.StatusOK, ch, nil
}

// unmarshal pre-serialized bytes, wraps them in the Code Assist envelope via BuildGeminiPayloadFromNative, and delegates to the HTTP layer
func (g *GeminiBackend) SendRaw(ctx context.Context, model string, body []byte) ([]byte, int, error) {
	var nativeReq map[string]any
	if err := json.Unmarshal(body, &nativeReq); err != nil {
		return nil, http.StatusBadRequest, fmt.Errorf("gemini SendRaw: unmarshal body: %w", err)
	}

	projectID, err := g.DiscoverProjectID()
	if err != nil {
		return nil, 0, fmt.Errorf("gemini SendRaw: %w", err)
	}

	// run onboarding if not yet done
	if !g.onboardingDone.Load() {
		if err := g.RunOnboarding(); err != nil {
			log.Printf("gemini: onboarding error (continuing): %v", err)
		}
	}

	wrapped := transform.BuildGeminiPayloadFromNative(nativeReq, model)
	// BuildGeminiPayloadFromNative uses config.Cfg.GoogleCloudProject for the project field
	// override with the discovered project when config has none
	if config.Cfg.GoogleCloudProject == "" {
		wrapped["project"] = projectID
	}

	reqBody, err := json.Marshal(wrapped)
	if err != nil {
		return nil, 0, fmt.Errorf("gemini SendRaw: marshal: %w", err)
	}

	url := config.CodeAssistEndpoint + "/v1internal:generateContent"

	var (
		resp     *http.Response
		respBody []byte
	)
	for attempt := 0; attempt <= len(retryDelays); attempt++ {
		resp, err = g.doRequest(ctx, http.MethodPost, url, reqBody)
		if err != nil {
			if attempt < len(retryDelays) {
				log.Printf("gemini SendRaw: request failed (attempt %d), retrying in %s: %v",
					attempt+1, retryDelays[attempt], err)
				select {
				case <-ctx.Done():
					return nil, 0, ctx.Err()
				case <-time.After(retryDelays[attempt]):
				}
				continue
			}
			return nil, 0, fmt.Errorf("gemini SendRaw: request failed: %w", err)
		}

		respBody, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, 0, fmt.Errorf("gemini SendRaw: read response: %w", err)
		}

		if resp.StatusCode == http.StatusTooManyRequests && attempt < len(retryDelays) {
			log.Printf("gemini SendRaw: rate limited (429), retrying in %s (attempt %d)",
				retryDelays[attempt], attempt+1)
			select {
			case <-ctx.Done():
				return nil, 0, ctx.Err()
			case <-time.After(retryDelays[attempt]):
			}
			continue
		}

		break
	}

	if resp.StatusCode >= 400 {
		return respBody, resp.StatusCode, &BackendError{StatusCode: resp.StatusCode, Message: string(respBody)}
	}

	inner, err := unwrapEnvelope(respBody)
	if err != nil {
		log.Printf("gemini SendRaw: envelope unwrap failed: %v", err)
		return respBody, resp.StatusCode, nil
	}
	return inner, http.StatusOK, nil
}

// streaming equivalent of SendRaw
func (g *GeminiBackend) StreamRaw(ctx context.Context, model string, body []byte) (int, <-chan StreamChunk, error) {
	var nativeReq map[string]any
	if err := json.Unmarshal(body, &nativeReq); err != nil {
		return http.StatusBadRequest, nil, fmt.Errorf("gemini StreamRaw: unmarshal body: %w", err)
	}

	projectID, err := g.DiscoverProjectID()
	if err != nil {
		return 0, nil, fmt.Errorf("gemini StreamRaw: %w", err)
	}

	// run onboarding if not yet done
	if !g.onboardingDone.Load() {
		if err := g.RunOnboarding(); err != nil {
			log.Printf("gemini: onboarding error (continuing): %v", err)
		}
	}

	wrapped := transform.BuildGeminiPayloadFromNative(nativeReq, model)
	if config.Cfg.GoogleCloudProject == "" {
		wrapped["project"] = projectID
	}

	reqBody, err := json.Marshal(wrapped)
	if err != nil {
		return 0, nil, fmt.Errorf("gemini StreamRaw: marshal: %w", err)
	}

	url := config.CodeAssistEndpoint + "/v1internal:streamGenerateContent?alt=sse"

	resp, err := g.doRequest(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return 0, nil, fmt.Errorf("gemini StreamRaw: request failed: %w", err)
	}

	if resp.StatusCode >= 400 {
		rbody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return resp.StatusCode, nil, &BackendError{StatusCode: resp.StatusCode, Message: string(rbody)}
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
			data := line[len("data: "):]

			inner, err := unwrapEnvelope([]byte(data))
			if err != nil {
				log.Printf("gemini StreamRaw: envelope unwrap failed: %v", err)
				inner = []byte(data)
			}

			select {
			case ch <- StreamChunk{Data: inner}:
			case <-ctx.Done():
				return
			}
		}

		if err := scanner.Err(); err != nil {
			select {
			case ch <- StreamChunk{Error: err}:
			case <-ctx.Done():
			}
		}
	}()

	return http.StatusOK, ch, nil
}

// ensures the user is onboarded via the Code Assist API
// It is idempotent: once onboarding is confirmed the method returns immediately on subsequent calls
func (g *GeminiBackend) RunOnboarding() error {
	if g.onboardingDone.Load() {
		return nil
	}

	token, err := auth.EnsureGeminiToken(g.store)
	if err != nil {
		return fmt.Errorf("onboarding: %w", err)
	}

	projectID, err := g.DiscoverProjectID()
	if err != nil {
		return fmt.Errorf("onboarding: %w", err)
	}

	headers := map[string]string{
		"Authorization": "Bearer " + token,
		"Content-Type":  "application/json",
		"User-Agent":    config.GetUserAgent(),
	}

	doPost := func(url string, payload any) (map[string]any, error) {
		body, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal: %w", err)
		}
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request: %w", err)
		}
		defer resp.Body.Close()
		rb, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("read response: %w", err)
		}
		if resp.StatusCode >= 400 {
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, rb)
		}
		var result map[string]any
		if err := json.Unmarshal(rb, &result); err != nil {
			return nil, fmt.Errorf("parse response: %w", err)
		}
		return result, nil
	}

	doGet := func(url string) (map[string]any, error) {
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request: %w", err)
		}
		defer resp.Body.Close()
		rb, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("read response: %w", err)
		}
		if resp.StatusCode >= 400 {
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, rb)
		}
		var result map[string]any
		if err := json.Unmarshal(rb, &result); err != nil {
			return nil, fmt.Errorf("parse response: %w", err)
		}
		return result, nil
	}

	// step 1: loadCodeAssist to determine current tier
	loadPayload := map[string]any{
		"cloudaicompanionProject": projectID,
		"metadata":                config.GetClientMetadata(projectID),
	}
	loadURL := config.CodeAssistEndpoint + "/v1internal:loadCodeAssist"
	loadData, err := doPost(loadURL, loadPayload)
	if err != nil {
		return fmt.Errorf("onboarding loadCodeAssist: %w", err)
	}

	// if currentTier is already set, the user is already onboarded
	if currentTier, ok := loadData["currentTier"]; ok && currentTier != nil {
		g.onboardingDone.Store(true)
		return nil
	}

	// select the default tier or fall back to the legacy-tier sentinel
	var tier map[string]any
	allowedTiers, _ := loadData["allowedTiers"].([]any)
	for _, t := range allowedTiers {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		if isDefault, _ := tm["isDefault"].(bool); isDefault {
			tier = tm
			break
		}
	}
	if tier == nil {
		tier = map[string]any{
			"name":                               "",
			"description":                        "",
			"id":                                 "legacy-tier",
			"userDefinedCloudaicompanionProject": true,
		}
	}

	if udp, _ := tier["userDefinedCloudaicompanionProject"].(bool); udp && projectID == "" {
		return fmt.Errorf("onboarding: this account requires GOOGLE_CLOUD_PROJECT to be set")
	}

	// step 2: onboardUser to initiate the LRO
	onboardPayload := map[string]any{
		"tierId":                  tier["id"],
		"cloudaicompanionProject": projectID,
		"metadata":                config.GetClientMetadata(projectID),
	}
	onboardURL := config.CodeAssistEndpoint + "/v1internal:onboardUser"

	for {
		lroData, err := doPost(onboardURL, onboardPayload)
		if err != nil {
			return fmt.Errorf("onboarding onboardUser: %w", err)
		}

		if done, _ := lroData["done"].(bool); done {
			g.onboardingDone.Store(true)
			return nil
		}

		// if the LRO returned an operation name, poll it; otherwise re-post
		if opName, _ := lroData["name"].(string); opName != "" {
			// poll the operation until done
			opURL := config.CodeAssistEndpoint + "/" + opName
			for {
				time.Sleep(5 * time.Second)
				opData, err := doGet(opURL)
				if err != nil {
					return fmt.Errorf("onboarding poll operation: %w", err)
				}
				if done, _ := opData["done"].(bool); done {
					g.onboardingDone.Store(true)
					return nil
				}
			}
		}

		time.Sleep(5 * time.Second)
	}
}

// extracts the inner response from the Code Assist JSON envelope
func unwrapEnvelope(data []byte) ([]byte, error) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("parse envelope: %w", err)
	}
	inner, ok := envelope["response"]
	if !ok {
		// not wrapped — return raw.
		return data, nil
	}
	return inner, nil
}
