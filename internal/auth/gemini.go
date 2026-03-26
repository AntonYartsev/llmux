package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"llmux/internal/config"
)

const (
	geminiAuthBaseURL         = "https://accounts.google.com/o/oauth2/v2/auth"
	geminiTokenURL            = "https://oauth2.googleapis.com/token"
	geminiDefaultCallbackHost = "localhost"
	geminiCallbackPort        = "8085"
)

// Gemini OAuth2 authorization Code flow
// callbackHost and callbackPort override the defaults used in the redirect URI
func RunGeminiOAuthFlow(store *CredentialStore, callbackHost, callbackPort string) error {
	if callbackHost == "" {
		callbackHost = geminiDefaultCallbackHost
	}
	if callbackPort == "" {
		callbackPort = geminiCallbackPort
	}
	redirectURI := "http://" + callbackHost + ":" + callbackPort + "/callback"

	// build authorization URL
	params := url.Values{}
	params.Set("client_id", config.GeminiClientID)
	params.Set("redirect_uri", redirectURI)
	params.Set("response_type", "code")
	params.Set("scope", strings.Join(config.GeminiScopes, " "))
	params.Set("access_type", "offline")
	params.Set("prompt", "consent")

	authURL := geminiAuthBaseURL + "?" + params.Encode()
	fmt.Println("Open this URL in your browser:\n" + authURL)

	// start local HTTP server to capture the authorization code
	codeCh := make(chan string, 1)
	errCh := make(chan error, 1)

	mux := http.NewServeMux()
	srv := &http.Server{
		Addr:    ":" + callbackPort,
		Handler: mux,
	}

	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		code := r.URL.Query().Get("code")
		if code == "" {
			http.Error(w, "<h1>Authentication failed.</h1><p>Please try again.</p>", http.StatusBadRequest)
			errCh <- fmt.Errorf("no authorization code received in callback")
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, successHTML)
		codeCh <- code
	})

	// run server in goroutine (and shut down once have the code)
	srvErrCh := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			srvErrCh <- err
		}
	}()

	var authCode string
	select {
	case authCode = <-codeCh:
	case err := <-errCh:
		_ = srv.Shutdown(context.Background())
		return fmt.Errorf("oauth callback error: %w", err)
	case err := <-srvErrCh:
		return fmt.Errorf("oauth callback server error: %w", err)
	}

	// shut down callback server (ignore error — the request is done)
	_ = srv.Shutdown(context.Background())

	// exchange the authorization code for tokens
	creds, err := exchangeCodeForTokens(authCode, redirectURI)
	if err != nil {
		return fmt.Errorf("token exchange failed: %w", err)
	}

	return store.UpdateGemini(creds)
}

// obtain a new access token by stored refresh token (and persists updated credentials via store)
func RefreshGeminiToken(store *CredentialStore) error {
	creds := store.GetGemini()
	if creds == nil {
		return fmt.Errorf("gemini credentials not found")
	}

	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("client_id", config.GeminiClientID)
	form.Set("client_secret", config.GeminiClientSecret)
	form.Set("refresh_token", creds.RefreshToken)

	resp, err := http.PostForm(geminiTokenURL, form)
	if err != nil {
		return fmt.Errorf("refresh token request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading refresh response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("refresh token returned HTTP %d: %s", resp.StatusCode, body)
	}

	var tr tokenResponse
	if err := json.Unmarshal(body, &tr); err != nil {
		return fmt.Errorf("parsing refresh response: %w", err)
	}
	if tr.AccessToken == "" {
		return fmt.Errorf("no access_token in refresh response")
	}

	creds.Token = tr.AccessToken
	creds.Expiry = time.Now().Add(time.Duration(tr.ExpiresIn) * time.Second)

	return store.UpdateGemini(creds)
}

// returns a valid Gemini access token (and refreshing it proactively)
func EnsureGeminiToken(store *CredentialStore) (string, error) {
	creds := store.GetGemini()
	if creds == nil {
		return "", fmt.Errorf("gemini credentials not found, run: llmux auth gemini")
	}

	if !creds.Expiry.IsZero() && creds.Expiry.Before(time.Now().Add(5*time.Minute)) {
		if err := RefreshGeminiToken(store); err != nil {
			return "", fmt.Errorf("refreshing gemini token: %w", err)
		}
		// re-read the updated credentials after refresh
		creds = store.GetGemini()
		if creds == nil {
			return "", fmt.Errorf("gemini credentials missing after refresh")
		}
	}

	return creds.Token, nil
}

// JSON body structure returned by the Google token endpoint
type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	TokenType    string `json:"token_type"`
}

// posts the authorization code to the token endpoint and returns a populated GeminiCredentials struct
func exchangeCodeForTokens(code, redirectURI string) (*GeminiCredentials, error) {
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("client_id", config.GeminiClientID)
	form.Set("client_secret", config.GeminiClientSecret)
	form.Set("code", code)
	form.Set("redirect_uri", redirectURI)

	resp, err := http.PostForm(geminiTokenURL, form)
	if err != nil {
		return nil, fmt.Errorf("token request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading token response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token endpoint returned HTTP %d: %s", resp.StatusCode, body)
	}

	var tr tokenResponse
	if err := json.Unmarshal(body, &tr); err != nil {
		return nil, fmt.Errorf("parsing token response: %w", err)
	}
	if tr.AccessToken == "" {
		return nil, fmt.Errorf("no access_token in token response")
	}

	creds := &GeminiCredentials{
		Token:        tr.AccessToken,
		RefreshToken: tr.RefreshToken,
		TokenURI:     geminiTokenURL,
		ClientID:     config.GeminiClientID,
		ClientSecret: config.GeminiClientSecret,
		Expiry:       time.Now().Add(time.Duration(tr.ExpiresIn) * time.Second),
	}
	return creds, nil
}

// success page
const successHTML = `<h1>Gemini authentication successful!</h1>` +
	`<p>You can close this window.</p>`
