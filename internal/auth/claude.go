package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"llmux/internal/config"
)

const claudeRedirectURI = "http://localhost:53692/callback"

// Claude OAuth2 authorization Code + PKCE flow
func RunClaudeOAuthFlow(store *CredentialStore) error {
	// generate PKCE (proof key for code exchange) verifier (32 random bytes -> base64url)
	verifierBytes := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, verifierBytes); err != nil {
		return fmt.Errorf("generate pkce verifier: %w", err)
	}
	verifier := base64.RawURLEncoding.EncodeToString(verifierBytes)

	// generate code challenge: SHA-256 of the verifier STRING (RFC 7636) -> base64url
	challengeSum := sha256.Sum256([]byte(verifier))
	challenge := base64.RawURLEncoding.EncodeToString(challengeSum[:])

	// generate random state for CSRF (cross-site request forgery) protection
	stateBytes := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, stateBytes); err != nil {
		return fmt.Errorf("generate state: %w", err)
	}
	state := hex.EncodeToString(stateBytes)

	// build authorization URL
	params := url.Values{}
	params.Set("client_id", config.ClaudeClientID)
	params.Set("response_type", "code")
	params.Set("redirect_uri", claudeRedirectURI)
	params.Set("scope", strings.Join(config.ClaudeScopes, " "))
	params.Set("code_challenge", challenge)
	params.Set("code_challenge_method", "S256")
	params.Set("state", state)

	authURL := config.ClaudeAuthURL + "?" + params.Encode()

	// print URL
	fmt.Println("Open this URL in your browser:\n" + authURL)

	// start local HTTP server to receive the OAuth callback
	codeCh := make(chan string, 1)
	errCh := make(chan error, 1)

	mux := http.NewServeMux()
	srv := &http.Server{Addr: ":53692", Handler: mux}

	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()

		// CSRF check
		if q.Get("state") != state {
			http.Error(w, "state mismatch", http.StatusBadRequest)
			errCh <- fmt.Errorf("oauth state mismatch: possible CSRF attack")
			return
		}

		code := q.Get("code")
		if code == "" {
			http.Error(w, "missing code", http.StatusBadRequest)
			errCh <- fmt.Errorf("oauth callback missing code parameter")
			return
		}

		// serve success page
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, `<!DOCTYPE html><html><body>`+
			`<h2>Claude authorization successful!</h2>`+
			`<p>You can close this window.</p>`+
			`</body></html>`)

		codeCh <- code

		// shut down server asynchronously so !response is sent first!
		go func() { _ = srv.Shutdown(context.Background()) }()
	})

	// run server in a goroutine.
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("callback server: %w", err)
		}
	}()

	// wait for either the code or error
	var authCode string
	select {
	case authCode = <-codeCh:
	case err := <-errCh:
		return err
	}

	// exchange the authorization code for tokens.
	tokens, err := claudeTokenRequest(map[string]string{
		"grant_type":    "authorization_code",
		"client_id":     config.ClaudeClientID,
		"code":          authCode,
		"redirect_uri":  claudeRedirectURI,
		"code_verifier": verifier,
		"state":         state,
	})
	if err != nil {
		return fmt.Errorf("token exchange: %w", err)
	}

	// build credentials and persist it
	creds := buildClaudeCredentials(tokens)
	return store.UpdateClaude(creds)
}

// obtain new access token by stored refresh token (and persists the updated creds)
func RefreshClaudeToken(store *CredentialStore) error {
	creds := store.GetClaude()
	if creds == nil {
		return fmt.Errorf("no Claude credentials found in store")
	}

	tokens, err := claudeTokenRequest(map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     config.ClaudeClientID,
		"refresh_token": creds.RefreshToken,
	})
	if err != nil {
		return fmt.Errorf("refresh token: %w", err)
	}

	return store.UpdateClaude(buildClaudeCredentials(tokens))
}

// returns valid Claude access token (and refreshing it automatically when close to expiry)
func EnsureClaudeToken(store *CredentialStore) (string, error) {
	creds := store.GetClaude()
	if creds == nil {
		return "", fmt.Errorf("claude credentials not found, run: llmux auth claude")
	}

	// refresh if token expires within 5 minutes
	if !creds.Expiry.IsZero() && creds.Expiry.Before(time.Now().Add(5*time.Minute)) {
		if err := RefreshClaudeToken(store); err != nil {
			return "", fmt.Errorf("refresh Claude token: %w", err)
		}
		// re-read the updated credentials
		creds = store.GetClaude()
		if creds == nil {
			return "", fmt.Errorf("credentials missing after refresh")
		}
	}

	return creds.AccessToken, nil
}

// subset of the token endpoint
type claudeTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
}

// make POSTs by given fields as JSON to the Claude token URL
// (platform.claude.com API uses JSON like the rest of Anthropic's platform)
func claudeTokenRequest(body map[string]string) (*claudeTokenResponse, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, config.ClaudeTokenURL, strings.NewReader(string(jsonBody)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token endpoint returned %d: %s", resp.StatusCode, raw)
	}

	var tokens claudeTokenResponse
	if err := json.Unmarshal(raw, &tokens); err != nil {
		return nil, fmt.Errorf("parse token response: %w", err)
	}

	return &tokens, nil
}

// converts a token response into a ClaudeCredentials
func buildClaudeCredentials(tokens *claudeTokenResponse) *ClaudeCredentials {
	expiry := time.Now().Add(time.Duration(tokens.ExpiresIn)*time.Second - 5*time.Minute)
	return &ClaudeCredentials{
		AccessToken:  tokens.AccessToken,
		RefreshToken: tokens.RefreshToken,
		Expiry:       expiry,
	}
}
