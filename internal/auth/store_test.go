package auth

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestCredentialStore_LoadSave(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "creds.json")

	expiry := time.Date(2025, 12, 31, 0, 0, 0, 0, time.UTC)

	initial := &CredentialStore{
		Gemini: &GeminiCredentials{
			Token:        "gemini-token",
			RefreshToken: "gemini-refresh",
			TokenURI:     "https://oauth2.googleapis.com/token",
			ClientID:     "client-id",
			ClientSecret: "client-secret",
			Expiry:       expiry,
			ProjectID:    "my-project",
		},
		Claude: &ClaudeCredentials{
			AccessToken:  "claude-access",
			RefreshToken: "claude-refresh",
			Expiry:       expiry,
		},
		filePath: filePath,
	}

	data, err := json.MarshalIndent(initial, "", "  ")
	if err != nil {
		t.Fatalf("marshal initial: %v", err)
	}
	if err := os.WriteFile(filePath, data, 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	store := NewCredentialStore(filePath)
	if err := store.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	g := store.GetGemini()
	if g == nil {
		t.Fatal("expected Gemini credentials, got nil")
	}
	if g.Token != "gemini-token" {
		t.Errorf("Token: got %q, want %q", g.Token, "gemini-token")
	}
	if g.ProjectID != "my-project" {
		t.Errorf("ProjectID: got %q, want %q", g.ProjectID, "my-project")
	}
	if !g.Expiry.Equal(expiry) {
		t.Errorf("Expiry: got %v, want %v", g.Expiry, expiry)
	}

	c := store.GetClaude()
	if c == nil {
		t.Fatal("expected Claude credentials, got nil")
	}
	if c.AccessToken != "claude-access" {
		t.Errorf("AccessToken: got %q, want %q", c.AccessToken, "claude-access")
	}

	newExpiry := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	if err := store.UpdateGemini(&GeminiCredentials{
		Token:  "gemini-token-v2",
		Expiry: newExpiry,
	}); err != nil {
		t.Fatalf("UpdateGemini: %v", err)
	}

	store2 := NewCredentialStore(filePath)
	if err := store2.Load(); err != nil {
		t.Fatalf("Load (store2): %v", err)
	}
	g2 := store2.GetGemini()
	if g2 == nil {
		t.Fatal("expected Gemini credentials after reload, got nil")
	}
	if g2.Token != "gemini-token-v2" {
		t.Errorf("Token after reload: got %q, want %q", g2.Token, "gemini-token-v2")
	}
	if !g2.Expiry.Equal(newExpiry) {
		t.Errorf("Expiry after reload: got %v, want %v", g2.Expiry, newExpiry)
	}
}

func TestCredentialStore_LoadFromEnv(t *testing.T) {
	expiry := time.Date(2025, 11, 1, 0, 0, 0, 0, time.UTC)

	geminiJSON, err := json.Marshal(&GeminiCredentials{
		Token:        "env-gemini-token",
		RefreshToken: "env-gemini-refresh",
		TokenURI:     "https://oauth2.googleapis.com/token",
		ClientID:     "env-client-id",
		ClientSecret: "env-client-secret",
		Expiry:       expiry,
		ProjectID:    "env-project",
	})
	if err != nil {
		t.Fatalf("marshal gemini: %v", err)
	}

	claudeJSON, err := json.Marshal(&ClaudeCredentials{
		AccessToken:  "env-claude-access",
		RefreshToken: "env-claude-refresh",
		Expiry:       expiry,
	})
	if err != nil {
		t.Fatalf("marshal claude: %v", err)
	}

	t.Setenv("GEMINI_CREDENTIALS", string(geminiJSON))
	t.Setenv("CLAUDE_CREDENTIALS", string(claudeJSON))

	store := NewCredentialStore("/tmp/nonexistent-creds-loadfromenv.json")
	if err := store.LoadFromEnv(); err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}

	g := store.GetGemini()
	if g == nil {
		t.Fatal("expected Gemini credentials, got nil")
	}
	if g.Token != "env-gemini-token" {
		t.Errorf("Token: got %q, want %q", g.Token, "env-gemini-token")
	}
	if g.ProjectID != "env-project" {
		t.Errorf("ProjectID: got %q, want %q", g.ProjectID, "env-project")
	}
	if !g.Expiry.Equal(expiry) {
		t.Errorf("Expiry: got %v, want %v", g.Expiry, expiry)
	}

	c := store.GetClaude()
	if c == nil {
		t.Fatal("expected Claude credentials, got nil")
	}
	if c.AccessToken != "env-claude-access" {
		t.Errorf("AccessToken: got %q, want %q", c.AccessToken, "env-claude-access")
	}
	if c.RefreshToken != "env-claude-refresh" {
		t.Errorf("RefreshToken: got %q, want %q", c.RefreshToken, "env-claude-refresh")
	}
}

func TestCredentialStore_ThreadSafety(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "creds-race.json")

	store := NewCredentialStore(filePath)
	store.Gemini = &GeminiCredentials{Token: "initial"}

	const goroutines = 10
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(i int) {
			defer wg.Done()
			if i%2 == 0 {
				_ = store.GetGemini()
			} else {
				_ = store.UpdateGemini(&GeminiCredentials{
					Token: "token-from-goroutine",
				})
			}
		}(i)
	}

	wg.Wait()

	g := store.GetGemini()
	if g == nil {
		t.Fatal("expected non-nil Gemini credentials after concurrent access")
	}
}

func TestCredentialStore_FileNotFound(t *testing.T) {
	store := NewCredentialStore("/tmp/this-file-definitely-does-not-exist-xyz123.json")
	if err := store.Load(); err != nil {
		t.Errorf("Load with nonexistent file should return nil, got: %v", err)
	}
	if store.GetGemini() != nil {
		t.Error("expected nil Gemini for empty store")
	}
	if store.GetClaude() != nil {
		t.Error("expected nil Claude for empty store")
	}
}
