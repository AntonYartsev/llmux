package auth

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// OAuth2 credentials for Gemini backend
type GeminiCredentials struct {
	Token        string    `json:"token"`
	RefreshToken string    `json:"refresh_token"`
	TokenURI     string    `json:"token_uri"`
	ClientID     string    `json:"client_id"`
	ClientSecret string    `json:"client_secret"`
	Expiry       time.Time `json:"expiry"`
	ProjectID    string    `json:"project_id,omitempty"`
}

// OAuth2 credentials for Claude backend
type ClaudeCredentials struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	Expiry       time.Time `json:"expiry"`
}

// persistent credential storage backed by JSON file
type CredentialStore struct {
	Gemini   *GeminiCredentials `json:"gemini,omitempty"`
	Claude   *ClaudeCredentials `json:"claude,omitempty"`
	filePath string
	mu       sync.RWMutex
}

// creates a CredentialStore for given file path
// leading "~/" is expanded to user's home directory
func NewCredentialStore(filePath string) *CredentialStore {
	if len(filePath) >= 2 && filePath[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err == nil {
			filePath = filepath.Join(home, filePath[2:])
		}
	}
	return &CredentialStore{filePath: filePath}
}

// reads credentials from JSON file on disk
// missing file is treated as an empty store (no error returned)
func (s *CredentialStore) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.filePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}

	return json.Unmarshal(data, s)
}

// writes the current credentials to the JSON file
// parent directories are created automatically if they do not exist
// it acquires a read lock to take a consistent snapshot before writing
func (s *CredentialStore) Save() error {
	s.mu.RLock()
	data, err := json.MarshalIndent(s, "", "  ")
	s.mu.RUnlock()
	if err != nil {
		return err
	}

	dir := filepath.Dir(s.filePath)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}

	return os.WriteFile(s.filePath, data, 0o600)
}

// save internal variant used by Update* methods that already hold the write lock
// it snapshots the data under the existing lock via a temporary plain struct, then releases the lock before performing I/O
func (s *CredentialStore) save() error {
	// copy pointer values while still under the caller's write lock so that
	// json.MarshalIndent sees a consistent view without needing to re-acquire
	type snapshot struct {
		Gemini *GeminiCredentials `json:"gemini,omitempty"`
		Claude *ClaudeCredentials `json:"claude,omitempty"`
	}
	snap := snapshot{Gemini: s.Gemini, Claude: s.Claude}
	// unlock before I/O so other readers are not blocked unnecessarily
	s.mu.Unlock()
	defer s.mu.Lock() // re-acquire so the caller's defer Unlock still works

	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return err
	}

	dir := filepath.Dir(s.filePath)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}

	return os.WriteFile(s.filePath, data, 0o600)
}

// parse the GEMINI_CREDENTIALS and CLAUDE_CREDENTIALS environment
// variables (each expected to be a JSON string) and overrides the in-memory credentials when the variable is non-empty
func (s *CredentialStore) LoadFromEnv() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if raw := os.Getenv("GEMINI_CREDENTIALS"); raw != "" {
		var creds GeminiCredentials
		if err := json.Unmarshal([]byte(raw), &creds); err != nil {
			return err
		}
		s.Gemini = &creds
	}

	if raw := os.Getenv("CLAUDE_CREDENTIALS"); raw != "" {
		var creds ClaudeCredentials
		if err := json.Unmarshal([]byte(raw), &creds); err != nil {
			return err
		}
		s.Claude = &creds
	}

	return nil
}

// returns the current Gemini credentials in a thread-safe mode
func (s *CredentialStore) GetGemini() *GeminiCredentials {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Gemini
}

// returns the current Claude credentials in a thread-safe mode
func (s *CredentialStore) GetClaude() *ClaudeCredentials {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Claude
}

// replace the stored Gemini credentials and persists
func (s *CredentialStore) UpdateGemini(creds *GeminiCredentials) error {
	s.mu.Lock()
	s.Gemini = creds
	err := s.save()
	s.mu.Unlock()
	return err
}

// replace the stored Claude credentials and persists
func (s *CredentialStore) UpdateClaude(creds *ClaudeCredentials) error {
	s.mu.Lock()
	s.Claude = creds
	err := s.save()
	s.mu.Unlock()
	return err
}
