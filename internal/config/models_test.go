package config

import (
	"testing"
)

func TestGeminiBaseModels(t *testing.T) {
	expected := []string{
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-3-pro-preview",
		"gemini-3-flash-preview",
		"gemini-3.1-pro-preview",
		"gemini-3.1-flash-lite-preview",
	}

	if len(GeminiBaseModels) != len(expected) {
		t.Fatalf("expected %d gemini models, got %d", len(expected), len(GeminiBaseModels))
	}

	names := make(map[string]bool)
	for _, m := range GeminiBaseModels {
		names[m.Name] = true
	}
	for _, name := range expected {
		if !names[name] {
			t.Errorf("expected gemini model %q not found", name)
		}
	}
}

func TestClaudeModels(t *testing.T) {
	expected := []string{
		"claude-sonnet-4-6",
		"claude-opus-4-6",
		"claude-haiku-4-5",
	}

	names := make(map[string]bool)
	for _, m := range ClaudeModels {
		names[m.Name] = true
	}
	for _, name := range expected {
		if !names[name] {
			t.Errorf("expected claude model %q not found", name)
		}
	}
}

func TestParseFallbackChains(t *testing.T) {
	t.Run("empty string", func(t *testing.T) {
		result := ParseFallbackChains("")
		if len(result) != 0 {
			t.Errorf("expected empty map, got %v", result)
		}
	})

	t.Run("single chain", func(t *testing.T) {
		result := ParseFallbackChains("model1:fallback1,fallback2")
		if len(result) != 1 {
			t.Fatalf("expected 1 entry, got %d", len(result))
		}
		fbs := result["model1"]
		if len(fbs) != 2 || fbs[0] != "fallback1" || fbs[1] != "fallback2" {
			t.Errorf("unexpected fallbacks: %v", fbs)
		}
	})

	t.Run("multiple chains", func(t *testing.T) {
		result := ParseFallbackChains("model1:fallback1,fallback2;model2:fallback3")
		if len(result) != 2 {
			t.Fatalf("expected 2 entries, got %d", len(result))
		}
		fbs1 := result["model1"]
		if len(fbs1) != 2 || fbs1[0] != "fallback1" || fbs1[1] != "fallback2" {
			t.Errorf("model1 fallbacks: %v", fbs1)
		}
		fbs2 := result["model2"]
		if len(fbs2) != 1 || fbs2[0] != "fallback3" {
			t.Errorf("model2 fallbacks: %v", fbs2)
		}
	})

	t.Run("trailing semicolon", func(t *testing.T) {
		result := ParseFallbackChains("model1:fb1;")
		if len(result) != 1 {
			t.Errorf("expected 1 entry, got %d", len(result))
		}
	})
}

func TestParsePrefixedModel(t *testing.T) {
	tests := []struct {
		input      string
		wantPrefix string
		wantBare   string
	}{
		{"gemini/gemini-2.5-pro", "gemini", "gemini-2.5-pro"},
		{"claude/claude-sonnet-4-6", "claude", "claude-sonnet-4-6"},
		{"openrouter/gemini-2.5-pro", "openrouter", "gemini-2.5-pro"},
		{"gemini-2.5-pro", "", "gemini-2.5-pro"},
		{"claude-sonnet-4-6", "", "claude-sonnet-4-6"},
		{"", "", ""},
	}
	for _, tc := range tests {
		prefix, bare := ParsePrefixedModel(tc.input)
		if prefix != tc.wantPrefix || bare != tc.wantBare {
			t.Errorf("ParsePrefixedModel(%q) = (%q, %q), want (%q, %q)",
				tc.input, prefix, bare, tc.wantPrefix, tc.wantBare)
		}
	}
}

func TestResolveBackendName(t *testing.T) {
	tests := []struct {
		model      string
		backendMap map[string]string
		want       string
	}{
		{"my-model", map[string]string{"my-model": "custom"}, "custom"},
		{"claude-sonnet-4-6", nil, "claude"},
		{"claude-opus-4-6", map[string]string{}, "claude"},
		{"models/gemini-2.5-pro", nil, "gemini"},
		{"some-other-model", map[string]string{}, "gemini"},
		{"claude-special", map[string]string{"claude-special": "gemini"}, "gemini"},
		// vendor-prefixed models
		{"gemini/gemini-2.5-pro", nil, "gemini"},
		{"claude/claude-sonnet-4-6", nil, "claude"},
		{"openrouter/gemini-2.5-pro", nil, "openrouter"},
		{"claude/custom-model", nil, "claude"},
		{"anthropic/claude-3", nil, "claude"},
		// bare name in backendMap takes priority when prefixed
		{"gemini/my-model", map[string]string{"my-model": "custom"}, "custom"},
		// prefixed model in backendMap takes priority
		{"gemini/gemini-2.5-pro", map[string]string{"gemini/gemini-2.5-pro": "custom"}, "custom"},
	}
	for _, tc := range tests {
		got := ResolveBackendName(tc.model, tc.backendMap)
		if got != tc.want {
			t.Errorf("ResolveBackendName(%q, %v) = %q, want %q", tc.model, tc.backendMap, got, tc.want)
		}
	}
}
