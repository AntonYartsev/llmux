package config

import (
	"strings"
	"testing"
)

func TestGetBaseModelName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"gemini-2.5-pro", "gemini-2.5-pro"},
		{"gemini-2.5-pro-search", "gemini-2.5-pro"},
		{"gemini-2.5-pro-nothinking", "gemini-2.5-pro"},
		{"gemini-2.5-pro-maxthinking", "gemini-2.5-pro"},
		{"gemini-2.5-flash-nothinking", "gemini-2.5-flash"},
		{"gemini-2.5-flash-maxthinking", "gemini-2.5-flash"},
		{"gemini-2.5-pro-search-maxthinking", "gemini-2.5-pro"},
		{"gemini-2.5-pro-search-nothinking", "gemini-2.5-pro"},
		{"gemini-3-pro-preview", "gemini-3-pro-preview"},
	}
	for _, tc := range tests {
		got := GetBaseModelName(tc.input)
		if got != tc.want {
			t.Errorf("GetBaseModelName(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestIsSearchModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-2.5-pro-search", true},
		{"gemini-2.5-pro-search-nothinking", true},
		{"gemini-2.5-pro-search-maxthinking", true},
		{"gemini-2.5-pro", false},
		{"gemini-2.5-pro-nothinking", false},
		{"gemini-2.5-pro-maxthinking", false},
	}
	for _, tc := range tests {
		got := IsSearchModel(tc.model)
		if got != tc.want {
			t.Errorf("IsSearchModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestIsNothinkingModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-2.5-pro-nothinking", true},
		{"gemini-2.5-pro-search-nothinking", true},
		{"gemini-2.5-pro", false},
		{"gemini-2.5-pro-maxthinking", false},
		{"gemini-2.5-pro-search", false},
	}
	for _, tc := range tests {
		got := IsNothinkingModel(tc.model)
		if got != tc.want {
			t.Errorf("IsNothinkingModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestIsMaxthinkingModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-2.5-pro-maxthinking", true},
		{"gemini-2.5-pro-search-maxthinking", true},
		{"gemini-2.5-pro", false},
		{"gemini-2.5-pro-nothinking", false},
		{"gemini-2.5-pro-search", false},
	}
	for _, tc := range tests {
		got := IsMaxthinkingModel(tc.model)
		if got != tc.want {
			t.Errorf("IsMaxthinkingModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestGetThinkingBudget(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"models/gemini-2.5-flash-nothinking", 0},
		{"models/gemini-2.5-flash-preview-05-20-nothinking", 0},
		{"models/gemini-2.5-pro-nothinking", 128},
		{"models/gemini-2.5-pro-preview-03-25-nothinking", 128},
		{"models/gemini-2.5-flash-maxthinking", 24576},
		{"models/gemini-2.5-flash-preview-05-20-maxthinking", 24576},
		{"models/gemini-2.5-pro-maxthinking", 32768},
		{"models/gemini-2.5-pro-preview-03-25-maxthinking", 32768},
		{"models/gemini-2.5-pro", -1},
		{"models/gemini-2.5-flash", -1},
		{"models/gemini-3-pro-preview", -1},
	}
	for _, tc := range tests {
		got := GetThinkingBudget(tc.model)
		if got != tc.want {
			t.Errorf("GetThinkingBudget(%q) = %d, want %d", tc.model, got, tc.want)
		}
	}
}

func TestShouldIncludeThoughts(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"models/gemini-2.5-pro-nothinking", true},
		{"models/gemini-3-pro-preview-nothinking", true},
		{"models/gemini-2.5-flash-nothinking", false},
		{"models/gemini-2.5-pro", true},
		{"models/gemini-2.5-flash", true},
		{"models/gemini-2.5-pro-maxthinking", true},
		{"models/gemini-2.5-flash-maxthinking", true},
	}
	for _, tc := range tests {
		got := ShouldIncludeThoughts(tc.model)
		if got != tc.want {
			t.Errorf("ShouldIncludeThoughts(%q) = %v, want %v", tc.model, got, tc.want)
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

func TestGenerateAllGeminiModels(t *testing.T) {
	all := GenerateAllGeminiModels()
	base := len(GeminiBaseModels)

	if len(all) <= base {
		t.Errorf("GenerateAllGeminiModels() returned %d models, expected more than %d (base count)", len(all), base)
	}

	for _, m := range all {
		if strings.Contains(m.Name, "gemini-2.5-flash-image") {
			if strings.Contains(m.Name, "-nothinking") || strings.Contains(m.Name, "-maxthinking") {
				t.Errorf("image model should not have thinking variant: %q", m.Name)
			}
		}
	}

	baseNames := make(map[string]bool)
	for _, m := range GeminiBaseModels {
		baseNames[m.Name] = false
	}
	for _, m := range all {
		if _, ok := baseNames[m.Name]; ok {
			baseNames[m.Name] = true
		}
	}
	for name, found := range baseNames {
		if !found {
			t.Errorf("base model %q missing from GenerateAllGeminiModels output", name)
		}
	}

	allNames := make(map[string]bool)
	for _, m := range all {
		allNames[m.Name] = true
	}
	for _, m := range GeminiBaseModels {
		if strings.Contains(m.Name, "gemini-2.5-flash-image") {
			continue
		}
		if strings.Contains(m.Name, "gemini-2.5-flash") || strings.Contains(m.Name, "gemini-2.5-pro") {
			for _, suffix := range []string{"-nothinking", "-maxthinking", "-search-nothinking", "-search-maxthinking"} {
				variantName := m.Name + suffix
				if !allNames[variantName] {
					t.Errorf("expected variant %q to exist", variantName)
				}
			}
		}
	}
}
