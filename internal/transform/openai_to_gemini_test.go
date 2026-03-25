package transform

import (
	"testing"

	"llmux/internal/config"
)

func TestBuildGeminiPayload(t *testing.T) {
	inner := map[string]any{
		"contents": []any{"hello"},
	}
	result := BuildGeminiPayload(inner, "gemini-2.5-pro", "my-project")

	if got, want := result["model"], "gemini-2.5-pro"; got != want {
		t.Errorf("model = %q, want %q", got, want)
	}
	if got, want := result["project"], "my-project"; got != want {
		t.Errorf("project = %q, want %q", got, want)
	}
	if result["request"] == nil {
		t.Error("request should not be nil")
	}
	req, ok := result["request"].(map[string]any)
	if !ok {
		t.Fatal("request should be map[string]any")
	}
	if req["contents"] == nil {
		t.Error("request.contents should be present")
	}
}

func TestBuildGeminiPayload_ModelPrefix(t *testing.T) {
	result := BuildGeminiPayload(map[string]any{}, "gemini-2.5-flash", "proj")
	if got := result["model"]; got != "gemini-2.5-flash" {
		t.Errorf("model = %q, want %q", got, "gemini-2.5-flash")
	}
}

func TestBuildGeminiPayloadFromNative_Wraps(t *testing.T) {
	config.Cfg.GoogleCloudProject = "test-project"

	native := map[string]any{
		"contents": []any{"hi"},
	}
	result := BuildGeminiPayloadFromNative(native, "gemini-2.5-pro")

	if got, want := result["model"], "gemini-2.5-pro"; got != want {
		t.Errorf("model = %q, want %q", got, want)
	}
	if got, want := result["project"], "test-project"; got != want {
		t.Errorf("project = %q, want %q", got, want)
	}
	if result["request"] == nil {
		t.Error("request should not be nil")
	}
}

func TestBuildGeminiPayloadFromNative_InjectsSafetySettings(t *testing.T) {
	config.Cfg.GoogleCloudProject = "proj"

	native := map[string]any{}
	result := BuildGeminiPayloadFromNative(native, "gemini-2.5-flash")

	req, ok := result["request"].(map[string]any)
	if !ok {
		t.Fatal("request should be map[string]any")
	}
	if req["safetySettings"] == nil {
		t.Error("safetySettings should be injected")
	}
	if req["generationConfig"] == nil {
		t.Error("generationConfig should be injected")
	}
}

func TestBuildGeminiPayloadFromNative_ThinkingBudgetPreserved(t *testing.T) {
	config.Cfg.GoogleCloudProject = "proj"

	native := map[string]any{
		"generationConfig": map[string]any{
			"thinkingConfig": map[string]any{
				"thinkingBudget": 9999,
			},
		},
	}
	result := BuildGeminiPayloadFromNative(native, "gemini-2.5-pro")

	req := result["request"].(map[string]any)
	genCfg := req["generationConfig"].(map[string]any)
	thinkingCfg := genCfg["thinkingConfig"].(map[string]any)
	if got := thinkingCfg["thinkingBudget"]; got != 9999 {
		t.Errorf("thinkingBudget = %v, want 9999 (should be preserved)", got)
	}
}

func TestBuildGeminiPayloadFromNative_SearchModel(t *testing.T) {
	config.Cfg.GoogleCloudProject = "proj"

	native := map[string]any{}
	result := BuildGeminiPayloadFromNative(native, "gemini-2.5-pro-search")

	req := result["request"].(map[string]any)
	tools, ok := req["tools"].([]any)
	if !ok || len(tools) == 0 {
		t.Fatal("tools should be set for search model")
	}
	foundSearch := false
	for _, t := range tools {
		if tm, ok := t.(map[string]any); ok {
			if _, ok := tm["googleSearch"]; ok {
				foundSearch = true
				break
			}
		}
	}
	if !foundSearch {
		t.Error("googleSearch tool should be present for search model")
	}
	if got, want := result["model"], "gemini-2.5-pro"; got != want {
		t.Errorf("model = %q, want %q", got, want)
	}
}
