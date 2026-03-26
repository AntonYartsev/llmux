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

func TestBuildGeminiPayloadFromNative_DefaultThinkingConfig(t *testing.T) {
	config.Cfg.GoogleCloudProject = "proj"

	native := map[string]any{}
	result := BuildGeminiPayloadFromNative(native, "gemini-2.5-pro")

	req := result["request"].(map[string]any)
	genCfg := req["generationConfig"].(map[string]any)
	thinkingCfg, ok := genCfg["thinkingConfig"].(map[string]any)
	if !ok {
		t.Fatal("thinkingConfig should be set")
	}
	if thinkingCfg["includeThoughts"] != true {
		t.Error("includeThoughts should be true by default")
	}
}

func TestOpenAIRequestToGemini_WebSearchPreview(t *testing.T) {
	req := &ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []ChatMessage{{Role: "user", Content: "search the web"}},
		Tools: []map[string]interface{}{
			{"type": "web_search_preview"},
		},
	}
	result := OpenAIRequestToGemini(req)

	tools, ok := result["tools"].([]map[string]interface{})
	if !ok || len(tools) == 0 {
		t.Fatal("tools should be set")
	}
	foundSearch := false
	for _, tool := range tools {
		if _, ok := tool["googleSearch"]; ok {
			foundSearch = true
			break
		}
	}
	if !foundSearch {
		t.Error("googleSearch tool should be present when web_search_preview is in tools")
	}
}

func TestOpenAIRequestToGemini_WebSearchPreviewWithFunctionTools(t *testing.T) {
	req := &ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []ChatMessage{{Role: "user", Content: "search and use tool"}},
		Tools: []map[string]interface{}{
			{"type": "web_search_preview"},
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather",
					"parameters":  map[string]interface{}{"type": "object"},
				},
			},
		},
	}
	result := OpenAIRequestToGemini(req)

	tools, ok := result["tools"].([]map[string]interface{})
	if !ok || len(tools) < 2 {
		t.Fatalf("expected at least 2 tool entries, got %d", len(tools))
	}
	foundSearch := false
	foundFunction := false
	for _, tool := range tools {
		if _, ok := tool["googleSearch"]; ok {
			foundSearch = true
		}
		if _, ok := tool["functionDeclarations"]; ok {
			foundFunction = true
		}
	}
	if !foundSearch {
		t.Error("googleSearch tool missing")
	}
	if !foundFunction {
		t.Error("functionDeclarations missing")
	}
}

func TestOpenAIRequestToGemini_ReasoningEffort(t *testing.T) {
	effort := "high"
	req := &ChatRequest{
		Model:           "gemini-2.5-pro",
		Messages:        []ChatMessage{{Role: "user", Content: "think hard"}},
		ReasoningEffort: &effort,
	}
	result := OpenAIRequestToGemini(req)

	genCfg, ok := result["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("generationConfig missing")
	}
	thinkingCfg, ok := genCfg["thinkingConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("thinkingConfig missing")
	}
	budget, ok := thinkingCfg["thinkingBudget"].(int)
	if !ok {
		t.Fatal("thinkingBudget missing")
	}
	if budget != 32768 {
		t.Errorf("thinkingBudget = %d, want 32768", budget)
	}
}
