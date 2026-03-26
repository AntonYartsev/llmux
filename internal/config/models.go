package config

import "strings"

// holds metadata about a single Gemini or Claude model
type ModelInfo struct {
	Name                       string
	Version                    string
	DisplayName                string
	Description                string
	InputTokenLimit            int
	OutputTokenLimit           int
	SupportedGenerationMethods []string
	Temperature                float64
	MaxTemperature             float64
	TopP                       float64
	TopK                       int
}

// canonical list of supported Gemini models
// source: https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/config/models.ts
var GeminiBaseModels = []ModelInfo{
	{
		Name:                       "gemini-2.5-pro",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Pro",
		Description:                "advanced multimodal model with enhanced capabilities",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-flash",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Flash",
		Description:                "fast and efficient multimodal model",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-flash-lite",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Flash Lite",
		Description:                "lightweight version of Gemini 2.5 Flash",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-3-pro-preview",
		Version:                    "001",
		DisplayName:                "Gemini 3.0 Pro Preview",
		Description:                "preview version of Gemini 3.0 Pro",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-3-flash-preview",
		Version:                    "001",
		DisplayName:                "Gemini 3.0 Flash Preview",
		Description:                "preview version of Gemini 3.0 Flash",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-3.1-pro-preview",
		Version:                    "001",
		DisplayName:                "Gemini 3.1 Pro Preview",
		Description:                "preview version of Gemini 3.1 Pro",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-3.1-flash-lite-preview",
		Version:                    "001",
		DisplayName:                "Gemini 3.1 Flash Lite Preview",
		Description:                "preview version of Gemini 3.1 Flash Lite",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
}

// static list of supported Claude models
// source: https://github.com/anthropics/claude-code/blob/main/packages/core/src/llm/model-config.ts
var ClaudeModels = []ModelInfo{
	{Name: "claude-sonnet-4-6", DisplayName: "Claude Sonnet 4.6", Description: "Claude Sonnet 4.6"},
	{Name: "claude-opus-4-6", DisplayName: "Claude Opus 4.6", Description: "Claude Opus 4.6"},
	{Name: "claude-haiku-4-5", DisplayName: "Claude Haiku 4.5", Description: "Claude Haiku 4.5"},
}

// parses a raw fallback chain string of the form
// "model1:fallback1,fallback2;model2:fallback3" into a map
func ParseFallbackChains(raw string) map[string][]string {
	result := make(map[string][]string)
	if raw == "" {
		return result
	}
	for _, entry := range strings.Split(raw, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.SplitN(entry, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		if key == "" {
			continue
		}
		var fallbacks []string
		for _, fb := range strings.Split(parts[1], ",") {
			fb = strings.TrimSpace(fb)
			if fb != "" {
				fallbacks = append(fallbacks, fb)
			}
		}
		result[key] = fallbacks
	}
	return result
}

// splits a "provider/model" string into its prefix and bare model name
// if there is no slash, prefix is empty and bare is the full input
// legacy "models/" prefix is not treated as a vendor prefix
func ParsePrefixedModel(model string) (prefix, bare string) {
	if i := strings.IndexByte(model, '/'); i >= 0 {
		p := model[:i]
		if p == "models" {
			return "", model[i+1:]
		}
		return p, model[i+1:]
	}
	return "", model
}

// determines which backend ("gemini" or "claude") should handle a given model
// an explicit entry in backendMap takes priority; a vendor prefix (e.g. "claude/model") is checked next
func ResolveBackendName(model string, backendMap map[string]string) string {
	if backend, ok := backendMap[model]; ok {
		return backend
	}
	// check vendor prefix (e.g. "gemini/gemini-2.5-pro" -> "gemini")
	if prefix, bare := ParsePrefixedModel(model); prefix != "" {
		// also check the bare name in the explicit map
		if backend, ok := backendMap[bare]; ok {
			return backend
		}
		// normalize known aliases
		if prefix == "anthropic" {
			return "claude"
		}
		return prefix
	}
	if strings.HasPrefix(model, "claude-") {
		return "claude"
	}
	return "gemini"
}
