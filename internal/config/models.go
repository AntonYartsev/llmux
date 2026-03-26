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

// canonical list of base Gemini models (no variant suffixes)
var GeminiBaseModels = []ModelInfo{
	{
		Name:                       "gemini-2.5-pro-preview-03-25",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Pro Preview 03-25",
		Description:                "Preview version of Gemini 2.5 Pro from May 6th",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-pro-preview-05-06",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Pro Preview 05-06",
		Description:                "Preview version of Gemini 2.5 Pro from May 6th",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-pro-preview-06-05",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Pro Preview 06-05",
		Description:                "Preview version of Gemini 2.5 Pro from June 5th",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-pro",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Pro",
		Description:                "Advanced multimodal model with enhanced capabilities",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-flash-preview-05-20",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Flash Preview 05-20",
		Description:                "Preview version of Gemini 2.5 Flash from May 20th",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-flash-preview-04-17",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Flash Preview 04-17",
		Description:                "Preview version of Gemini 2.5 Flash from April 17th",
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
		Description:                "Fast and efficient multimodal model with latest improvements",
		InputTokenLimit:            1048576,
		OutputTokenLimit:           65535,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-2.5-flash-image-preview",
		Version:                    "001",
		DisplayName:                "Gemini 2.5 Flash Image Preview",
		Description:                "Gemini 2.5 Flash Image Preview",
		InputTokenLimit:            32768,
		OutputTokenLimit:           32768,
		SupportedGenerationMethods: []string{"generateContent", "streamGenerateContent"},
		Temperature:                1.0,
		MaxTemperature:             2.0,
		TopP:                       0.95,
		TopK:                       64,
	},
	{
		Name:                       "gemini-3-pro-preview",
		Version:                    "001",
		DisplayName:                "Gemini 3.0 Pro Preview 11-2025",
		Description:                "Preview version of Gemini 3.0 Pro from November 2025",
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
		Description:                "Preview version of Gemini 3.0 Flash",
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
var ClaudeModels = []ModelInfo{
	{Name: "claude-sonnet-4-6", DisplayName: "Claude Sonnet 4.6", Description: "Claude Sonnet 4.6"},
	{Name: "claude-opus-4-6", DisplayName: "Claude Opus 4.6", Description: "Claude Opus 4.6"},
	{Name: "claude-haiku-4-5", DisplayName: "Claude Haiku 4.5", Description: "Claude Haiku 4.5"},
}

// returns the full set of Gemini models: base models plus
// search, thinking, and combined (search+thinking) variants
func GenerateAllGeminiModels() []ModelInfo {
	all := make([]ModelInfo, 0, len(GeminiBaseModels)*6)

	// base models
	all = append(all, GeminiBaseModels...)

	// search variants: all models except those containing "gemini-2.5-flash-image"
	for _, m := range GeminiBaseModels {
		if strings.Contains(m.Name, "gemini-2.5-flash-image") {
			continue
		}
		variant := m
		variant.Name = m.Name + "-search"
		variant.DisplayName = m.DisplayName + " with Google Search"
		variant.Description = m.Description + " (includes Google Search grounding)"
		all = append(all, variant)
	}

	// thinking variants: models containing "gemini-2.5-flash" or "gemini-2.5-pro"
	// but NOT "gemini-2.5-flash-image"
	for _, m := range GeminiBaseModels {
		if strings.Contains(m.Name, "gemini-2.5-flash-image") {
			continue
		}
		if !strings.Contains(m.Name, "gemini-2.5-flash") && !strings.Contains(m.Name, "gemini-2.5-pro") {
			continue
		}

		nothinking := m
		nothinking.Name = m.Name + "-nothinking"
		nothinking.DisplayName = m.DisplayName + " (No Thinking)"
		nothinking.Description = m.Description + " (thinking disabled)"
		all = append(all, nothinking)

		maxthinking := m
		maxthinking.Name = m.Name + "-maxthinking"
		maxthinking.DisplayName = m.DisplayName + " (Max Thinking)"
		maxthinking.Description = m.Description + " (maximum thinking budget)"
		all = append(all, maxthinking)
	}

	// combined variants (search + thinking): same eligibility as thinking variants
	for _, m := range GeminiBaseModels {
		if strings.Contains(m.Name, "gemini-2.5-flash-image") {
			continue
		}
		if !strings.Contains(m.Name, "gemini-2.5-flash") && !strings.Contains(m.Name, "gemini-2.5-pro") {
			continue
		}

		searchNothinking := m
		searchNothinking.Name = m.Name + "-search-nothinking"
		searchNothinking.DisplayName = m.DisplayName + " with Google Search (No Thinking)"
		searchNothinking.Description = m.Description + " (includes Google Search grounding, thinking disabled)"
		all = append(all, searchNothinking)

		searchMaxthinking := m
		searchMaxthinking.Name = m.Name + "-search-maxthinking"
		searchMaxthinking.DisplayName = m.DisplayName + " with Google Search (Max Thinking)"
		searchMaxthinking.Description = m.Description + " (includes Google Search grounding, maximum thinking budget)"
		all = append(all, searchMaxthinking)
	}

	return all
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
