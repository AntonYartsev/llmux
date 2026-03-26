// Package transform handles conversion between OpenAI and Gemini API formats.
package transform

import (
	"encoding/json"
	"regexp"
	"strings"

	"llmux/internal/config"
)

// represents an OpenAI chat message
type ChatMessage struct {
	Role             string                   `json:"role"`
	Content          interface{}              `json:"content"` // string or []map[string]interface{}
	ReasoningContent *string                  `json:"reasoning_content,omitempty"`
	ToolCalls        []map[string]interface{} `json:"tool_calls,omitempty"`
	ToolCallID       *string                  `json:"tool_call_id,omitempty"`
	Name             *string                  `json:"name,omitempty"`
}

// represents OpenAI stream_options parameter
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// represents an OpenAI chat completion request
type ChatRequest struct {
	Model               string                   `json:"model"`
	Messages            []ChatMessage            `json:"messages"`
	Stream              bool                     `json:"stream"`
	StreamOptions       *StreamOptions           `json:"stream_options,omitempty"`
	Temperature         *float64                 `json:"temperature,omitempty"`
	TopP                *float64                 `json:"top_p,omitempty"`
	MaxTokens           *int                     `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                     `json:"max_completion_tokens,omitempty"`
	Stop                interface{}              `json:"stop,omitempty"` // string or []string
	FrequencyPenalty    *float64                 `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64                 `json:"presence_penalty,omitempty"`
	N                   *int                     `json:"n,omitempty"`
	Seed                *int                     `json:"seed,omitempty"`
	ResponseFormat      map[string]interface{}   `json:"response_format,omitempty"`
	ReasoningEffort     *string                  `json:"reasoning_effort,omitempty"`
	Tools               []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice          interface{}              `json:"tool_choice,omitempty"`
}

// returns MaxCompletionTokens if set, otherwise MaxTokens
func (r *ChatRequest) EffectiveMaxTokens() *int {
	if r.MaxCompletionTokens != nil {
		return r.MaxCompletionTokens
	}
	return r.MaxTokens
}

// converts the OpenAI tools array to a Gemini functionDeclarations entry
func OpenaiToolsToGemini(tools []map[string]interface{}) []map[string]interface{} {
	var declarations []map[string]interface{}
	for _, tool := range tools {
		if tool["type"] != "function" {
			continue
		}
		fn, _ := tool["function"].(map[string]interface{})
		if fn == nil {
			continue
		}
		name, _ := fn["name"].(string)
		if name == "" {
			continue
		}
		decl := map[string]interface{}{"name": name}
		if desc, ok := fn["description"].(string); ok {
			decl["description"] = desc
		}
		if params, ok := fn["parameters"].(map[string]interface{}); ok {
			decl["parameters"] = NormalizeSchema(params)
		}
		declarations = append(declarations, decl)
	}
	if len(declarations) == 0 {
		return nil
	}
	return []map[string]interface{}{{"functionDeclarations": declarations}}
}

// converts an OpenAI tool_choice value to a Gemini toolConfig
func OpenaiToolChoiceToGemini(toolChoice interface{}) map[string]interface{} {
	if toolChoice == nil {
		return map[string]interface{}{"functionCallingConfig": map[string]interface{}{"mode": "AUTO"}}
	}
	switch tc := toolChoice.(type) {
	case string:
		switch tc {
		case "none":
			return map[string]interface{}{"functionCallingConfig": map[string]interface{}{"mode": "NONE"}}
		case "required":
			return map[string]interface{}{"functionCallingConfig": map[string]interface{}{"mode": "ANY"}}
		default: // "auto" or anything else
			return map[string]interface{}{"functionCallingConfig": map[string]interface{}{"mode": "AUTO"}}
		}
	case map[string]interface{}:
		if tc["type"] == "function" {
			cfg := map[string]interface{}{"mode": "ANY"}
			if fn, ok := tc["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok && name != "" {
					cfg["allowedFunctionNames"] = []string{name}
				}
			}
			return map[string]interface{}{"functionCallingConfig": cfg}
		}
	}
	return map[string]interface{}{"functionCallingConfig": map[string]interface{}{"mode": "AUTO"}}
}

// matches Markdown image syntax: ![alt](url)
var markdownImageRe = regexp.MustCompile(`!\[[^\]]*\]\(([^)]+)\)`)

// converts a text string into Gemini parts, splitting out any embedded data-URI images (![alt](data:image/...;base64,...))
func extractPartsFromText(text string) []map[string]interface{} {
	matches := markdownImageRe.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return []map[string]interface{}{{"text": text}}
	}

	var parts []map[string]interface{}
	lastIdx := 0
	for _, m := range matches {
		// full match: m[0]..m[1]; capture group 1: m[2]..m[3]
		before := text[lastIdx:m[0]]
		if before != "" {
			parts = append(parts, map[string]interface{}{"text": before})
		}

		rawURL := strings.TrimSpace(text[m[2]:m[3]])
		rawURL = strings.Trim(rawURL, `"'`)

		if strings.HasPrefix(rawURL, "data:") {
			// data:image/png;base64,xxxx
			headerAndData := strings.SplitN(rawURL, ",", 2)
			if len(headerAndData) == 2 {
				header := headerAndData[0] // "data:image/png;base64"
				b64 := headerAndData[1]
				mime := ""
				if colon := strings.Index(header, ":"); colon >= 0 {
					afterColon := header[colon+1:]
					mime = strings.SplitN(afterColon, ";", 2)[0]
				}
				if strings.HasPrefix(mime, "image/") {
					parts = append(parts, map[string]interface{}{
						"inlineData": map[string]interface{}{
							"mimeType": mime,
							"data":     b64,
						},
					})
				} else {
					// non-image data URI: keep as text
					parts = append(parts, map[string]interface{}{"text": text[m[0]:m[1]]})
				}
			} else {
				parts = append(parts, map[string]interface{}{"text": text[m[0]:m[1]]})
			}
		} else {
			// external URL: cannot inline without fetching – keep as markdown text
			parts = append(parts, map[string]interface{}{"text": text[m[0]:m[1]]})
		}
		lastIdx = m[1]
	}
	if lastIdx < len(text) {
		parts = append(parts, map[string]interface{}{"text": text[lastIdx:]})
	}
	return parts
}

// extracts a plain string from either a string or list-of-parts content
func messageContentText(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var sb strings.Builder
		for _, p := range v {
			if pm, ok := p.(map[string]interface{}); ok {
				if pm["type"] == "text" {
					if t, ok := pm["text"].(string); ok {
						sb.WriteString(t)
					}
				}
			}
		}
		return sb.String()
	}
	return ""
}

// transforms a ChatRequest into a Gemini API request map
func OpenAIRequestToGemini(req *ChatRequest) map[string]interface{} {
	var contents []map[string]interface{}
	var systemParts []map[string]interface{}
	var pendingToolResponses []map[string]interface{}

	// pre-scan: build tool_call_id -> function_name map from all assistant messages
	// OpenAI tool result messages identify the call via tool_call_id, not by name,
	// but Gemini requires the function name in functionResponse.name
	toolCallIDToName := make(map[string]string)
	for _, msg := range req.Messages {
		if msg.Role == "assistant" {
			for _, tc := range msg.ToolCalls {
				id, _ := tc["id"].(string)
				fn, _ := tc["function"].(map[string]interface{})
				if id != "" && fn != nil {
					if name, ok := fn["name"].(string); ok && name != "" {
						toolCallIDToName[id] = name
					}
				}
			}
		}
	}

	// lastToolCallNames holds the ordered function names from the most recent
	// assistant message with tool_calls. Used as a positional fallback when
	// tool_call_id lookup fails (e.g. some clients omit tool_call_id or name)
	var lastToolCallNames []string
	var pendingToolResponseIdx int

	for _, msg := range req.Messages {
		role := msg.Role

		if role == "system" {
			text := messageContentText(msg.Content)
			if text != "" {
				systemParts = append(systemParts, map[string]interface{}{"text": text})
			}
			continue
		}

		if role == "tool" {
			// resolution order:
			// 1. Explicit name field in the tool message
			// 2. Lookup tool_call_id in the pre-built map
			// 3. Positional match against the preceding assistant's tool_calls
			funcName := ""
			if msg.Name != nil && *msg.Name != "" {
				funcName = *msg.Name
			}
			if funcName == "" && msg.ToolCallID != nil && *msg.ToolCallID != "" {
				funcName = toolCallIDToName[*msg.ToolCallID]
			}
			if funcName == "" && pendingToolResponseIdx < len(lastToolCallNames) {
				funcName = lastToolCallNames[pendingToolResponseIdx]
			}
			pendingToolResponseIdx++

			resultContent := messageContentText(msg.Content)
			pendingToolResponses = append(pendingToolResponses, map[string]interface{}{
				"functionResponse": map[string]interface{}{
					"name": funcName,
					"response": map[string]interface{}{
						"result": resultContent,
					},
				},
			})
			continue
		}

		// flush accumulated tool responses before any non-tool message
		if len(pendingToolResponses) > 0 {
			contents = append(contents, map[string]interface{}{
				"role":  "user",
				"parts": pendingToolResponses,
			})
			pendingToolResponses = nil
			pendingToolResponseIdx = 0
		}

		// map OpenAI role to Gemini role
		geminiRole := role
		if role == "assistant" {
			geminiRole = "model"
		}

		if len(msg.ToolCalls) > 0 {
			// record ordered names for positional fallback when resolving tool results
			lastToolCallNames = lastToolCallNames[:0]
			pendingToolResponseIdx = 0
			var parts []map[string]interface{}
			if text := messageContentText(msg.Content); text != "" {
				parts = append(parts, map[string]interface{}{"text": text})
			}
			for _, tc := range msg.ToolCalls {
				fn, _ := tc["function"].(map[string]interface{})
				if fn == nil {
					fn = map[string]interface{}{}
				}
				name, _ := fn["name"].(string)
				lastToolCallNames = append(lastToolCallNames, name)

				args := fn["arguments"]
				var argsMap map[string]interface{}
				switch a := args.(type) {
				case string:
					json.Unmarshal([]byte(a), &argsMap) //nolint:errcheck
				case map[string]interface{}:
					argsMap = a
				}
				if argsMap == nil {
					argsMap = map[string]interface{}{}
				}
				parts = append(parts, map[string]interface{}{
					"functionCall": map[string]interface{}{
						"name": name,
						"args": argsMap,
					},
				})
			}
			if len(parts) > 0 {
				contents = append(contents, map[string]interface{}{"role": geminiRole, "parts": parts})
			}
			continue
		}

		var parts []map[string]interface{}

		switch c := msg.Content.(type) {
		case string:
			// simple text with possible Markdown images
			parts = extractPartsFromText(c)

		case []interface{}:
			// list of content parts (text, image_url, etc.)
			for _, p := range c {
				pm, ok := p.(map[string]interface{})
				if !ok {
					continue
				}
				switch pm["type"] {
				case "text":
					text, _ := pm["text"].(string)
					parts = append(parts, extractPartsFromText(text)...)

				case "image_url":
					imageURL, _ := pm["image_url"].(map[string]interface{})
					if imageURL == nil {
						continue
					}
					rawURL, _ := imageURL["url"].(string)
					if rawURL == "" {
						continue
					}
					// parse data URI: "data:image/jpeg;base64,{base64}"
					headerAndData := strings.SplitN(rawURL, ",", 2)
					if len(headerAndData) == 2 {
						headerPart := headerAndData[0] // "data:image/jpeg;base64"
						b64 := headerAndData[1]
						mimePart := strings.SplitN(headerPart, ":", 2)
						if len(mimePart) == 2 {
							mime := strings.SplitN(mimePart[1], ";", 2)[0]
							parts = append(parts, map[string]interface{}{
								"inlineData": map[string]interface{}{
									"mimeType": mime,
									"data":     b64,
								},
							})
						}
					}
				}
			}
		}

		if len(parts) == 0 {
			parts = []map[string]interface{}{{"text": ""}}
		}
		contents = append(contents, map[string]interface{}{"role": geminiRole, "parts": parts})
	}

	// flush any trailing tool responses
	if len(pendingToolResponses) > 0 {
		contents = append(contents, map[string]interface{}{
			"role":  "user",
			"parts": pendingToolResponses,
		})
	}

	// post-processing: fill any functionResponse with empty name by looking at
	// the preceding model message's functionCall parts (matched positionally)
	// this handles edge cases where name could not be resolved during the main
	// loop (e.g. session repair stripped the assistant tool_calls message, or
	// tool_call_id mapping failed)
	for i, content := range contents {
		if content["role"] != "user" {
			continue
		}
		parts, _ := content["parts"].([]map[string]interface{})
		// check whether any functionResponse has an empty name
		needsFix := false
		for _, part := range parts {
			if fr, ok := part["functionResponse"].(map[string]interface{}); ok {
				if n, _ := fr["name"].(string); n == "" {
					needsFix = true
					break
				}
			}
		}
		if !needsFix {
			continue
		}
		// collect functionCall names from the closest preceding model message
		var callNames []string
		for j := i - 1; j >= 0; j-- {
			if contents[j]["role"] != "model" {
				continue
			}
			modelParts, _ := contents[j]["parts"].([]map[string]interface{})
			for _, mp := range modelParts {
				if fc, ok := mp["functionCall"].(map[string]interface{}); ok {
					name, _ := fc["name"].(string)
					callNames = append(callNames, name)
				}
			}
			if len(callNames) > 0 {
				break
			}
		}
		// apply positional fix
		callIdx := 0
		for _, part := range parts {
			if fr, ok := part["functionResponse"].(map[string]interface{}); ok {
				if n, _ := fr["name"].(string); n == "" && callIdx < len(callNames) {
					fr["name"] = callNames[callIdx]
				}
				callIdx++
			}
		}
	}

	genCfg := map[string]interface{}{}
	if req.Temperature != nil {
		genCfg["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		genCfg["topP"] = *req.TopP
	}
	if mt := req.EffectiveMaxTokens(); mt != nil {
		genCfg["maxOutputTokens"] = *mt
	}
	if req.Stop != nil {
		switch s := req.Stop.(type) {
		case string:
			genCfg["stopSequences"] = []string{s}
		case []interface{}:
			var seqs []string
			for _, sv := range s {
				if ss, ok := sv.(string); ok {
					seqs = append(seqs, ss)
				}
			}
			genCfg["stopSequences"] = seqs
		}
	}
	if req.FrequencyPenalty != nil {
		genCfg["frequencyPenalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		genCfg["presencePenalty"] = *req.PresencePenalty
	}
	if req.N != nil {
		genCfg["candidateCount"] = *req.N
	}
	if req.Seed != nil {
		genCfg["seed"] = *req.Seed
	}
	if req.ResponseFormat != nil {
		if req.ResponseFormat["type"] == "json_object" {
			genCfg["responseMimeType"] = "application/json"
		}
	}

	// thinking config based on reasoning_effort
	if req.ReasoningEffort != nil {
		var thinkingBudget int
		switch *req.ReasoningEffort {
		case "minimal":
			if strings.Contains(req.Model, "flash") {
				thinkingBudget = 0
			} else {
				thinkingBudget = 128
			}
		case "low":
			thinkingBudget = 1000
		case "medium":
			thinkingBudget = -1
		case "high":
			switch {
			case strings.Contains(req.Model, "gemini-2.5-flash"):
				thinkingBudget = 24576
			case strings.Contains(req.Model, "gemini-2.5-pro"):
				thinkingBudget = 32768
			case strings.Contains(req.Model, "gemini-3"):
				thinkingBudget = 45000
			default:
				thinkingBudget = 32768
			}
		default:
			thinkingBudget = -1
		}
		genCfg["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget":  thinkingBudget,
			"includeThoughts": true,
		}
	}

	payload := map[string]interface{}{
		"contents":         contents,
		"generationConfig": genCfg,
		"safetySettings":   config.DefaultSafetySettings,
		"model":            req.Model,
	}

	if len(systemParts) > 0 {
		payload["systemInstruction"] = map[string]interface{}{
			"role":  "user",
			"parts": systemParts,
		}
	}

	var toolsList []map[string]interface{}
	var hasFunctionTools bool
	for _, tool := range req.Tools {
		switch tool["type"] {
		case "web_search_preview":
			toolsList = append(toolsList, map[string]interface{}{"googleSearch": map[string]interface{}{}})
		case "function":
			hasFunctionTools = true
		}
	}
	if hasFunctionTools {
		geminiTools := OpenaiToolsToGemini(req.Tools)
		toolsList = append(toolsList, geminiTools...)
		payload["toolConfig"] = OpenaiToolChoiceToGemini(req.ToolChoice)
	}
	if len(toolsList) > 0 {
		payload["tools"] = toolsList
	}

	return payload
}

// wraps an inner Gemini request in the Code Assist envelope
func BuildGeminiPayload(innerRequest map[string]any, model string, projectID string) map[string]any {
	modelName := strings.TrimPrefix(model, "models/")
	return map[string]any{
		"model":   modelName,
		"project": projectID,
		"request": innerRequest,
	}
}

// takes a native Gemini API request body and wraps it in the Code Assist envelope
func BuildGeminiPayloadFromNative(nativeReq map[string]any, modelFromPath string) map[string]any {
	// inject safety settings
	nativeReq["safetySettings"] = config.DefaultSafetySettings

	// ensure generationConfig exists
	if _, ok := nativeReq["generationConfig"]; !ok {
		nativeReq["generationConfig"] = map[string]any{}
	}

	genCfg, _ := nativeReq["generationConfig"].(map[string]any)
	if genCfg == nil {
		genCfg = map[string]any{}
		nativeReq["generationConfig"] = genCfg
	}

	// ensure thinkingConfig exists with includeThoughts enabled
	if _, ok := genCfg["thinkingConfig"]; !ok {
		genCfg["thinkingConfig"] = map[string]any{
			"includeThoughts": true,
		}
	}

	projectID := config.Cfg.GoogleCloudProject

	return map[string]any{
		"model":   modelFromPath,
		"project": projectID,
		"request": nativeReq,
	}
}
