package transform

import (
	"strings"
	"testing"
)

func strPtr(s string) *string { return &s }
func intPtr(i int) *int       { return &i }


func TestClaudeHeaders(t *testing.T) {
	token := "test-token-abc"
	h := ClaudeHeaders(token)

	if got := h["Authorization"]; got != "Bearer "+token {
		t.Errorf("Authorization: got %q, want %q", got, "Bearer "+token)
	}
	if got := h["content-type"]; got != "application/json" {
		t.Errorf("content-type: got %q", got)
	}
	if got := h["anthropic-beta"]; !strings.Contains(got, "claude-code-20250219") {
		t.Errorf("anthropic-beta missing expected value, got %q", got)
	}
	if got := h["user-agent"]; got != "claude-cli/2.1.75" {
		t.Errorf("user-agent: got %q", got)
	}
	if got := h["x-app"]; got != "cli" {
		t.Errorf("x-app: got %q", got)
	}
	if got := h["anthropic-dangerous-direct-browser-access"]; got != "true" {
		t.Errorf("anthropic-dangerous-direct-browser-access: got %q", got)
	}
}

func systemBlocksText(result map[string]any) string {
	blocks, ok := result["system"].([]map[string]any)
	if !ok {
		return ""
	}
	var parts []string
	for _, b := range blocks {
		if t, ok := b["text"].(string); ok {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, "\n\n")
}

func TestOpenAIToClaude_SystemPrefixPrepended(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-opus-4-5",
		Messages: []ChatMessage{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello"},
		},
	}
	result := OpenAIRequestToClaude(req)

	blocks, ok := result["system"].([]map[string]any)
	if !ok {
		t.Fatal("system field missing or not []map[string]any")
	}
	if len(blocks) < 1 {
		t.Fatal("system blocks empty")
	}
	if blocks[0]["text"] != ClaudeSystemPrefix {
		t.Errorf("first system block is not ClaudeSystemPrefix: %q", blocks[0]["text"])
	}
	combined := systemBlocksText(result)
	if !strings.Contains(combined, "You are a helpful assistant.") {
		t.Errorf("system missing original content: %q", combined)
	}
}

func TestOpenAIToClaude_SystemPrefixNotDuplicated(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-opus-4-5",
		Messages: []ChatMessage{
			{Role: "system", Content: ClaudeSystemPrefix + " Extra context."},
			{Role: "user", Content: "Hello"},
		},
	}
	result := OpenAIRequestToClaude(req)

	combined := systemBlocksText(result)
	count := strings.Count(combined, ClaudeSystemPrefix)
	if count < 1 {
		t.Errorf("ClaudeSystemPrefix missing from system: %q", combined)
	}
}

func TestOpenAIToClaude_NoSystemMessages(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-opus-4-5",
		Messages: []ChatMessage{{Role: "user", Content: "Hello"}},
	}
	result := OpenAIRequestToClaude(req)

	blocks, ok := result["system"].([]map[string]any)
	if !ok {
		t.Fatal("system field missing or not []map[string]any")
	}
	if len(blocks) != 1 {
		t.Errorf("expected 1 system block, got %d", len(blocks))
	}
	if blocks[0]["text"] != ClaudeSystemPrefix {
		t.Errorf("expected bare ClaudeSystemPrefix, got %q", blocks[0]["text"])
	}
}

func TestOpenAIToClaude_UserAssistantMessages(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-opus-4-5",
		Messages: []ChatMessage{
			{Role: "user", Content: "What is Go?"},
			{Role: "assistant", Content: "Go is a compiled language."},
		},
	}
	result := OpenAIRequestToClaude(req)

	msgs, ok := result["messages"].([]map[string]any)
	if !ok {
		t.Fatal("messages field missing")
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0]["role"] != "user" {
		t.Errorf("first message role: got %q", msgs[0]["role"])
	}
	if msgs[1]["role"] != "assistant" {
		t.Errorf("second message role: got %q", msgs[1]["role"])
	}
	userContent, _ := msgs[0]["content"].([]any)
	if len(userContent) == 0 {
		t.Error("user content blocks empty")
	}
	block, _ := userContent[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("user content block type: got %v", block["type"])
	}
	if block["text"] != "What is Go?" {
		t.Errorf("user content text: got %v", block["text"])
	}
}

func TestOpenAIToClaude_ToolMessage(t *testing.T) {
	callID := "call_123"
	req := &ChatRequest{
		Model: "claude-opus-4-5",
		Messages: []ChatMessage{
			{Role: "user", Content: "Run the tool"},
			{
				Role:       "tool",
				Content:    "tool output here",
				ToolCallID: &callID,
			},
		},
	}
	result := OpenAIRequestToClaude(req)

	msgs := result["messages"].([]map[string]any)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	toolMsg := msgs[1]
	if toolMsg["role"] != "user" {
		t.Errorf("tool result role: got %q", toolMsg["role"])
	}
	content, _ := toolMsg["content"].([]any)
	if len(content) == 0 {
		t.Fatal("tool result content empty")
	}
	block, _ := content[0].(map[string]any)
	if block["type"] != "tool_result" {
		t.Errorf("block type: got %v", block["type"])
	}
	if block["tool_use_id"] != callID {
		t.Errorf("tool_use_id: got %v", block["tool_use_id"])
	}
	if block["content"] != "tool output here" {
		t.Errorf("tool content: got %v", block["content"])
	}
}

func TestOpenAIToClaude_ImageDataURI(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-opus-4-5",
		Messages: []ChatMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "data:image/jpeg;base64,/9j/4AAQ",
						},
					},
				},
			},
		},
	}
	result := OpenAIRequestToClaude(req)

	msgs := result["messages"].([]map[string]any)
	content := msgs[0]["content"].([]any)
	if len(content) == 0 {
		t.Fatal("content empty")
	}
	block := content[0].(map[string]any)
	if block["type"] != "image" {
		t.Errorf("block type: got %v", block["type"])
	}
	src, _ := block["source"].(map[string]any)
	if src == nil {
		t.Fatal("source missing")
	}
	if src["type"] != "base64" {
		t.Errorf("source type: got %v", src["type"])
	}
	if src["media_type"] != "image/jpeg" {
		t.Errorf("media_type: got %v", src["media_type"])
	}
	if src["data"] != "/9j/4AAQ" {
		t.Errorf("data: got %v", src["data"])
	}
}

func TestOpenAIToClaude_AssistantToolCalls(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-opus-4-5",
		Messages: []ChatMessage{
			{
				Role:    "assistant",
				Content: "",
				ToolCalls: []map[string]interface{}{
					{
						"id":   "call_abc",
						"type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"location":"London"}`,
						},
					},
				},
			},
		},
	}
	result := OpenAIRequestToClaude(req)

	msgs := result["messages"].([]map[string]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	content := msgs[0]["content"].([]any)
	var toolUse map[string]any
	for _, c := range content {
		if b, ok := c.(map[string]any); ok && b["type"] == "tool_use" {
			toolUse = b
			break
		}
	}
	if toolUse == nil {
		t.Fatal("no tool_use block found")
	}
	if toolUse["id"] != "call_abc" {
		t.Errorf("id: got %v", toolUse["id"])
	}
	if toolUse["name"] != "get_weather" {
		t.Errorf("name: got %v", toolUse["name"])
	}
	input, _ := toolUse["input"].(map[string]interface{})
	if input["location"] != "London" {
		t.Errorf("input.location: got %v", input["location"])
	}
}

func TestOpenAIToClaude_MaxTokensDefault(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-opus-4-5",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	}
	result := OpenAIRequestToClaude(req)

	if result["max_tokens"] != 8096 {
		t.Errorf("max_tokens default: got %v, want 8096", result["max_tokens"])
	}
}

func TestOpenAIToClaude_MaxTokensExplicit(t *testing.T) {
	req := &ChatRequest{
		Model:     "claude-opus-4-5",
		Messages:  []ChatMessage{{Role: "user", Content: "hi"}},
		MaxTokens: intPtr(4096),
	}
	result := OpenAIRequestToClaude(req)

	if result["max_tokens"] != 4096 {
		t.Errorf("max_tokens: got %v, want 4096", result["max_tokens"])
	}
}

func TestOpenAIToClaude_ReasoningEffort(t *testing.T) {
	cases := []struct {
		effort     string
		wantEffort string
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		{"max", "max"},
	}
	for _, tc := range cases {
		t.Run(tc.effort, func(t *testing.T) {
			req := &ChatRequest{
				Model:           "claude-opus-4-5",
				Messages:        []ChatMessage{{Role: "user", Content: "think"}},
				ReasoningEffort: strPtr(tc.effort),
			}
			result := OpenAIRequestToClaude(req)

			thinking, ok := result["thinking"].(map[string]any)
			if !ok {
				t.Fatalf("thinking field missing for effort=%q", tc.effort)
			}
			if thinking["type"] != "adaptive" {
				t.Errorf("thinking.type: got %v", thinking["type"])
			}

			outCfg, ok := result["output_config"].(map[string]any)
			if !ok {
				t.Fatalf("output_config field missing for effort=%q", tc.effort)
			}
			if outCfg["effort"] != tc.wantEffort {
				t.Errorf("output_config.effort: got %v, want %v", outCfg["effort"], tc.wantEffort)
			}
		})
	}
}

func TestOpenAIToClaude_NoReasoningEffort(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-opus-4-5",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	}
	result := OpenAIRequestToClaude(req)

	if _, ok := result["thinking"]; ok {
		t.Error("thinking should not be set when ReasoningEffort is nil")
	}
	if _, ok := result["output_config"]; ok {
		t.Error("output_config should not be set when ReasoningEffort is nil")
	}
}

func TestOpenAIToClaude_StopSequences(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-opus-4-5",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
		Stop:     []interface{}{"STOP", "END"},
	}
	result := OpenAIRequestToClaude(req)

	seqs, ok := result["stop_sequences"].([]string)
	if !ok {
		t.Fatal("stop_sequences missing or wrong type")
	}
	if len(seqs) != 2 || seqs[0] != "STOP" || seqs[1] != "END" {
		t.Errorf("stop_sequences: got %v", seqs)
	}
}

func TestOpenAIToClaude_Tools(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-opus-4-5",
		Messages: []ChatMessage{{Role: "user", Content: "use a tool"}},
		Tools: []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "search",
					"description": "Search the web",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"query": map[string]interface{}{"type": "string"},
						},
						"required": []interface{}{"query"},
					},
				},
			},
		},
	}
	result := OpenAIRequestToClaude(req)

	tools, ok := result["tools"].([]map[string]any)
	if !ok || len(tools) == 0 {
		t.Fatal("tools missing")
	}
	tool := tools[0]
	if tool["name"] != "search" {
		t.Errorf("tool name: got %v", tool["name"])
	}
	if tool["description"] != "Search the web" {
		t.Errorf("tool description: got %v", tool["description"])
	}
	schema, _ := tool["input_schema"].(map[string]any)
	if schema == nil {
		t.Fatal("input_schema missing")
	}
	if schema["type"] != "object" {
		t.Errorf("input_schema type: got %v", schema["type"])
	}
}

func TestOpenAIToClaude_WebSearchPreviewIgnored(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []ChatMessage{{Role: "user", Content: "search the web"}},
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
	result := OpenAIRequestToClaude(req)

	tools, ok := result["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool (web_search_preview ignored), got %d", len(tools))
	}
	if tools[0]["name"] != "get_weather" {
		t.Errorf("tool name: got %v, want get_weather", tools[0]["name"])
	}
}

func TestOpenAIToClaude_OnlyWebSearchPreview(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []ChatMessage{{Role: "user", Content: "search"}},
		Tools: []map[string]interface{}{
			{"type": "web_search_preview"},
		},
	}
	result := OpenAIRequestToClaude(req)

	if _, ok := result["tools"]; ok {
		t.Error("tools should not be set when only web_search_preview is present")
	}
}
