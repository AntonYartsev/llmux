package handler

import (
	"testing"
)

func TestResponsesInputToMessages_String(t *testing.T) {
	msgs, _ := responsesInputToMessages("Hello", "Be helpful", nil)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Errorf("first message role: got %q, want system", msgs[0].Role)
	}
	text, _ := msgs[0].Content.(string)
	if text != "Be helpful" {
		t.Errorf("instructions content: got %q", text)
	}
	if msgs[1].Role != "user" {
		t.Errorf("second message role: got %q, want user", msgs[1].Role)
	}
}

func TestResponsesInputToMessages_Array(t *testing.T) {
	input := []any{
		map[string]any{"role": "user", "content": "Hi"},
		map[string]any{"role": "assistant", "content": "Hello!"},
		map[string]any{"role": "user", "content": "How are you?"},
	}
	msgs, _ := responsesInputToMessages(input, "", nil)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("msg[0] role: got %q", msgs[0].Role)
	}
}

func TestResponsesInputToMessages_FunctionCallOutput(t *testing.T) {
	input := []any{
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_abc",
			"output":  `{"temp": 20}`,
		},
	}
	msgs, _ := responsesInputToMessages(input, "", nil)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "tool" {
		t.Errorf("role: got %q, want tool", msgs[0].Role)
	}
	if msgs[0].ToolCallID == nil || *msgs[0].ToolCallID != "call_abc" {
		t.Error("tool_call_id not set")
	}
}

func TestResponsesInputToMessages_StringNoInstructions(t *testing.T) {
	msgs, _ := responsesInputToMessages("Hello", "", nil)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("role: got %q, want user", msgs[0].Role)
	}
}

func TestResponsesToolsToOpenAI_FunctionTool(t *testing.T) {
	tools := []any{
		map[string]any{
			"type":        "function",
			"name":        "get_weather",
			"description": "Get weather",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
			"strict": true,
		},
	}
	result := responsesToolsToOpenAI(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0]["type"] != "function" {
		t.Errorf("type: got %v", result[0]["type"])
	}
	fn, _ := result[0]["function"].(map[string]interface{})
	if fn == nil {
		t.Fatal("function missing")
	}
	if fn["name"] != "get_weather" {
		t.Errorf("name: got %v", fn["name"])
	}
	if fn["strict"] != true {
		t.Errorf("strict: got %v", fn["strict"])
	}
}

func TestResponsesToolsToOpenAI_WebSearchIgnored(t *testing.T) {
	tools := []any{
		map[string]any{"type": "web_search"},
		map[string]any{"type": "function", "name": "foo"},
	}
	result := responsesToolsToOpenAI(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool (web_search dropped), got %d", len(result))
	}
}

func TestResponsesToolsToOpenAI_Empty(t *testing.T) {
	result := responsesToolsToOpenAI([]any{
		map[string]any{"type": "web_search"},
		map[string]any{"type": "file_search"},
	})
	if len(result) != 0 {
		t.Errorf("expected 0 tools, got %d", len(result))
	}
}

func TestBuildResponsesUsage(t *testing.T) {
	chatUsage := map[string]any{
		"prompt_tokens":     100,
		"completion_tokens": 50,
		"total_tokens":      150,
		"prompt_tokens_details": map[string]any{
			"cached_tokens": 10,
		},
		"completion_tokens_details": map[string]any{
			"reasoning_tokens": 5,
		},
	}
	result := buildResponsesUsage(chatUsage)
	if toIntHandler(result["input_tokens"]) != 100 {
		t.Errorf("input_tokens: got %v", result["input_tokens"])
	}
	if toIntHandler(result["output_tokens"]) != 50 {
		t.Errorf("output_tokens: got %v", result["output_tokens"])
	}
	details, _ := result["output_tokens_details"].(map[string]any)
	if details == nil {
		t.Fatal("output_tokens_details missing")
	}
	if toIntHandler(details["reasoning_tokens"]) != 5 {
		t.Errorf("reasoning_tokens: got %v", details["reasoning_tokens"])
	}
}

func TestBuildResponsesUsage_Nil(t *testing.T) {
	result := buildResponsesUsage(nil)
	if toIntHandler(result["input_tokens"]) != 0 {
		t.Errorf("input_tokens: got %v", result["input_tokens"])
	}
	if toIntHandler(result["output_tokens"]) != 0 {
		t.Errorf("output_tokens: got %v", result["output_tokens"])
	}
}

func TestChatToolCallsToOutputItems(t *testing.T) {
	toolCalls := []any{
		map[string]any{
			"id":   "call_123",
			"type": "function",
			"function": map[string]any{
				"name":      "get_weather",
				"arguments": `{"city":"Moscow"}`,
			},
		},
	}
	items := chatToolCallsToOutputItems(toolCalls)
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	item, _ := items[0].(map[string]any)
	if item["type"] != "function_call" {
		t.Errorf("type: got %v", item["type"])
	}
	if item["call_id"] != "call_123" {
		t.Errorf("call_id: got %v", item["call_id"])
	}
	if item["name"] != "get_weather" {
		t.Errorf("name: got %v", item["name"])
	}
	if item["arguments"] != `{"city":"Moscow"}` {
		t.Errorf("arguments: got %v", item["arguments"])
	}
}
