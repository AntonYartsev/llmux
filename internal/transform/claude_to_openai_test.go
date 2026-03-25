package transform

import (
	"encoding/json"
	"testing"
)

func claudeResp(stopReason string, blocks []any, usage map[string]any) map[string]any {
	r := map[string]any{
		"id":          "msg_test",
		"type":        "message",
		"role":        "assistant",
		"content":     blocks,
		"stop_reason": stopReason,
	}
	if usage != nil {
		r["usage"] = usage
	}
	return r
}

func getChoice(t *testing.T, resp map[string]any) map[string]any {
	t.Helper()
	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("no choices in response")
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		t.Fatal("choice is not a map")
	}
	return choice
}

func getMessage(t *testing.T, resp map[string]any) map[string]any {
	t.Helper()
	choice := getChoice(t, resp)
	msg, ok := choice["message"].(map[string]any)
	if !ok {
		t.Fatal("message is not a map")
	}
	return msg
}

func TestClaudeResponseToOpenAI_TextBlock(t *testing.T) {
	resp := ClaudeResponseToOpenAI(claudeResp("end_turn", []any{
		map[string]any{"type": "text", "text": "Hello, world!"},
	}, map[string]any{"input_tokens": float64(10), "output_tokens": float64(5)}), "claude-3-opus")

	if resp["object"] != "chat.completion" {
		t.Errorf("expected object=chat.completion, got %v", resp["object"])
	}
	if resp["model"] != "claude-3-opus" {
		t.Errorf("expected model=claude-3-opus, got %v", resp["model"])
	}

	msg := getMessage(t, resp)
	if msg["content"] != "Hello, world!" {
		t.Errorf("expected content='Hello, world!', got %v", msg["content"])
	}
	if _, ok := msg["reasoning_content"]; ok {
		t.Error("reasoning_content should not be present for pure text response")
	}
	if _, ok := msg["tool_calls"]; ok {
		t.Error("tool_calls should not be present for pure text response")
	}

	choice := getChoice(t, resp)
	if choice["finish_reason"] != "stop" {
		t.Errorf("expected finish_reason=stop, got %v", choice["finish_reason"])
	}

	usage, ok := resp["usage"].(map[string]any)
	if !ok {
		t.Fatal("usage missing")
	}
	if toInt(usage["prompt_tokens"]) != 10 {
		t.Errorf("expected prompt_tokens=10, got %v", usage["prompt_tokens"])
	}
	if toInt(usage["completion_tokens"]) != 5 {
		t.Errorf("expected completion_tokens=5, got %v", usage["completion_tokens"])
	}
	if toInt(usage["total_tokens"]) != 15 {
		t.Errorf("expected total_tokens=15, got %v", usage["total_tokens"])
	}
}

func TestClaudeResponseToOpenAI_ThinkingBlock(t *testing.T) {
	resp := ClaudeResponseToOpenAI(claudeResp("end_turn", []any{
		map[string]any{"type": "thinking", "thinking": "Let me think..."},
		map[string]any{"type": "text", "text": "The answer is 42."},
	}, nil), "claude-3-5-sonnet")

	msg := getMessage(t, resp)
	if msg["content"] != "The answer is 42." {
		t.Errorf("unexpected content: %v", msg["content"])
	}
	if msg["reasoning_content"] != "Let me think..." {
		t.Errorf("unexpected reasoning_content: %v", msg["reasoning_content"])
	}
}

func TestClaudeResponseToOpenAI_ToolUse(t *testing.T) {
	input := map[string]any{"city": "Paris", "units": "celsius"}
	resp := ClaudeResponseToOpenAI(claudeResp("tool_use", []any{
		map[string]any{
			"type":  "tool_use",
			"id":    "toolu_01",
			"name":  "get_weather",
			"input": input,
		},
	}, nil), "claude-3-haiku")

	choice := getChoice(t, resp)
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("expected finish_reason=tool_calls, got %v", choice["finish_reason"])
	}

	msg := getMessage(t, resp)
	tcs, ok := msg["tool_calls"].([]map[string]any)
	if !ok || len(tcs) == 0 {
		t.Fatal("tool_calls missing or empty")
	}
	tc := tcs[0]
	if tc["id"] != "toolu_01" {
		t.Errorf("expected id=toolu_01, got %v", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("expected type=function, got %v", tc["type"])
	}
	fn, ok := tc["function"].(map[string]any)
	if !ok {
		t.Fatal("function field missing")
	}
	if fn["name"] != "get_weather" {
		t.Errorf("expected name=get_weather, got %v", fn["name"])
	}
	argsStr, _ := fn["arguments"].(string)
	if argsStr == "" {
		t.Fatal("arguments is empty")
	}
	var argsOut map[string]any
	if err := json.Unmarshal([]byte(argsStr), &argsOut); err != nil {
		t.Fatalf("arguments is not valid JSON: %v", err)
	}
	if argsOut["city"] != "Paris" {
		t.Errorf("expected city=Paris in arguments, got %v", argsOut["city"])
	}
}

func TestClaudeResponseToOpenAI_StopReasonMapping(t *testing.T) {
	cases := []struct {
		reason   string
		expected string
	}{
		{"end_turn", "stop"},
		{"max_tokens", "length"},
		{"tool_use", "tool_calls"},
		{"stop_sequence", "stop"},
		{"", "stop"},
	}
	for _, tc := range cases {
		resp := ClaudeResponseToOpenAI(claudeResp(tc.reason, []any{
			map[string]any{"type": "text", "text": "hi"},
		}, nil), "claude-3")
		choice := getChoice(t, resp)
		if choice["finish_reason"] != tc.expected {
			t.Errorf("stop_reason=%q: expected finish_reason=%q, got %v",
				tc.reason, tc.expected, choice["finish_reason"])
		}
	}
}

func TestClaudeStreamEventToOpenAI_MessageStart(t *testing.T) {
	state := &ClaudeStreamState{}
	chunks := ClaudeStreamEventToOpenAI(
		map[string]any{"type": "message_start", "message": map[string]any{}},
		"message_start",
		"claude-3",
		"chatcmpl-test",
		state,
	)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	chunk := chunks[0]
	if chunk["id"] != "chatcmpl-test" {
		t.Errorf("expected id=chatcmpl-test, got %v", chunk["id"])
	}
	if chunk["object"] != "chat.completion.chunk" {
		t.Errorf("expected object=chat.completion.chunk, got %v", chunk["object"])
	}
	choices := chunk["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	if delta["role"] != "assistant" {
		t.Errorf("expected role=assistant in delta, got %v", delta["role"])
	}
}

func TestClaudeStreamEventToOpenAI_TextDelta(t *testing.T) {
	state := &ClaudeStreamState{CurrentBlockType: "text"}
	chunks := ClaudeStreamEventToOpenAI(
		map[string]any{
			"type":  "content_block_delta",
			"index": float64(0),
			"delta": map[string]any{"type": "text_delta", "text": "Hello"},
		},
		"content_block_delta",
		"claude-3",
		"chatcmpl-test",
		state,
	)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	choices := chunks[0]["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	if delta["content"] != "Hello" {
		t.Errorf("expected content=Hello, got %v", delta["content"])
	}
}

func TestClaudeStreamEventToOpenAI_ThinkingDelta(t *testing.T) {
	state := &ClaudeStreamState{CurrentBlockType: "thinking"}
	chunks := ClaudeStreamEventToOpenAI(
		map[string]any{
			"type":  "content_block_delta",
			"index": float64(0),
			"delta": map[string]any{"type": "thinking_delta", "thinking": "hmm"},
		},
		"content_block_delta",
		"claude-3",
		"chatcmpl-test",
		state,
	)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	choices := chunks[0]["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	if delta["reasoning_content"] != "hmm" {
		t.Errorf("expected reasoning_content=hmm, got %v", delta["reasoning_content"])
	}
}

func TestClaudeStreamEventToOpenAI_ToolUseSequence(t *testing.T) {
	state := &ClaudeStreamState{}

	startEvent := map[string]any{
		"type":  "content_block_start",
		"index": float64(0),
		"content_block": map[string]any{
			"type": "tool_use",
			"id":   "toolu_01",
			"name": "search",
		},
	}
	chunks := ClaudeStreamEventToOpenAI(startEvent, "content_block_start", "claude-3", "chatcmpl-test", state)
	if len(chunks) != 1 {
		t.Fatalf("content_block_start: expected 1 chunk, got %d", len(chunks))
	}
	if state.CurrentToolID != "toolu_01" {
		t.Errorf("expected CurrentToolID=toolu_01, got %v", state.CurrentToolID)
	}
	if state.CurrentToolName != "search" {
		t.Errorf("expected CurrentToolName=search, got %v", state.CurrentToolName)
	}

	choices := chunks[0]["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	tcs := delta["tool_calls"].([]any)
	tc := tcs[0].(map[string]any)
	if tc["id"] != "toolu_01" {
		t.Errorf("expected tool_call id=toolu_01, got %v", tc["id"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "search" {
		t.Errorf("expected function name=search, got %v", fn["name"])
	}

	deltaEvent := map[string]any{
		"type":  "content_block_delta",
		"index": float64(0),
		"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"q":"Go"`},
	}
	chunks = ClaudeStreamEventToOpenAI(deltaEvent, "content_block_delta", "claude-3", "chatcmpl-test", state)
	if len(chunks) != 1 {
		t.Fatalf("input_json_delta: expected 1 chunk, got %d", len(chunks))
	}
	choices = chunks[0]["choices"].([]any)
	choice = choices[0].(map[string]any)
	delta = choice["delta"].(map[string]any)
	tcs = delta["tool_calls"].([]any)
	tc = tcs[0].(map[string]any)
	fn = tc["function"].(map[string]any)
	if fn["arguments"] != `{"q":"Go"` {
		t.Errorf("unexpected partial arguments: %v", fn["arguments"])
	}
	if state.InputJSONBuffer != `{"q":"Go"` {
		t.Errorf("unexpected InputJSONBuffer: %v", state.InputJSONBuffer)
	}

	chunks = ClaudeStreamEventToOpenAI(
		map[string]any{"type": "content_block_stop", "index": float64(0)},
		"content_block_stop", "claude-3", "chatcmpl-test", state,
	)
	if len(chunks) != 0 {
		t.Errorf("content_block_stop: expected 0 chunks, got %d", len(chunks))
	}
}

func TestClaudeStreamEventToOpenAI_MessageDelta(t *testing.T) {
	state := &ClaudeStreamState{}
	chunks := ClaudeStreamEventToOpenAI(
		map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn"},
			"usage": map[string]any{"output_tokens": float64(42)},
		},
		"message_delta",
		"claude-3",
		"chatcmpl-test",
		state,
	)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	choices := chunks[0]["choices"].([]any)
	choice := choices[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Errorf("expected finish_reason=stop, got %v", choice["finish_reason"])
	}
	usage, ok := chunks[0]["usage"].(map[string]any)
	if !ok {
		t.Fatal("usage missing from message_delta chunk")
	}
	if toInt(usage["completion_tokens"]) != 42 {
		t.Errorf("expected completion_tokens=42, got %v", usage["completion_tokens"])
	}
}

func TestClaudeStreamEventToOpenAI_MessageStop(t *testing.T) {
	state := &ClaudeStreamState{}
	chunks := ClaudeStreamEventToOpenAI(
		map[string]any{"type": "message_stop"},
		"message_stop", "claude-3", "chatcmpl-test", state,
	)
	if len(chunks) != 0 {
		t.Errorf("message_stop: expected 0 chunks, got %d", len(chunks))
	}
}
