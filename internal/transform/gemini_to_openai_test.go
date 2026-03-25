package transform

import (
	"encoding/json"
	"strings"
	"testing"
)

func geminiCandidate(finishReason string, parts []any) map[string]any {
	return map[string]any{
		"content": map[string]any{
			"role":  "model",
			"parts": parts,
		},
		"finishReason": finishReason,
		"index":        float64(0),
	}
}

func geminiResponse(candidates []any, usage map[string]any) map[string]any {
	r := map[string]any{
		"candidates": candidates,
	}
	if usage != nil {
		r["usageMetadata"] = usage
	}
	return r
}

func getGeminiChoice(t *testing.T, resp map[string]any) map[string]any {
	t.Helper()
	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("no choices in response")
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		t.Fatal("choice[0] is not a map")
	}
	return choice
}

func getGeminiMessage(t *testing.T, resp map[string]any) map[string]any {
	t.Helper()
	choice := getGeminiChoice(t, resp)
	msg, ok := choice["message"].(map[string]any)
	if !ok {
		t.Fatalf("message is not a map, got %T", choice["message"])
	}
	return msg
}

func TestMapFinishReason_STOP(t *testing.T) {
	r := mapFinishReason("STOP")
	if r == nil || *r != "stop" {
		t.Errorf("expected 'stop', got %v", r)
	}
}

func TestMapFinishReason_MAX_TOKENS(t *testing.T) {
	r := mapFinishReason("MAX_TOKENS")
	if r == nil || *r != "length" {
		t.Errorf("expected 'length', got %v", r)
	}
}

func TestMapFinishReason_SAFETY(t *testing.T) {
	r := mapFinishReason("SAFETY")
	if r == nil || *r != "content_filter" {
		t.Errorf("expected 'content_filter', got %v", r)
	}
}

func TestMapFinishReason_RECITATION(t *testing.T) {
	r := mapFinishReason("RECITATION")
	if r == nil || *r != "content_filter" {
		t.Errorf("expected 'content_filter', got %v", r)
	}
}

func TestMapFinishReason_Empty(t *testing.T) {
	r := mapFinishReason("")
	if r != nil {
		t.Errorf("expected nil for empty string, got %v", *r)
	}
}

func TestMapFinishReason_Unknown(t *testing.T) {
	r := mapFinishReason("SOMETHING_ELSE")
	if r == nil || *r != "stop" {
		t.Errorf("expected 'stop' for unknown reason, got %v", r)
	}
}

func TestGeminiResponseToOpenAI_BasicText(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{"text": "Hello, world!"},
		}),
	}, map[string]any{
		"promptTokenCount":     float64(10),
		"candidatesTokenCount": float64(5),
		"totalTokenCount":      float64(15),
	}), "gemini-pro")

	if resp["object"] != "chat.completion" {
		t.Errorf("expected object=chat.completion, got %v", resp["object"])
	}
	if resp["model"] != "gemini-pro" {
		t.Errorf("expected model=gemini-pro, got %v", resp["model"])
	}
	if _, ok := resp["id"].(string); !ok {
		t.Error("id should be a string")
	}
	if _, ok := resp["created"].(int64); !ok {
		t.Errorf("created should be int64, got %T", resp["created"])
	}

	msg := getGeminiMessage(t, resp)
	if msg["role"] != "assistant" {
		t.Errorf("expected role=assistant, got %v", msg["role"])
	}
	if msg["content"] != "Hello, world!" {
		t.Errorf("expected content='Hello, world!', got %v", msg["content"])
	}
	if _, ok := msg["reasoning_content"]; ok {
		t.Error("reasoning_content should not be present for plain text")
	}
	if _, ok := msg["tool_calls"]; ok {
		t.Error("tool_calls should not be present for plain text")
	}

	choice := getGeminiChoice(t, resp)
	fr := choice["finish_reason"].(*string)
	if *fr != "stop" {
		t.Errorf("expected finish_reason=stop, got %v", *fr)
	}

	usage, ok := resp["usage"].(map[string]any)
	if !ok {
		t.Fatal("usage should be a map")
	}
	if usage["prompt_tokens"] != 10 {
		t.Errorf("expected prompt_tokens=10, got %v", usage["prompt_tokens"])
	}
	if usage["completion_tokens"] != 5 {
		t.Errorf("expected completion_tokens=5, got %v", usage["completion_tokens"])
	}
	if usage["total_tokens"] != 15 {
		t.Errorf("expected total_tokens=15, got %v", usage["total_tokens"])
	}
}

func TestGeminiResponseToOpenAI_MultipleTextParts(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{"text": "Part one"},
			map[string]any{"text": "Part two"},
		}),
	}, nil), "gemini-pro")

	msg := getGeminiMessage(t, resp)
	if msg["content"] != "Part one\n\nPart two" {
		t.Errorf("expected joined content, got %q", msg["content"])
	}
}

func TestGeminiResponseToOpenAI_ThoughtPart(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{"text": "thinking...", "thought": true},
			map[string]any{"text": "The answer is 42."},
		}),
	}, nil), "gemini-pro")

	msg := getGeminiMessage(t, resp)
	if msg["content"] != "The answer is 42." {
		t.Errorf("expected content='The answer is 42.', got %q", msg["content"])
	}
	if msg["reasoning_content"] != "thinking..." {
		t.Errorf("expected reasoning_content='thinking...', got %q", msg["reasoning_content"])
	}
}

func TestGeminiResponseToOpenAI_FunctionCall(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{
				"functionCall": map[string]any{
					"name": "get_weather",
					"args": map[string]any{"location": "London"},
				},
			},
		}),
	}, nil), "gemini-pro")

	msg := getGeminiMessage(t, resp)
	toolCalls, ok := msg["tool_calls"].([]map[string]any)
	if !ok || len(toolCalls) == 0 {
		t.Fatal("expected tool_calls in message")
	}
	tc := toolCalls[0]
	if tc["type"] != "function" {
		t.Errorf("expected type=function, got %v", tc["type"])
	}
	id, _ := tc["id"].(string)
	if !strings.HasPrefix(id, "call_") {
		t.Errorf("expected id to start with call_, got %q", id)
	}
	fn, ok := tc["function"].(map[string]any)
	if !ok {
		t.Fatal("function field missing")
	}
	if fn["name"] != "get_weather" {
		t.Errorf("expected name=get_weather, got %v", fn["name"])
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %v", err)
	}
	if args["location"] != "London" {
		t.Errorf("expected location=London in args, got %v", args["location"])
	}

	choice := getGeminiChoice(t, resp)
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("expected finish_reason=tool_calls, got %v", choice["finish_reason"])
	}
}

func TestGeminiResponseToOpenAI_InlineImage(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{
				"inlineData": map[string]any{
					"mimeType": "image/png",
					"data":     "abc123==",
				},
			},
		}),
	}, nil), "gemini-pro")

	msg := getGeminiMessage(t, resp)
	expected := "![image](data:image/png;base64,abc123==)"
	if msg["content"] != expected {
		t.Errorf("expected markdown image, got %q", msg["content"])
	}
}

func TestGeminiResponseToOpenAI_InlineImageNonImage(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{
				"inlineData": map[string]any{
					"mimeType": "application/pdf",
					"data":     "abc123==",
				},
			},
		}),
	}, nil), "gemini-pro")

	msg := getGeminiMessage(t, resp)
	if msg["content"] != "" {
		t.Errorf("expected empty content for non-image inline data, got %q", msg["content"])
	}
}

func TestGeminiResponseToOpenAI_FinishReasonMAX_TOKENS(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("MAX_TOKENS", []any{
			map[string]any{"text": "truncated"},
		}),
	}, nil), "gemini-pro")

	choice := getGeminiChoice(t, resp)
	fr := choice["finish_reason"].(*string)
	if *fr != "length" {
		t.Errorf("expected finish_reason=length, got %v", *fr)
	}
}

func TestGeminiResponseToOpenAI_FinishReasonSAFETY(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("SAFETY", []any{
			map[string]any{"text": ""},
		}),
	}, nil), "gemini-pro")

	choice := getGeminiChoice(t, resp)
	fr := choice["finish_reason"].(*string)
	if *fr != "content_filter" {
		t.Errorf("expected finish_reason=content_filter, got %v", *fr)
	}
}

func TestGeminiResponseToOpenAI_NoUsageMetadata(t *testing.T) {
	resp := GeminiResponseToOpenAI(geminiResponse([]any{
		geminiCandidate("STOP", []any{map[string]any{"text": "hi"}}),
	}, nil), "gemini-pro")

	usage, ok := resp["usage"].(map[string]any)
	if !ok {
		t.Fatal("usage map missing")
	}
	if usage["prompt_tokens"] != 0 || usage["completion_tokens"] != 0 || usage["total_tokens"] != 0 {
		t.Errorf("expected zero usage, got %v", usage)
	}
}

func TestGeminiStreamChunkToOpenAI_TextDelta(t *testing.T) {
	chunk := geminiResponse([]any{
		geminiCandidate("", []any{
			map[string]any{"text": "Hello"},
		}),
	}, nil)

	resp := GeminiStreamChunkToOpenAI(chunk, "gemini-pro", "chatcmpl-test-id")

	if resp["object"] != "chat.completion.chunk" {
		t.Errorf("expected object=chat.completion.chunk, got %v", resp["object"])
	}
	if resp["id"] != "chatcmpl-test-id" {
		t.Errorf("expected id=chatcmpl-test-id, got %v", resp["id"])
	}
	if resp["model"] != "gemini-pro" {
		t.Errorf("expected model=gemini-pro, got %v", resp["model"])
	}

	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		t.Fatal("no choices in chunk")
	}
	choice := choices[0].(map[string]any)
	delta, ok := choice["delta"].(map[string]any)
	if !ok {
		t.Fatal("delta not a map")
	}
	if delta["content"] != "Hello" {
		t.Errorf("expected delta.content=Hello, got %v", delta["content"])
	}
	fr := choice["finish_reason"]
	if fr != nil {
		if p, ok := fr.(*string); !ok || p != nil {
			t.Errorf("expected finish_reason=nil when no finishReason, got %v", fr)
		}
	}
}

func TestGeminiStreamChunkToOpenAI_ThoughtDelta(t *testing.T) {
	chunk := geminiResponse([]any{
		geminiCandidate("", []any{
			map[string]any{"text": "I'm thinking", "thought": true},
		}),
	}, nil)

	resp := GeminiStreamChunkToOpenAI(chunk, "gemini-pro", "chatcmpl-x")

	choices, _ := resp["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	if delta["reasoning_content"] != "I'm thinking" {
		t.Errorf("expected reasoning_content in delta, got %v", delta["reasoning_content"])
	}
	if _, ok := delta["content"]; ok {
		t.Error("content should not be present for thought-only chunk")
	}
}

func TestGeminiStreamChunkToOpenAI_FunctionCallDelta(t *testing.T) {
	chunk := geminiResponse([]any{
		geminiCandidate("", []any{
			map[string]any{
				"functionCall": map[string]any{
					"name": "search",
					"args": map[string]any{"query": "Go lang"},
				},
			},
		}),
	}, nil)

	resp := GeminiStreamChunkToOpenAI(chunk, "gemini-pro", "chatcmpl-x")

	choices, _ := resp["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	toolCalls, ok := delta["tool_calls"].([]map[string]any)
	if !ok || len(toolCalls) == 0 {
		t.Fatal("expected tool_calls in streaming delta")
	}
	tc := toolCalls[0]
	if _, hasIndex := tc["index"]; !hasIndex {
		t.Error("streaming tool_call entry should have 'index' field")
	}
	if tc["type"] != "function" {
		t.Errorf("expected type=function, got %v", tc["type"])
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "search" {
		t.Errorf("expected name=search, got %v", fn["name"])
	}
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("expected finish_reason=tool_calls, got %v", choice["finish_reason"])
	}
}

func TestGeminiStreamChunkToOpenAI_FinalChunk(t *testing.T) {
	chunk := geminiResponse([]any{
		geminiCandidate("STOP", []any{}),
	}, nil)

	resp := GeminiStreamChunkToOpenAI(chunk, "gemini-pro", "chatcmpl-x")

	choices, _ := resp["choices"].([]any)
	choice := choices[0].(map[string]any)
	fr, ok := choice["finish_reason"].(*string)
	if !ok || fr == nil || *fr != "stop" {
		t.Errorf("expected finish_reason=stop on final chunk, got %v", choice["finish_reason"])
	}
}

func TestGeminiStreamChunkToOpenAI_ImageDelta(t *testing.T) {
	chunk := geminiResponse([]any{
		geminiCandidate("STOP", []any{
			map[string]any{
				"inlineData": map[string]any{
					"mimeType": "image/jpeg",
					"data":     "xyz789",
				},
			},
		}),
	}, nil)

	resp := GeminiStreamChunkToOpenAI(chunk, "gemini-pro", "chatcmpl-x")

	choices, _ := resp["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	expected := "![image](data:image/jpeg;base64,xyz789)"
	if delta["content"] != expected {
		t.Errorf("expected image markdown in delta, got %q", delta["content"])
	}
}
