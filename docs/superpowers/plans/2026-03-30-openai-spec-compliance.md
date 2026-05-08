# OpenAI Spec Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make llmux proxy fully compliant with the OpenAI Chat Completions and Responses API specifications defined in `specifications/openai-chat-completions-openapi.yaml` and `specifications/openai-responses-openapi.yaml`.

**Architecture:** Extend the existing `ChatRequest` struct with all missing spec fields. Update transform layers to forward mappable fields to backends and silently drop unmappable ones. Extend response transformers to emit all spec-required fields (usage details, refusal, service_tier). Rewrite the Responses API handler to properly convert between Responses format and internal ChatRequest, including tool calls and usage translation.

**Tech Stack:** Go 1.26, gin-gonic, stdlib testing

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `internal/transform/openai_to_gemini.go` | Modify | Add missing fields to ChatRequest, handle `developer` role, forward mappable params |
| `internal/transform/openai_to_claude.go` | Modify | Handle `developer` role, forward `user` metadata, `parallel_tool_calls` |
| `internal/transform/claude_to_openai.go` | Modify | Add `prompt_tokens_details`, `completion_tokens_details`, `refusal`, `service_tier` to response |
| `internal/transform/gemini_to_openai.go` | Modify | Add `prompt_tokens_details`, `completion_tokens_details`, `service_tier` to response |
| `internal/handler/openai.go` | Modify | Rewrite Responses API handler for full spec compliance |
| `internal/transform/openai_to_gemini_test.go` | Modify | Tests for new fields |
| `internal/transform/openai_to_claude_test.go` | Modify | Tests for new fields |
| `internal/transform/claude_to_openai_test.go` | Modify | Tests for new response fields |
| `internal/transform/gemini_to_openai_test.go` | Modify | Tests for new response fields |
| `internal/handler/openai_test.go` | Create | Tests for Responses API handler |

---

### Task 1: Extend ChatRequest with all spec fields

**Files:**
- Modify: `internal/transform/openai_to_gemini.go:13-46` (ChatRequest struct)

- [ ] **Step 1: Add missing fields to ChatRequest**

```go
// represents an OpenAI chat message
type ChatMessage struct {
	Role             string                   `json:"role"`
	Content          interface{}              `json:"content"`
	ReasoningContent *string                  `json:"reasoning_content,omitempty"`
	Refusal          *string                  `json:"refusal,omitempty"`
	ToolCalls        []map[string]interface{} `json:"tool_calls,omitempty"`
	ToolCallID       *string                  `json:"tool_call_id,omitempty"`
	Name             *string                  `json:"name,omitempty"`
	Audio            map[string]interface{}   `json:"audio,omitempty"`
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
	Stop                interface{}              `json:"stop,omitempty"`
	FrequencyPenalty    *float64                 `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64                 `json:"presence_penalty,omitempty"`
	N                   *int                     `json:"n,omitempty"`
	Seed                *int                     `json:"seed,omitempty"`
	ResponseFormat      map[string]interface{}   `json:"response_format,omitempty"`
	ReasoningEffort     *string                  `json:"reasoning_effort,omitempty"`
	Tools               []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice          interface{}              `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool                    `json:"parallel_tool_calls,omitempty"`
	LogitBias           map[string]float64       `json:"logit_bias,omitempty"`
	Logprobs            *bool                    `json:"logprobs,omitempty"`
	TopLogprobs         *int                     `json:"top_logprobs,omitempty"`
	Store               *bool                    `json:"store,omitempty"`
	Metadata            map[string]string         `json:"metadata,omitempty"`
	User                *string                  `json:"user,omitempty"`
	ServiceTier         *string                  `json:"service_tier,omitempty"`
	Modalities          []string                 `json:"modalities,omitempty"`
	Audio               map[string]interface{}   `json:"audio,omitempty"`
	Prediction          map[string]interface{}   `json:"prediction,omitempty"`
	WebSearchOptions    map[string]interface{}   `json:"web_search_options,omitempty"`
}
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/...`
Expected: All existing tests PASS (new fields are all optional with omitempty)

- [ ] **Step 3: Commit**

```bash
git add internal/transform/openai_to_gemini.go
git commit -m "feat: extend ChatRequest/ChatMessage with all OpenAI spec fields"
```

---

### Task 2: Handle `developer` role in transforms

**Files:**
- Modify: `internal/transform/openai_to_claude.go:96-98`
- Modify: `internal/transform/openai_to_gemini.go:228-234`
- Test: `internal/transform/openai_to_claude_test.go`
- Test: `internal/transform/openai_to_gemini_test.go`

- [ ] **Step 1: Write failing test for developer role in Claude transform**

Add to `internal/transform/openai_to_claude_test.go`:

```go
func TestOpenAIToClaude_DeveloperRole(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4-6",
		Messages: []ChatMessage{
			{Role: "developer", Content: "You must always respond in JSON"},
			{Role: "user", Content: "Hello"},
		},
	}
	result := OpenAIRequestToClaude(req)

	// developer messages should be treated as system messages
	systemBlocks := result["system"].([]map[string]any)
	found := false
	for _, block := range systemBlocks {
		if text, _ := block["text"].(string); text == "You must always respond in JSON" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("developer message not found in system blocks: %v", systemBlocks)
	}

	// should NOT appear in messages array
	messages := result["messages"].([]map[string]any)
	for _, msg := range messages {
		if msg["role"] == "developer" {
			t.Error("developer role should not appear in Claude messages")
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIToClaude_DeveloperRole -v`
Expected: FAIL — developer message ends up in messages as "user" instead of system blocks

- [ ] **Step 3: Fix Claude transform to handle developer role**

In `internal/transform/openai_to_claude.go`, in the `OpenAIRequestToClaude` function, add `"developer"` case alongside `"system"`:

```go
		case "system", "developer":
			text := messageContentText(msg.Content)
			if text != "" {
				systemParts = append(systemParts, text)
			}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIToClaude_DeveloperRole -v`
Expected: PASS

- [ ] **Step 5: Write failing test for developer role in Gemini transform**

Add to `internal/transform/openai_to_gemini_test.go`:

```go
func TestOpenAIRequestToGemini_DeveloperRole(t *testing.T) {
	req := &ChatRequest{
		Model: "gemini-2.5-flash",
		Messages: []ChatMessage{
			{Role: "developer", Content: "You must respond in JSON"},
			{Role: "user", Content: "Hello"},
		},
	}
	result := OpenAIRequestToGemini(req)

	// developer messages should go into systemInstruction
	sysInstr, ok := result["systemInstruction"].(map[string]interface{})
	if !ok {
		t.Fatal("systemInstruction missing")
	}
	parts, _ := sysInstr["parts"].([]map[string]interface{})
	if len(parts) == 0 {
		t.Fatal("systemInstruction has no parts")
	}
	found := false
	for _, p := range parts {
		if text, _ := p["text"].(string); text == "You must respond in JSON" {
			found = true
		}
	}
	if !found {
		t.Errorf("developer message not in systemInstruction parts: %v", parts)
	}

	// should NOT appear in contents
	contents, _ := result["contents"].([]map[string]interface{})
	for _, c := range contents {
		if c["role"] == "developer" {
			t.Error("developer role should not appear in Gemini contents")
		}
	}
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIRequestToGemini_DeveloperRole -v`
Expected: FAIL

- [ ] **Step 7: Fix Gemini transform to handle developer role**

In `internal/transform/openai_to_gemini.go`, in the `OpenAIRequestToGemini` function, change the system check:

```go
		if role == "system" || role == "developer" {
			text := messageContentText(msg.Content)
			if text != "" {
				systemParts = append(systemParts, map[string]interface{}{"text": text})
			}
			continue
		}
```

- [ ] **Step 8: Run all transform tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add internal/transform/openai_to_claude.go internal/transform/openai_to_gemini.go internal/transform/openai_to_claude_test.go internal/transform/openai_to_gemini_test.go
git commit -m "feat: handle developer role as system message in both backends"
```

---

### Task 3: Forward mappable request fields to Claude

**Files:**
- Modify: `internal/transform/openai_to_claude.go:89-297`
- Test: `internal/transform/openai_to_claude_test.go`

- [ ] **Step 1: Write failing test for user metadata forwarding**

Add to `internal/transform/openai_to_claude_test.go`:

```go
func TestOpenAIToClaude_UserMetadata(t *testing.T) {
	user := "user-123"
	req := &ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []ChatMessage{{Role: "user", Content: "Hi"}},
		User:     &user,
		Metadata: map[string]string{"request_id": "abc"},
	}
	result := OpenAIRequestToClaude(req)

	meta, ok := result["metadata"].(map[string]any)
	if !ok {
		t.Fatal("metadata missing from Claude request")
	}
	if meta["user_id"] != "user-123" {
		t.Errorf("expected user_id=user-123, got %v", meta["user_id"])
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIToClaude_UserMetadata -v`
Expected: FAIL

- [ ] **Step 3: Add user/metadata forwarding to OpenAIRequestToClaude**

At the end of `OpenAIRequestToClaude`, before `return result`, add:

```go
	if req.User != nil && *req.User != "" {
		meta, _ := result["metadata"].(map[string]any)
		if meta == nil {
			meta = make(map[string]any)
		}
		meta["user_id"] = *req.User
		result["metadata"] = meta
	}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIToClaude_UserMetadata -v`
Expected: PASS

- [ ] **Step 5: Write test for reasoning_effort expanded enum**

Add to `internal/transform/openai_to_claude_test.go`:

```go
func TestOpenAIToClaude_ReasoningEffortExpanded(t *testing.T) {
	tests := []struct {
		input    string
		wantNorm string
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		{"none", "low"},
		{"minimal", "low"},
		{"xhigh", "max"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			effort := tt.input
			req := &ChatRequest{
				Model:           "claude-sonnet-4-6",
				Messages:        []ChatMessage{{Role: "user", Content: "Hi"}},
				ReasoningEffort: &effort,
			}
			result := OpenAIRequestToClaude(req)
			oc, ok := result["output_config"].(map[string]any)
			if !ok {
				t.Fatal("output_config missing")
			}
			if oc["effort"] != tt.wantNorm {
				t.Errorf("effort=%q: got %q, want %q", tt.input, oc["effort"], tt.wantNorm)
			}
		})
	}
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIToClaude_ReasoningEffortExpanded -v`
Expected: FAIL — "none", "minimal", "xhigh" fall through to "medium"

- [ ] **Step 7: Update reasoning_effort normalization in OpenAIRequestToClaude**

Replace the reasoning_effort switch block in `openai_to_claude.go`:

```go
	if req.ReasoningEffort != nil && *req.ReasoningEffort != "" {
		effort := *req.ReasoningEffort
		// normalize OpenAI reasoning_effort values to Claude effort values
		switch effort {
		case "low", "medium", "high", "max":
			// valid as-is
		case "none", "minimal":
			effort = "low"
		case "xhigh":
			effort = "max"
		default:
			effort = "medium"
		}
		result["thinking"] = map[string]any{"type": "adaptive"}
		result["output_config"] = map[string]any{"effort": effort}
	}
```

- [ ] **Step 8: Run all transform tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add internal/transform/openai_to_claude.go internal/transform/openai_to_claude_test.go
git commit -m "feat: forward user metadata and expand reasoning_effort enum for Claude"
```

---

### Task 4: Forward mappable request fields to Gemini

**Files:**
- Modify: `internal/transform/openai_to_gemini.go:196-545`
- Test: `internal/transform/openai_to_gemini_test.go`

- [ ] **Step 1: Write failing test for expanded reasoning_effort in Gemini**

Add to `internal/transform/openai_to_gemini_test.go`:

```go
func TestOpenAIRequestToGemini_ReasoningEffortExpanded(t *testing.T) {
	tests := []struct {
		input      string
		wantBudget int
	}{
		{"none", 0},
		{"xhigh", 32768},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			effort := tt.input
			req := &ChatRequest{
				Model:           "gemini-2.5-pro",
				Messages:        []ChatMessage{{Role: "user", Content: "Hi"}},
				ReasoningEffort: &effort,
			}
			result := OpenAIRequestToGemini(req)
			genCfg, _ := result["generationConfig"].(map[string]interface{})
			thinkCfg, _ := genCfg["thinkingConfig"].(map[string]interface{})
			budget := toInt(thinkCfg["thinkingBudget"])
			if budget != tt.wantBudget {
				t.Errorf("effort=%q: got budget %d, want %d", tt.input, budget, tt.wantBudget)
			}
		})
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestOpenAIRequestToGemini_ReasoningEffortExpanded -v`
Expected: FAIL — "none" and "xhigh" fall to default case (budget=-1)

- [ ] **Step 3: Add none/xhigh cases to Gemini reasoning_effort switch**

In `internal/transform/openai_to_gemini.go`, update the reasoning_effort switch:

```go
	if req.ReasoningEffort != nil {
		var thinkingBudget int
		switch *req.ReasoningEffort {
		case "none":
			thinkingBudget = 0
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
		case "xhigh":
			switch {
			case strings.Contains(req.Model, "gemini-2.5-flash"):
				thinkingBudget = 24576
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
```

- [ ] **Step 4: Run all transform tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add internal/transform/openai_to_gemini.go internal/transform/openai_to_gemini_test.go
git commit -m "feat: expand reasoning_effort enum for Gemini (none, xhigh)"
```

---

### Task 5: Add spec-required fields to Chat Completions response (Claude)

**Files:**
- Modify: `internal/transform/claude_to_openai.go:48-152`
- Test: `internal/transform/claude_to_openai_test.go`

- [ ] **Step 1: Write failing test for usage details in Claude response**

Add to `internal/transform/claude_to_openai_test.go`:

```go
func TestClaudeResponseToOpenAI_UsageDetails(t *testing.T) {
	resp := map[string]any{
		"content":     []any{map[string]any{"type": "text", "text": "Hi"}},
		"stop_reason": "end_turn",
		"usage": map[string]any{
			"input_tokens":              100,
			"output_tokens":             50,
			"cache_creation_input_tokens": 10,
			"cache_read_input_tokens":    20,
		},
	}
	result := ClaudeResponseToOpenAI(resp, "claude-sonnet-4-6")

	usage, _ := result["usage"].(map[string]any)
	ptd, ok := usage["prompt_tokens_details"].(map[string]any)
	if !ok {
		t.Fatal("prompt_tokens_details missing")
	}
	if ptd["cached_tokens"] != 20 {
		t.Errorf("cached_tokens: got %v, want 20", ptd["cached_tokens"])
	}

	ctd, ok := usage["completion_tokens_details"].(map[string]any)
	if !ok {
		t.Fatal("completion_tokens_details missing")
	}
	if _, exists := ctd["reasoning_tokens"]; !exists {
		t.Error("reasoning_tokens missing from completion_tokens_details")
	}
}
```

- [ ] **Step 2: Write failing test for refusal field**

Add to `internal/transform/claude_to_openai_test.go`:

```go
func TestClaudeResponseToOpenAI_Refusal(t *testing.T) {
	resp := map[string]any{
		"content":     []any{},
		"stop_reason": "end_turn",
		"usage":       map[string]any{"input_tokens": 10, "output_tokens": 0},
	}
	result := ClaudeResponseToOpenAI(resp, "claude-sonnet-4-6")

	msg := getMessage(t, result)
	// refusal should be nil when no refusal occurred
	if msg["refusal"] != nil {
		t.Errorf("expected refusal=nil, got %v", msg["refusal"])
	}
}

func TestClaudeResponseToOpenAI_ServiceTier(t *testing.T) {
	resp := map[string]any{
		"content":     []any{map[string]any{"type": "text", "text": "Hi"}},
		"stop_reason": "end_turn",
		"usage":       map[string]any{"input_tokens": 10, "output_tokens": 5},
	}
	result := ClaudeResponseToOpenAI(resp, "claude-sonnet-4-6")

	if _, exists := result["service_tier"]; !exists {
		t.Error("service_tier missing from response")
	}
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run "TestClaudeResponseToOpenAI_(UsageDetails|Refusal|ServiceTier)" -v`
Expected: FAIL

- [ ] **Step 4: Update ClaudeResponseToOpenAI to add spec-required fields**

Replace the usage and return section in `internal/transform/claude_to_openai.go`:

```go
	// build message object
	message := map[string]any{
		"role":    "assistant",
		"refusal": nil,
	}
	// OpenAI returns content: null when only tool_calls are present
	if len(toolCalls) > 0 && content == "" {
		message["content"] = nil
	} else {
		message["content"] = content
	}
	if reasoningContent != "" {
		message["reasoning_content"] = reasoningContent
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	// build usage object with details
	inputTokens := 0
	outputTokens := 0
	cachedTokens := 0
	reasoningTokens := 0
	if usageIn, ok := claudeResp["usage"].(map[string]any); ok {
		inputTokens = toInt(usageIn["input_tokens"])
		outputTokens = toInt(usageIn["output_tokens"])
		cachedTokens = toInt(usageIn["cache_read_input_tokens"])
	}

	usageOut := map[string]any{
		"prompt_tokens":     inputTokens,
		"completion_tokens": outputTokens,
		"total_tokens":      inputTokens + outputTokens,
		"prompt_tokens_details": map[string]any{
			"cached_tokens": cachedTokens,
			"audio_tokens":  0,
		},
		"completion_tokens_details": map[string]any{
			"reasoning_tokens":            reasoningTokens,
			"audio_tokens":               0,
			"accepted_prediction_tokens":  0,
			"rejected_prediction_tokens":  0,
		},
	}

	return map[string]any{
		"id":                 newResponseID(),
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": nil,
		"service_tier":       "default",
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"logprobs":      nil,
				"finish_reason": finishReason,
			},
		},
		"usage": usageOut,
	}
```

- [ ] **Step 5: Run all claude_to_openai tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestClaudeResponseToOpenAI -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add internal/transform/claude_to_openai.go internal/transform/claude_to_openai_test.go
git commit -m "feat: add usage details, refusal, service_tier to Claude response transform"
```

---

### Task 6: Add spec-required fields to Chat Completions response (Gemini)

**Files:**
- Modify: `internal/transform/gemini_to_openai.go:116-198`
- Test: `internal/transform/gemini_to_openai_test.go`

- [ ] **Step 1: Write failing test for Gemini usage details**

Add to `internal/transform/gemini_to_openai_test.go`:

```go
func TestGeminiResponseToOpenAI_UsageDetails(t *testing.T) {
	resp := geminiResponse(
		geminiCandidate([]any{map[string]any{"text": "Hi"}}, "STOP"),
	)
	resp["usageMetadata"] = map[string]any{
		"promptTokenCount":     100,
		"candidatesTokenCount": 50,
		"totalTokenCount":      150,
		"cachedContentTokenCount": 30,
	}

	result := GeminiResponseToOpenAI(resp, "gemini-2.5-pro")

	usage, _ := result["usage"].(map[string]any)
	ptd, ok := usage["prompt_tokens_details"].(map[string]any)
	if !ok {
		t.Fatal("prompt_tokens_details missing")
	}
	if ptd["cached_tokens"] != 30 {
		t.Errorf("cached_tokens: got %v, want 30", ptd["cached_tokens"])
	}

	if _, exists := result["service_tier"]; !exists {
		t.Error("service_tier missing")
	}
}

func TestGeminiResponseToOpenAI_MessageRefusal(t *testing.T) {
	resp := geminiResponse(
		geminiCandidate([]any{map[string]any{"text": "Hi"}}, "STOP"),
	)
	result := GeminiResponseToOpenAI(resp, "gemini-2.5-pro")
	msg := getGeminiMessage(t, result)
	if _, exists := msg["refusal"]; !exists {
		t.Error("refusal field missing from message")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run "TestGeminiResponseToOpenAI_(UsageDetails|MessageRefusal)" -v`
Expected: FAIL

- [ ] **Step 3: Update GeminiResponseToOpenAI with spec-required fields**

In `internal/transform/gemini_to_openai.go`, update the message building and usage/return sections:

```go
		message := map[string]any{
			"role":    "assistant",
			"refusal": nil,
		}
```

And update the usage section:

```go
	// build usage from usageMetadata
	prompt := 0
	completion := 0
	total := 0
	cachedTokens := 0
	if meta, ok := geminiResp["usageMetadata"].(map[string]any); ok {
		prompt = toInt(meta["promptTokenCount"])
		completion = toInt(meta["candidatesTokenCount"])
		total = toInt(meta["totalTokenCount"])
		cachedTokens = toInt(meta["cachedContentTokenCount"])
		if total == 0 {
			total = prompt + completion
		}
	}

	usageOut := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"total_tokens":      total,
		"prompt_tokens_details": map[string]any{
			"cached_tokens": cachedTokens,
			"audio_tokens":  0,
		},
		"completion_tokens_details": map[string]any{
			"reasoning_tokens":            0,
			"audio_tokens":               0,
			"accepted_prediction_tokens":  0,
			"rejected_prediction_tokens":  0,
		},
	}

	return map[string]any{
		"id":                 id,
		"object":             "chat.completion",
		"created":            ts,
		"model":              model,
		"system_fingerprint": nil,
		"service_tier":       "default",
		"choices":            choices,
		"usage":              usageOut,
	}
```

- [ ] **Step 4: Run all gemini_to_openai tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestGeminiResponseToOpenAI -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add internal/transform/gemini_to_openai.go internal/transform/gemini_to_openai_test.go
git commit -m "feat: add usage details, refusal, service_tier to Gemini response transform"
```

---

### Task 7: Add streaming usage details to both backends

**Files:**
- Modify: `internal/transform/claude_to_openai.go:271-289`
- Modify: `internal/transform/gemini_to_openai.go:204-261`
- Test: `internal/transform/claude_to_openai_test.go`
- Test: `internal/transform/gemini_to_openai_test.go`

- [ ] **Step 1: Write failing test for Claude streaming usage details**

Add to `internal/transform/claude_to_openai_test.go`:

```go
func TestClaudeStreamEventToOpenAI_MessageDeltaUsageDetails(t *testing.T) {
	state := &ClaudeStreamState{}
	event := map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn"},
		"usage": map[string]any{"output_tokens": float64(42)},
	}
	chunks := ClaudeStreamEventToOpenAI(event, "message_delta", "claude-sonnet-4-6", "chatcmpl-test", state)
	if len(chunks) == 0 {
		t.Fatal("expected chunk")
	}
	usage, ok := chunks[0]["usage"].(map[string]any)
	if !ok {
		t.Fatal("usage missing")
	}
	if _, exists := usage["prompt_tokens_details"]; !exists {
		t.Error("prompt_tokens_details missing from streaming usage")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -run TestClaudeStreamEventToOpenAI_MessageDeltaUsageDetails -v`
Expected: FAIL — usage only has `completion_tokens`

- [ ] **Step 3: Update Claude stream message_delta to include full usage**

In `internal/transform/claude_to_openai.go`, update the `message_delta` case:

```go
	case "message_delta":
		deltaObj, _ := event["delta"].(map[string]any)
		stopReason := ""
		if deltaObj != nil {
			stopReason, _ = deltaObj["stop_reason"].(string)
		}
		finishReason := mapClaudeStopReason(stopReason)

		chunk := makeChunk(map[string]any{}, finishReason)

		// include usage if present in the event
		if usage, ok := event["usage"].(map[string]any); ok {
			outputTokens := toInt(usage["output_tokens"])
			chunk["usage"] = map[string]any{
				"prompt_tokens":     0,
				"completion_tokens": outputTokens,
				"total_tokens":      outputTokens,
				"prompt_tokens_details": map[string]any{
					"cached_tokens": 0,
					"audio_tokens":  0,
				},
				"completion_tokens_details": map[string]any{
					"reasoning_tokens":            0,
					"audio_tokens":               0,
					"accepted_prediction_tokens":  0,
					"rejected_prediction_tokens":  0,
				},
			}
		}

		return []map[string]any{chunk}
```

- [ ] **Step 4: Update Gemini stream chunk to include service_tier**

In `internal/transform/gemini_to_openai.go`, in `GeminiStreamChunkToOpenAI`, add `"service_tier": "default"` to the returned map:

```go
	return map[string]any{
		"id":                 responseID,
		"object":             "chat.completion.chunk",
		"created":            ts,
		"model":              model,
		"system_fingerprint": nil,
		"service_tier":       "default",
		"choices":            choices,
	}
```

- [ ] **Step 5: Run all streaming tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/transform/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add internal/transform/claude_to_openai.go internal/transform/gemini_to_openai.go internal/transform/claude_to_openai_test.go internal/transform/gemini_to_openai_test.go
git commit -m "feat: add full usage details to streaming responses"
```

---

### Task 8: Rewrite Responses API handler — non-streaming path

**Files:**
- Modify: `internal/handler/openai.go:176-460`
- Create: `internal/handler/openai_test.go`

- [ ] **Step 1: Create test file with test for Responses API input parsing**

Create `internal/handler/openai_test.go`:

```go
package handler

import (
	"testing"

	"llmux/internal/transform"
)

func TestResponsesInputToMessages_String(t *testing.T) {
	msgs, instructions := responsesInputToMessages(
		"Hello",
		"Be helpful",
		nil,
	)
	if len(msgs) == 0 {
		t.Fatal("expected messages")
	}
	// instructions become system message
	if msgs[0].Role != "system" {
		t.Errorf("first message role: got %q, want system", msgs[0].Role)
	}
	text, _ := msgs[0].Content.(string)
	if text != "Be helpful" {
		t.Errorf("instructions content: got %q", text)
	}
	// input string becomes user message
	if msgs[1].Role != "user" {
		t.Errorf("second message role: got %q, want user", msgs[1].Role)
	}
	_ = instructions
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
}

func TestResponsesToolsToOpenAI_WebSearchIgnored(t *testing.T) {
	tools := []any{
		map[string]any{"type": "web_search"},
		map[string]any{
			"type": "function",
			"name": "foo",
		},
	}
	result := responsesToolsToOpenAI(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool (web_search dropped), got %d", len(result))
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
	if toIntH(result["input_tokens"]) != 100 {
		t.Errorf("input_tokens: got %v", result["input_tokens"])
	}
	if toIntH(result["output_tokens"]) != 50 {
		t.Errorf("output_tokens: got %v", result["output_tokens"])
	}
	details, _ := result["output_tokens_details"].(map[string]any)
	if details == nil {
		t.Fatal("output_tokens_details missing")
	}
	if toIntH(details["reasoning_tokens"]) != 5 {
		t.Errorf("reasoning_tokens: got %v", details["reasoning_tokens"])
	}
}

// helper to convert any numeric to int (mirrors transform.toInt but in handler pkg)
func toIntH(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	}
	return 0
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/handler/ -run "TestResponses" -v`
Expected: FAIL — functions don't exist yet

- [ ] **Step 3: Extract helper functions and rewrite ResponsesAPI handler**

Replace the entire `ResponsesAPI` function in `internal/handler/openai.go` with a properly decomposed implementation. Add these helper functions above `ResponsesAPI`:

```go
// converts Responses API input (string or array) + instructions into ChatMessages
func responsesInputToMessages(input any, instructions string, tools []any) ([]transform.ChatMessage, string) {
	var messages []transform.ChatMessage

	if instructions != "" {
		messages = append(messages, transform.ChatMessage{Role: "system", Content: instructions})
	}

	switch rawInput := input.(type) {
	case string:
		messages = append(messages, transform.ChatMessage{Role: "user", Content: rawInput})
	case []any:
		for _, item := range rawInput {
			switch v := item.(type) {
			case string:
				messages = append(messages, transform.ChatMessage{Role: "user", Content: v})
			case map[string]any:
				itemType, _ := v["type"].(string)

				// function_call_output -> tool message
				if itemType == "function_call_output" {
					callID, _ := v["call_id"].(string)
					output, _ := v["output"].(string)
					messages = append(messages, transform.ChatMessage{
						Role:       "tool",
						Content:    output,
						ToolCallID: &callID,
					})
					continue
				}

				// standard message item
				role, _ := v["role"].(string)
				if role == "" {
					role = "user"
				}
				if content, ok := v["content"]; ok {
					messages = append(messages, transform.ChatMessage{Role: role, Content: content})
				} else if text, ok := v["text"].(string); ok {
					messages = append(messages, transform.ChatMessage{Role: role, Content: text})
				}
			}
		}
	default:
		if rawInput != nil {
			messages = append(messages, transform.ChatMessage{Role: "user", Content: fmt.Sprintf("%v", rawInput)})
		}
	}

	return messages, instructions
}

// converts Responses API tools to Chat Completions tools format
// function tools in Responses API have name/description/parameters at top level (no "function" wrapper)
// built-in tools (web_search, file_search, etc.) are silently dropped
func responsesToolsToOpenAI(tools []any) []map[string]interface{} {
	var result []map[string]interface{}
	for _, t := range tools {
		tm, ok := t.(map[string]any)
		if !ok {
			continue
		}
		toolType, _ := tm["type"].(string)
		if toolType != "function" {
			continue
		}
		fn := map[string]interface{}{}
		if name, ok := tm["name"].(string); ok {
			fn["name"] = name
		}
		if desc, ok := tm["description"].(string); ok {
			fn["description"] = desc
		}
		if params, ok := tm["parameters"]; ok {
			fn["parameters"] = params
		}
		if strict, ok := tm["strict"]; ok {
			fn["strict"] = strict
		}
		result = append(result, map[string]interface{}{
			"type":     "function",
			"function": fn,
		})
	}
	return result
}

// converts Chat Completions usage to Responses API usage format
func buildResponsesUsage(chatUsage map[string]any) map[string]any {
	if chatUsage == nil {
		return map[string]any{
			"input_tokens":  0,
			"output_tokens": 0,
			"total_tokens":  0,
		}
	}

	inputTokens := toIntHandler(chatUsage["prompt_tokens"])
	outputTokens := toIntHandler(chatUsage["completion_tokens"])
	totalTokens := toIntHandler(chatUsage["total_tokens"])

	result := map[string]any{
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
		"total_tokens":  totalTokens,
	}

	if ptd, ok := chatUsage["prompt_tokens_details"].(map[string]any); ok {
		result["input_tokens_details"] = map[string]any{
			"cached_tokens": toIntHandler(ptd["cached_tokens"]),
		}
	} else {
		result["input_tokens_details"] = map[string]any{"cached_tokens": 0}
	}

	if ctd, ok := chatUsage["completion_tokens_details"].(map[string]any); ok {
		result["output_tokens_details"] = map[string]any{
			"reasoning_tokens": toIntHandler(ctd["reasoning_tokens"]),
		}
	} else {
		result["output_tokens_details"] = map[string]any{"reasoning_tokens": 0}
	}

	return result
}

// converts OpenAI chat completion tool_calls to Responses API function_call output items
func chatToolCallsToOutputItems(toolCalls []any) []any {
	var items []any
	for _, tc := range toolCalls {
		tcMap, ok := tc.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := tcMap["function"].(map[string]any)
		if fn == nil {
			continue
		}
		callID, _ := tcMap["id"].(string)
		name, _ := fn["name"].(string)
		args, _ := fn["arguments"].(string)
		items = append(items, map[string]any{
			"id":        callID,
			"type":      "function_call",
			"call_id":   callID,
			"name":      name,
			"arguments": args,
			"status":    "completed",
		})
	}
	return items
}

func toIntHandler(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}
```

Now rewrite the `ResponsesAPI` function:

```go
// handles POST /v1/responses (OpenAI Responses API)
func ResponsesAPI(r *Router) gin.HandlerFunc {
	return func(c *gin.Context) {
		var data map[string]any
		if err := c.ShouldBindJSON(&data); err != nil {
			writeOpenAIError(c, http.StatusBadRequest, err)
			return
		}

		model, _ := data["model"].(string)
		if model == "" {
			model = defaultModel
		}

		stream, _ := data["stream"].(bool)
		instructions, _ := data["instructions"].(string)
		input := data["input"]

		// convert Responses API tools to Chat Completions format
		var chatTools []map[string]interface{}
		if rawTools, ok := data["tools"].([]any); ok {
			chatTools = responsesToolsToOpenAI(rawTools)
		}

		messages, _ := responsesInputToMessages(input, instructions, nil)

		req := transform.ChatRequest{
			Model:    model,
			Messages: messages,
			Stream:   stream,
		}

		// forward supported parameters
		if temp, ok := data["temperature"].(float64); ok {
			req.Temperature = &temp
		}
		if topP, ok := data["top_p"].(float64); ok {
			req.TopP = &topP
		}
		if maxOut, ok := data["max_output_tokens"].(float64); ok {
			v := int(maxOut)
			req.MaxCompletionTokens = &v
		}
		if stop, ok := data["stop"]; ok {
			req.Stop = stop
		}
		if fp, ok := data["frequency_penalty"].(float64); ok {
			req.FrequencyPenalty = &fp
		}
		if pp, ok := data["presence_penalty"].(float64); ok {
			req.PresencePenalty = &pp
		}
		if len(chatTools) > 0 {
			req.Tools = chatTools
		}
		if tc, ok := data["tool_choice"]; ok {
			req.ToolChoice = tc
		}
		if ptc, ok := data["parallel_tool_calls"].(bool); ok {
			req.ParallelToolCalls = &ptc
		}
		if user, ok := data["user"].(string); ok {
			req.User = &user
		}
		if st, ok := data["service_tier"].(string); ok {
			req.ServiceTier = &st
		}

		// reasoning: { effort: "low" | "medium" | "high" | ... }
		if reasoning, ok := data["reasoning"].(map[string]any); ok {
			if effort, ok := reasoning["effort"].(string); ok && effort != "" {
				req.ReasoningEffort = &effort
			}
		}

		// text.format -> response_format
		if text, ok := data["text"].(map[string]any); ok {
			if format, ok := text["format"].(map[string]any); ok {
				req.ResponseFormat = format
			}
		}

		if stream {
			handleResponsesStream(c, r, model, &req, data)
			return
		}

		body, status, actualModel, err := r.Send(c.Request.Context(), model, &req)
		if err != nil {
			code := status
			if code == 0 {
				code = http.StatusInternalServerError
			}
			writeOpenAIError(c, code, err)
			return
		}
		if actualModel != model {
			c.Header("X-Fallback-Model", actualModel)
		}

		var openaiResp map[string]any
		if err := json.Unmarshal(body, &openaiResp); err != nil {
			writeOpenAIError(c, http.StatusInternalServerError, err)
			return
		}

		// extract content and tool_calls from chat completion response
		outputText := ""
		var outputItems []any
		var toolCallItems []any

		if choices, ok := openaiResp["choices"].([]any); ok && len(choices) > 0 {
			if cm, ok := choices[0].(map[string]any); ok {
				if msg, ok := cm["message"].(map[string]any); ok {
					outputText, _ = msg["content"].(string)

					if tcs, ok := msg["tool_calls"].([]any); ok && len(tcs) > 0 {
						toolCallItems = chatToolCallsToOutputItems(tcs)
					}
				}
			}
		}

		responseID := "resp_" + strings.ReplaceAll(fmt.Sprintf("%x", time.Now().UnixNano()), "-", "")
		msgID := fmt.Sprintf("msg_%x", time.Now().UnixNano())

		// build message output item
		msgItem := map[string]any{
			"type": "message",
			"id":   msgID,
			"role": "assistant",
			"content": []any{map[string]any{
				"type":        "output_text",
				"text":        outputText,
				"annotations": []any{},
			}},
			"status": "completed",
		}
		outputItems = append(outputItems, msgItem)
		outputItems = append(outputItems, toolCallItems...)

		// convert usage
		chatUsage, _ := openaiResp["usage"].(map[string]any)
		responsesUsage := buildResponsesUsage(chatUsage)

		respBody := map[string]any{
			"id":                 responseID,
			"object":             "response",
			"created_at":         float64(time.Now().Unix()),
			"model":              actualModel,
			"status":             "completed",
			"error":              nil,
			"incomplete_details": nil,
			"instructions":       nilIfEmpty(instructions),
			"metadata":           data["metadata"],
			"output":             outputItems,
			"output_text":        outputText,
			"usage":              responsesUsage,
			"service_tier":       "default",
		}

		out, err := json.Marshal(respBody)
		if err != nil {
			writeOpenAIError(c, http.StatusInternalServerError, err)
			return
		}
		c.Data(status, "application/json; charset=utf-8", out)
	}
}

func nilIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./internal/handler/ -run "TestResponses|TestBuildResponsesUsage" -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add internal/handler/openai.go internal/handler/openai_test.go
git commit -m "feat: rewrite Responses API handler with full spec compliance"
```

---

### Task 9: Rewrite Responses API handler — streaming path

**Files:**
- Modify: `internal/handler/openai.go`

- [ ] **Step 1: Add handleResponsesStream function**

Add to `internal/handler/openai.go`:

```go
// handles streaming for the Responses API
func handleResponsesStream(c *gin.Context, r *Router, model string, req *transform.ChatRequest, data map[string]any) {
	status, ch, actualModel, err := r.Stream(c.Request.Context(), model, req)
	if err != nil {
		code := status
		if code == 0 {
			code = http.StatusInternalServerError
		}
		writeOpenAIError(c, code, err)
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")
	if actualModel != model {
		c.Header("X-Fallback-Model", actualModel)
	}

	responseID := "resp_" + strings.ReplaceAll(fmt.Sprintf("%x", time.Now().UnixNano()), "-", "")
	createdAt := float64(time.Now().Unix())
	msgID := fmt.Sprintf("msg_%x", time.Now().UnixNano())
	instructions, _ := data["instructions"].(string)

	flusher, canFlush := c.Writer.(http.Flusher)

	writeSSE := func(eventType string, payload any) {
		b, err := json.Marshal(payload)
		if err != nil {
			return
		}
		fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", eventType, b)
		if canFlush {
			flusher.Flush()
		}
	}

	// lifecycle events
	writeSSE("response.created", map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id":                 responseID,
			"object":             "response",
			"model":              actualModel,
			"status":             "in_progress",
			"created_at":         createdAt,
			"error":              nil,
			"incomplete_details": nil,
			"instructions":       nilIfEmpty(instructions),
			"metadata":           data["metadata"],
			"output":             []any{},
			"usage":              nil,
			"service_tier":       "default",
		},
	})
	writeSSE("response.in_progress", map[string]any{
		"type": "response.in_progress",
		"response": map[string]any{
			"id":         responseID,
			"object":     "response",
			"model":      actualModel,
			"status":     "in_progress",
			"created_at": createdAt,
			"output":     []any{},
		},
	})
	writeSSE("response.output_item.added", map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"type":    "message",
			"id":      msgID,
			"role":    "assistant",
			"content": []any{},
			"status":  "in_progress",
		},
	})
	writeSSE("response.content_part.added", map[string]any{
		"type":          "response.content_part.added",
		"output_index":  0,
		"content_index": 0,
		"part":          map[string]any{"type": "output_text", "text": ""},
	})

	// stream deltas
	var fullText strings.Builder
	var toolCallItems []any

	for raw := range ch {
		line := strings.TrimSpace(string(raw))
		if line == "data: [DONE]" || line == "" {
			continue
		}
		jsonPart := strings.TrimPrefix(line, "data: ")
		var chunk map[string]any
		if err := json.Unmarshal([]byte(jsonPart), &chunk); err != nil {
			continue
		}
		choices, _ := chunk["choices"].([]any)
		for _, ch := range choices {
			cm, _ := ch.(map[string]any)
			if cm == nil {
				continue
			}
			delta, _ := cm["delta"].(map[string]any)
			if delta == nil {
				continue
			}

			// text content
			if text, _ := delta["content"].(string); text != "" {
				fullText.WriteString(text)
				writeSSE("response.output_text.delta", map[string]any{
					"type":          "response.output_text.delta",
					"output_index":  0,
					"content_index": 0,
					"delta":         text,
				})
			}

			// tool calls
			if tcs, ok := delta["tool_calls"].([]any); ok {
				for _, tc := range tcs {
					tcMap, _ := tc.(map[string]any)
					if tcMap == nil {
						continue
					}
					fn, _ := tcMap["function"].(map[string]any)
					if fn == nil {
						continue
					}
					// accumulate for final response
					if id, ok := tcMap["id"].(string); ok && id != "" {
						name, _ := fn["name"].(string)
						toolCallItems = append(toolCallItems, map[string]any{
							"id":        id,
							"type":      "function_call",
							"call_id":   id,
							"name":      name,
							"arguments": "",
							"status":    "completed",
						})
					}
					if args, ok := fn["arguments"].(string); ok && args != "" && len(toolCallItems) > 0 {
						last := toolCallItems[len(toolCallItems)-1].(map[string]any)
						last["arguments"] = last["arguments"].(string) + args
					}

					writeSSE("response.function_call_arguments.delta", map[string]any{
						"type":         "response.function_call_arguments.delta",
						"output_index": len(toolCallItems) - 1,
						"delta":        fn["arguments"],
					})
				}
			}
		}
	}

	accumulated := fullText.String()

	// completion events
	writeSSE("response.output_text.done", map[string]any{
		"type":          "response.output_text.done",
		"output_index":  0,
		"content_index": 0,
		"text":          accumulated,
	})
	writeSSE("response.content_part.done", map[string]any{
		"type":          "response.content_part.done",
		"output_index":  0,
		"content_index": 0,
		"part":          map[string]any{"type": "output_text", "text": accumulated},
	})
	writeSSE("response.output_item.done", map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":    "message",
			"id":      msgID,
			"role":    "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": accumulated, "annotations": []any{}}},
			"status":  "completed",
		},
	})

	// emit function_call output items done events
	for i, item := range toolCallItems {
		writeSSE("response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"output_index": i + 1,
			"item":         item,
		})
	}

	// final output array
	var allOutput []any
	allOutput = append(allOutput, map[string]any{
		"type":    "message",
		"id":      msgID,
		"role":    "assistant",
		"content": []any{map[string]any{"type": "output_text", "text": accumulated, "annotations": []any{}}},
		"status":  "completed",
	})
	allOutput = append(allOutput, toolCallItems...)

	writeSSE("response.completed", map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id":                 responseID,
			"object":             "response",
			"model":              actualModel,
			"status":             "completed",
			"created_at":         createdAt,
			"error":              nil,
			"incomplete_details": nil,
			"instructions":       nilIfEmpty(instructions),
			"metadata":           data["metadata"],
			"output":             allOutput,
			"output_text":        accumulated,
			"service_tier":       "default",
			"usage": map[string]any{
				"input_tokens":          0,
				"output_tokens":         0,
				"total_tokens":          0,
				"input_tokens_details":  map[string]any{"cached_tokens": 0},
				"output_tokens_details": map[string]any{"reasoning_tokens": 0},
			},
		},
	})
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd /Users/anton/Code/go/llmux && go build ./...`
Expected: Compiles successfully

- [ ] **Step 3: Run all tests**

Run: `cd /Users/anton/Code/go/llmux && go test ./...`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add internal/handler/openai.go
git commit -m "feat: rewrite Responses API streaming with tool call support"
```

---

### Task 10: Final integration verification

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/anton/Code/go/llmux && go test ./... -v`
Expected: All PASS

- [ ] **Step 2: Run go vet**

Run: `cd /Users/anton/Code/go/llmux && go vet ./...`
Expected: No issues

- [ ] **Step 3: Verify build**

Run: `cd /Users/anton/Code/go/llmux && go build -o /dev/null ./cmd/server/`
Expected: Builds successfully

- [ ] **Step 4: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "chore: final cleanup for OpenAI spec compliance"
```
