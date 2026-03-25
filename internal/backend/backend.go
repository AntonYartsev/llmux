package backend

import "context"

// returned when a backend returns an HTTP error response
type BackendError struct {
	StatusCode int
	Message    string
}

func (e *BackendError) Error() string { return e.Message }

// single chunk from streaming response
type StreamChunk struct {
	Data  []byte
	Error error
}

// describes a model available from backend
type ModelInfo struct {
	ID          string
	Provider    string
	DisplayName string
	Description string
}

// interface implemented by all provider backends
type Backend interface {
	Name() string
	IsAvailable() bool
	// sends request and returns the full response body, HTTP status, and error
	Send(ctx context.Context, model string, payload map[string]any) ([]byte, int, error)
	// sends pre-serialized bytes (used by native proxy handlers to avoid unmarshal/remarshal)
	SendRaw(ctx context.Context, model string, body []byte) ([]byte, int, error)
	// sends request and returns a channel of response chunks
	Stream(ctx context.Context, model string, payload map[string]any) (int, <-chan StreamChunk, error)
	// streams from pre-serialized bytes (used by native proxy handlers)
	StreamRaw(ctx context.Context, model string, body []byte) (int, <-chan StreamChunk, error)
	// returns the list of models available from this backend
	ListModels() []ModelInfo
}
