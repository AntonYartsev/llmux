package auth

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"llmux/internal/config"
)

func setupRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(Authenticate())
	r.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})
	return r
}

func TestAuthenticate_NoPassword(t *testing.T) {
	config.Cfg.AuthPassword = ""
	r := setupRouter()

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestAuthenticate_BearerToken(t *testing.T) {
	config.Cfg.AuthPassword = "secret"
	r := setupRouter()

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("Authorization", "Bearer secret")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("valid bearer: expected 200, got %d", w.Code)
	}

	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.Header.Set("Authorization", "Bearer wrong")
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusUnauthorized {
		t.Errorf("invalid bearer: expected 401, got %d", w2.Code)
	}
	assertErrorBody(t, w2)
}

func TestAuthenticate_QueryKey(t *testing.T) {
	config.Cfg.AuthPassword = "secret"
	r := setupRouter()

	req := httptest.NewRequest(http.MethodGet, "/test?key=secret", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("valid key param: expected 200, got %d", w.Code)
	}

	req2 := httptest.NewRequest(http.MethodGet, "/test?key=wrong", nil)
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusUnauthorized {
		t.Errorf("invalid key param: expected 401, got %d", w2.Code)
	}
	assertErrorBody(t, w2)
}

func TestAuthenticate_BasicAuth(t *testing.T) {
	config.Cfg.AuthPassword = "secret"
	r := setupRouter()

	encoded := base64.StdEncoding.EncodeToString([]byte("user:secret"))
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("Authorization", "Basic "+encoded)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("valid basic auth: expected 200, got %d", w.Code)
	}

	encodedWrong := base64.StdEncoding.EncodeToString([]byte("user:wrong"))
	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.Header.Set("Authorization", "Basic "+encodedWrong)
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusUnauthorized {
		t.Errorf("invalid basic auth: expected 401, got %d", w2.Code)
	}
	assertErrorBody(t, w2)
}

func TestAuthenticate_GoogApiKey(t *testing.T) {
	config.Cfg.AuthPassword = "secret"
	r := setupRouter()

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("x-goog-api-key", "secret")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("valid x-goog-api-key: expected 200, got %d", w.Code)
	}

	req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
	req2.Header.Set("x-goog-api-key", "wrong")
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusUnauthorized {
		t.Errorf("invalid x-goog-api-key: expected 401, got %d", w2.Code)
	}
	assertErrorBody(t, w2)
}

func TestAuthenticate_NoCredentials(t *testing.T) {
	config.Cfg.AuthPassword = "secret"
	r := setupRouter()

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("no credentials: expected 401, got %d", w.Code)
	}
	assertErrorBody(t, w)
}

func assertErrorBody(t *testing.T, w *httptest.ResponseRecorder) {
	t.Helper()
	var body map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("failed to parse response body: %v", err)
	}
	errObj, ok := body["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected 'error' object in response body, got: %v", body)
	}
	if errObj["code"] != "invalid_api_key" {
		t.Errorf("expected error code 'invalid_api_key', got: %v", errObj["code"])
	}
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("expected error type 'invalid_request_error', got: %v", errObj["type"])
	}
}
