package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"

	"llmux/internal/auth"
	"llmux/internal/backend"
	"llmux/internal/config"
	"llmux/internal/handler"
)

func main() {
	_ = godotenv.Load()

	subcommand := "serve"
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "auth", "status":
			subcommand = os.Args[1]
		}
	}

	config.Load()

	setupSlog(config.Cfg)

	switch subcommand {
	case "auth":
		runAuth()
	case "status":
		runStatus()
	default:
		runServer()
	}
}

// setup slog based logs
func setupSlog(cfg config.AppConfig) {
	var level slog.Level
	switch cfg.LogLevel {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	opts := &slog.HandlerOptions{Level: level}

	var w = os.Stderr
	if cfg.LogFile != "" {
		logPath := cfg.LogFile
		if len(logPath) >= 2 && logPath[:2] == "~/" {
			if home, err := os.UserHomeDir(); err == nil {
				logPath = filepath.Join(home, logPath[2:])
			}
		}
		if dir := filepath.Dir(logPath); dir != "" {
			_ = os.MkdirAll(dir, 0o700)
		}
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			slog.Warn("cannot open log file, falling back to stderr", "path", logPath, "err", err)
		} else {
			w = f
		}
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(w, opts)))
}

// handle the auth subcommand
func runAuth() {
	provider := ""
	callbackHost := ""
	callbackPort := ""

	// parse: llmux auth <provider> [--callback-host <host>] [--callback-port <port>]
	for i := 2; i < len(os.Args); i++ {
		switch {
		case os.Args[i] == "--callback-host" && i+1 < len(os.Args):
			callbackHost = os.Args[i+1]
			i++
		case os.Args[i] == "--callback-port" && i+1 < len(os.Args):
			callbackPort = os.Args[i+1]
			i++
		case provider == "":
			provider = os.Args[i]
		}
	}

	store := auth.NewCredentialStore(config.Cfg.CredentialFile)
	if err := store.Load(); err != nil {
		slog.Warn("failed to load existing credentials", "err", err)
	}

	switch provider {
	case "gemini":
		if err := auth.RunGeminiOAuthFlow(store, callbackHost, callbackPort); err != nil {
			fmt.Fprintf(os.Stderr, "Gemini auth failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Gemini authentication successful.")
	case "claude":
		if err := auth.RunClaudeOAuthFlow(store, callbackHost, callbackPort); err != nil {
			fmt.Fprintf(os.Stderr, "Claude auth failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Claude authentication successful.")
	default:
		fmt.Println("Usage: llmux auth [gemini|claude] [--callback-host <host>] [--callback-port <port>]")
	}
}

// prints authentication status for each provider.
func runStatus() {
	store := auth.NewCredentialStore(config.Cfg.CredentialFile)

	if err := store.LoadFromEnv(); err != nil {
		slog.Warn("failed to load credentials from env", "err", err)
	}
	if err := store.Load(); err != nil {
		slog.Warn("failed to load credentials from file", "err", err)
	}

	// gemini
	if _, err := auth.EnsureGeminiToken(store); err != nil {
		fmt.Printf("✗ Gemini: %v\n", err)
	} else {
		fmt.Println("✓ Gemini: authenticated")
	}

	// claude
	if _, err := auth.EnsureClaudeToken(store); err != nil {
		fmt.Printf("✗ Claude: %v\n", err)
	} else {
		fmt.Println("✓ Claude: authenticated")
	}
}

// starts proxy server
func runServer() {
	// load creds store (try env first, then file)
	store := auth.NewCredentialStore(config.Cfg.CredentialFile)
	if err := store.LoadFromEnv(); err != nil {
		slog.Warn("failed to load credentials from env", "err", err)
	}
	if err := store.Load(); err != nil {
		slog.Warn("failed to load credentials from file", "err", err)
	}

	// create backends (if credentials available)
	var geminiBackend *backend.GeminiBackend
	var claudeBackend *backend.ClaudeBackend

	if store.GetGemini() != nil {
		geminiBackend = backend.NewGeminiBackend(store)
		// run gemini onboarding in background goroutine
		go func() {
			if err := geminiBackend.RunOnboarding(); err != nil {
				slog.Warn("gemini onboarding failed", "err", err)
			}
		}()
	}
	if store.GetClaude() != nil {
		claudeBackend = backend.NewClaudeBackend(store)
	}

	// create router
	var gB backend.Backend
	if geminiBackend != nil {
		gB = geminiBackend
	}
	var cB backend.Backend
	if claudeBackend != nil {
		cB = claudeBackend
	}
	r := handler.NewRouter(gB, cB, config.Cfg)

	// setup Gin
	if config.Cfg.LogLevel != "debug" {
		gin.SetMode(gin.ReleaseMode)
	}
	engine := gin.New()
	engine.Use(gin.Recovery())
	engine.Use(corsMiddleware())

	// public routes
	engine.GET("/", rootHandler)
	engine.GET("/health", healthHandler(store, geminiBackend, claudeBackend))
	engine.OPTIONS("/*path", func(c *gin.Context) { c.Status(http.StatusNoContent) })

	// protected routes
	protected := engine.Group("/")
	protected.Use(auth.Authenticate())

	protected.POST("/v1/chat/completions", handler.ChatCompletions(r))
	protected.POST("/v1/responses", handler.ResponsesAPI(r))
	protected.GET("/v1/models", handler.ListModels(r))
	protected.GET("/v1/models/*id", handler.GetModel(r))

	// experimental: native Gemini routes (if gemini backend available)
	if geminiBackend != nil {
		protected.GET("/gemini/v1beta/models", handler.GeminiListModels(geminiBackend))
		protected.GET("/gemini/v1beta/models/:model", handler.GeminiGetModel(geminiBackend))
		protected.POST("/gemini/v1beta/models/:model/:action", handler.GeminiProxy(geminiBackend))
	}

	// experimental: native Claude routes (if claude backend available)
	if claudeBackend != nil {
		protected.POST("/claude/v1/messages", handler.ClaudeMessages(claudeBackend))
		protected.GET("/claude/v1/models", handler.ClaudeListModels(claudeBackend))
	}

	// setup HTTP server
	addr := config.Cfg.Host + ":" + config.Cfg.Port
	srv := &http.Server{
		Addr:    addr,
		Handler: engine,
	}

	// graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		slog.Info("starting server", "addr", addr)
		var err error
		if config.Cfg.TLSCertFile != "" && config.Cfg.TLSKeyFile != "" {
			err = srv.ListenAndServeTLS(config.Cfg.TLSCertFile, config.Cfg.TLSKeyFile)
		} else {
			err = srv.ListenAndServe()
		}
		if err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	<-quit
	slog.Info("shutting down server..")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("server forced to shutdown", "err", err)
	}
	slog.Info("server stopped")
}

// returns brief project summary
func rootHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"name":        "llmux",
		"version":     config.GeminiCLIVersion,
		"description": "LLM multiplexer",
	})
}

// returns the liveness status of each backend
func healthHandler(store *auth.CredentialStore, gemini *backend.GeminiBackend, claude *backend.ClaudeBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		geminiOK := gemini != nil && gemini.IsAvailable()
		claudeOK := claude != nil && claude.IsAvailable()
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"gemini": geminiOK,
			"claude": claudeOK,
		})
	}
}

// adds CORS headers to every response
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key, x-goog-api-key")
		c.Next()
	}
}
