package config

import (
	"os"
	"runtime"
	"strings"
)

// API Endpoints
const CodeAssistEndpoint = "https://cloudcode-pa.googleapis.com"

// client configuration
const GeminiCLIVersion = "0.1.20"

// Gemini OAuth configuration
const GeminiClientID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
const GeminiClientSecret = "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"

var GeminiScopes = []string{
	"https://www.googleapis.com/auth/cloud-platform",
	"https://www.googleapis.com/auth/userinfo.email",
	"https://www.googleapis.com/auth/userinfo.profile",
}

// Claude OAuth configuration
const ClaudeClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
const ClaudeAuthURL = "https://claude.ai/oauth/authorize"
const ClaudeTokenURL = "https://platform.claude.com/v1/oauth/token"

var ClaudeScopes = []string{
	"org:create_api_key",
	"user:profile",
	"user:inference",
	"user:sessions:claude_code",
	"user:mcp_servers",
	"user:file_upload",
}

// contains all safety categories set to BLOCK_NONE
var DefaultSafetySettings = []map[string]string{
	{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_UNSPECIFIED", "threshold": "BLOCK_NONE"},
	{"category": "HARM_CATEGORY_JAILBREAK", "threshold": "BLOCK_NONE"},
}

// holds all application configuration values
type AppConfig struct {
	Host               string
	Port               string
	AuthPassword       string
	LogLevel           string
	LogFile            string
	TLSCertFile        string
	TLSKeyFile         string
	FallbackChains     string // raw env value
	ModelBackendMap    string // raw env value, format "model:backend,..."
	CredentialFile     string
	GeminiCredentials  string // JSON string from env
	ClaudeCredentials  string // JSON string from env
	GoogleCloudProject string
}

// global application configuration instance
var Cfg AppConfig

// populates Cfg by parsing CLI flags then falling back to environment
// variables and finally built-in defaults. godotenv.Load() should be called
// before Load() so that .env values are already present in the environment
func Load() {
	// collect flag values from os.Args
	flags := parseFlags()

	Cfg.Host = firstNonEmpty(flags["host"], os.Getenv("HOST"), "0.0.0.0")
	Cfg.Port = firstNonEmpty(flags["port"], os.Getenv("PORT"), "8888")
	Cfg.AuthPassword = firstNonEmpty(flags["password"], os.Getenv("AUTH_PASSWORD"), os.Getenv("GEMINI_AUTH_PASSWORD"))
	Cfg.LogLevel = firstNonEmpty(flags["log-level"], os.Getenv("LOG_LEVEL"), "info")
	Cfg.LogFile = firstNonEmpty(flags["log-file"], os.Getenv("LOG_FILE"))
	Cfg.TLSCertFile = firstNonEmpty(flags["tls-cert"], os.Getenv("TLS_CERT_FILE"))
	Cfg.TLSKeyFile = firstNonEmpty(flags["tls-key"], os.Getenv("TLS_KEY_FILE"))
	Cfg.FallbackChains = firstNonEmpty(os.Getenv("FALLBACK_CHAINS"))
	Cfg.ModelBackendMap = firstNonEmpty(os.Getenv("MODEL_BACKEND_MAP"))
	Cfg.CredentialFile = firstNonEmpty(flags["credential-file"], os.Getenv("CREDENTIAL_FILE"), "~/.llmux/credentials.json")
	Cfg.GeminiCredentials = firstNonEmpty(os.Getenv("GEMINI_CREDENTIALS"))
	Cfg.ClaudeCredentials = firstNonEmpty(os.Getenv("CLAUDE_CREDENTIALS"))
	Cfg.GoogleCloudProject = firstNonEmpty(os.Getenv("GOOGLE_CLOUD_PROJECT"), os.Getenv("GCLOUD_PROJECT"))
}

// scans os.Args for --key=value style flags and returns a map
func parseFlags() map[string]string {
	result := make(map[string]string)
	known := []string{"port", "host", "password", "log-level", "log-file", "tls-cert", "tls-key", "credential-file"}
	for _, arg := range os.Args[1:] {
		for _, key := range known {
			prefix := "--" + key + "="
			if strings.HasPrefix(arg, prefix) {
				result[key] = strings.TrimPrefix(arg, prefix)
				break
			}
		}
	}
	return result
}

// returns the first non-empty string from the provided values
func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// returns a human-readable platform identifier matching the gemini-cli user-agent format
func GetPlatformString() string {
	goos := runtime.GOOS
	goarch := runtime.GOARCH
	switch {
	case goos == "linux" && goarch == "amd64":
		return "Linux; x86_64"
	case goos == "linux" && goarch == "arm64":
		return "Linux; aarch64"
	case goos == "darwin" && goarch == "amd64":
		return "Macintosh; Intel Mac OS X"
	case goos == "darwin" && goarch == "arm64":
		return "Macintosh; Apple Silicon"
	default:
		return "Unknown"
	}
}

// returns the User-Agent string used for API requests
func GetUserAgent() string {
	return "GeminiCLI/" + GeminiCLIVersion + " (" + GetPlatformString() + ")"
}

// returns the platform string used in Code Assist API
// metadata payloads (e.g. "DARWIN_ARM64"), matching gemini-cli conventions
func getMetadataPlatform() string {
	goos := runtime.GOOS
	goarch := runtime.GOARCH
	switch {
	case goos == "linux" && goarch == "amd64":
		return "LINUX_AMD64"
	case goos == "linux" && goarch == "arm64":
		return "LINUX_ARM64"
	case goos == "darwin" && goarch == "amd64":
		return "DARWIN_AMD64"
	case goos == "darwin" && goarch == "arm64":
		return "DARWIN_ARM64"
	default:
		return "PLATFORM_UNSPECIFIED"
	}
}

// returns the metadata object sent in loadCodeAssist and onboardUser requests
// field names match what the Code Assist API proto expects
func GetClientMetadata(projectID string) map[string]any {
	return map[string]any{
		"ideType":     "IDE_UNSPECIFIED",
		"platform":    getMetadataPlatform(),
		"pluginType":  "GEMINI",
		"duetProject": projectID,
	}
}
