package auth

import (
	"encoding/base64"
	"net/http"
	"strings"

	"llmux/internal/config"

	"github.com/gin-gonic/gin"
)

// Gin middleware: checks the request for a valid password
// If config.Cfg.AuthPassword is empty - authentication is skipped
func Authenticate() gin.HandlerFunc {
	return func(c *gin.Context) {
		if config.Cfg.AuthPassword == "" {
			c.Next()
			return
		}

		password := ""

		// check ?key= query param
		if key := c.Query("key"); key != "" {
			password = key
		}

		// check Authorization: Bearer <token>
		if password == "" {
			if auth := c.GetHeader("Authorization"); strings.HasPrefix(auth, "Bearer ") {
				password = strings.TrimPrefix(auth, "Bearer ")
			}
		}

		// check Authorization: Basic <base64(user:pass)>
		if password == "" {
			if auth := c.GetHeader("Authorization"); strings.HasPrefix(auth, "Basic ") {
				encoded := strings.TrimPrefix(auth, "Basic ")
				if decoded, err := base64.StdEncoding.DecodeString(encoded); err == nil {
					parts := strings.SplitN(string(decoded), ":", 2)
					if len(parts) == 2 {
						password = parts[1]
					}
				}
			}
		}

		// check x-goog-api-key header
		if password == "" {
			if key := c.GetHeader("x-goog-api-key"); key != "" {
				password = key
			}
		}

		if password != config.Cfg.AuthPassword {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": gin.H{
					"message": "Invalid API key",
					"type":    "invalid_request_error",
					"code":    "invalid_api_key",
				},
			})
			return
		}

		c.Set("authenticated", true)
		c.Next()
	}
}
