<p align="center">
  <img src="https://github.com/user-attachments/assets/77d5c34e-da5f-4bea-a61a-e71c16b65ec3" width="200" alt="" />
  <br><br>
  <a href="https://github.com/AntonYartsev/llmux/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/AntonYartsev/llmux/ci.yml?branch=main" alt="Build" />
  </a>
  <a href="https://github.com/AntonYartsev/llmux/releases">
    <img src="https://img.shields.io/github/v/release/AntonYartsev/llmux" alt="Release" />
  </a>
</p>

# llmux

One endpoint, any model.

## The idea
Use Gemini CLI and Claude Code as an [OpenAI-compatible](https://developers.openai.com/api/reference/overview) provider
```
Any OpenAI-compatible app
        ↓
    llmux (:8888)
        ├── gemini-*  →  Google Code Assist API  (Gemini CLI OAuth)
        └── claude-*  →  Anthropic Messages API  (Claude Code OAuth)
```

## Quick start

```shell
go build -o llmux ./cmd/server

./llmux auth gemini
./llmux auth claude
./llmux

# additional you may set callback host and port
./llmux auth claude --callback-host myserver.com --callback-port 8888
```

## Usage

```shell
# Start the server
./llmux

# Or with a custom credentials file
./llmux --credential-file=/path/to/credentials.json

# Or with a custom port and password
./llmux --port=9000 --password=secret
```

Works with Cursor, Continue, Raycast, Codex, and anything else that speaks OpenAI.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `POST` | `/v1/responses` | Responses API |
| `GET`  | `/v1/models` | List available models |
| `GET`  | `/health` | Health check |

<details>
<summary>curl example</summary>

```shell
curl http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer password" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

</details>

## Authentication

```shell
./llmux auth gemini   # opens browser → Google OAuth → saves tokens
./llmux auth claude   # opens browser → Claude OAuth (PKCE) → saves tokens
./llmux status        # check auth status
```

Credentials are stored in `~/.llmux/credentials.json` (mode `0600`). Override the path:

```shell
./llmux --credential-file=/path/to/creds.json auth gemini
# or
CREDENTIAL_FILE=/path/to/creds.json ./llmux auth gemini
```

Tokens refresh automatically. To skip the file and pass credentials directly, use `GEMINI_CREDENTIALS` / `CLAUDE_CREDENTIALS` env vars with the JSON string.

## Configuration

Env vars, `.env` file, or `--flags`. Priority: flags → env → `.env` → defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8888` | Listen port |
| `AUTH_PASSWORD` | | API key for clients |
| `CREDENTIAL_FILE` | `~/.llmux/credentials.json` | Path to credentials file |
| `GEMINI_CREDENTIALS` | | Gemini OAuth credentials (JSON string, overrides file) |
| `CLAUDE_CREDENTIALS` | | Claude OAuth credentials (JSON string, overrides file) |
| `GOOGLE_CLOUD_PROJECT` | auto | GCP project ID |
| `FALLBACK_CHAINS` | | Model fallback config |
| `MODEL_BACKEND_MAP` | | Force model→backend routing |
| `LOG_LEVEL` | `info` | `debug` `info` `warn` `error` |
| `LOG_FILE` | | Log file path |
| `TLS_CERT_FILE` | | TLS certificate path |
| `TLS_KEY_FILE` | | TLS key path |

## Roadmap

- [ ] Anthropic-compatible endpoint (`/v1/messages`) — use Claude Code with Gemini
- [ ] Gemini-compatible endpoint (`generateContent`) — use Gemini CLI with Claude
- [ ] Codex (Open Ai) backend
- [ ] OpenRouter backend
- [ ] Ollama / local models
- [ ] Response caching

## Acknowledgements

Gemini auth approach from [gzzhongqi/geminicli2api](https://github.com/gzzhongqi/geminicli2api).

## License

MIT
