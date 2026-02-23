# 🧠 Mista Server — Local LLM Flask API

Run GGUF models locally via `llama.cpp` with a simple OpenAI-compatible REST API.

---

## 📁 Project Structure

```
mista-server/
├── app.py
├── models/
│   ├── qwen2.5-7b/
│   │   └── qwen2.5-7b-instruct-q5_k_m.gguf
│   ├── qwen2.5-3b/
│   │   └── qwen2.5-3b-instruct-q4_k_m.gguf
│   └── tinyllama/
│       └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install flask llama-cpp-python
```

### Enable Metal GPU (Mac — highly recommended for speed)

```bash
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

---

## 📥 Downloading Models

### Qwen 2.5 7B (default)
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q5_k_m*.gguf" \
  --local-dir ./models/qwen2.5-7b \
  --local-dir-use-symlinks False
```

### Qwen 2.5 3B (faster)
```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  --include "qwen2.5-3b-instruct-q4_k_m.gguf" \
  --local-dir ./models/qwen2.5-3b \
  --local-dir-use-symlinks False
```

### TinyLlama 1.1B (lightest)
```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --include "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --local-dir ./models/tinyllama \
  --local-dir-use-symlinks False
```

> **Note:** If the model downloads as split files (e.g. `*-00001-of-00002.gguf`), merge them first:
> ```bash
> llama-gguf-split --merge qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf qwen2.5-7b-instruct-q5_k_m.gguf
> ```
> Then delete the split files, keeping only the merged `.gguf`.

---

## 🚀 Running the Server

```bash
# Default model (qwen2.5-7b)
python app.py

# Choose a different model
MODEL=qwen2.5-3b python app.py

# With GPU acceleration
GPU_LAYERS=35 MODEL=qwen2.5-3b python app.py
```

Server runs at: `http://localhost:5001`

---

## 🌐 API Reference

### `GET /health`
Check server and model status.

```bash
curl http://localhost:5001/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "qwen2.5-3b",
  "loaded": true,
  "ctx": 2048,
  "threads": 8,
  "gpu_layers": 35
}
```

---

### `GET /v1/models`
List all available models and whether they are downloaded.

```bash
curl http://localhost:5001/v1/models
```

---

### `POST /v1/chat/completions`
Chat with the loaded model.

```bash
curl -X POST http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
  }'
```

**Streaming:**
```bash
curl -X POST http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

**Python example:**
```python
import requests

res = requests.post("http://localhost:5001/v1/chat/completions", json={
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
})

print(res.json()["choices"][0]["message"]["content"])
```

---

### `POST /models/switch`
Hot-swap the loaded model without restarting the server.

```bash
curl -X POST http://localhost:5001/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-3b"}'
```

Valid model keys: `qwen2.5-7b`, `qwen2.5-3b`, `tinyllama`

---

## ⚡ Performance Tuning

| Setting | Default | Recommended |
|--------|---------|-------------|
| `N_GPU_LAYERS` | `0` | `35` (Mac Metal) |
| `N_THREADS` | `cpu_count / 2` | `cpu_count` |
| `N_CTX` | `8192` | `2048` for speed |

Set via environment variables:
```bash
GPU_LAYERS=35 MODEL=qwen2.5-3b python app.py
```

Or edit directly in `app.py`:
```python
N_CTX = 2048
N_THREADS = os.cpu_count()
N_GPU_LAYERS = int(os.getenv("GPU_LAYERS", "35"))
```

---

## 📬 Postman Setup

1. Create a Collection called `Mista Server`
2. Add variable: `base_url = http://localhost:5001`
3. Use `{{base_url}}/health` etc. for all requests
4. For all POST requests: set **Body → raw → JSON** and header `Content-Type: application/json`

---

## 🛠 Supported Models

| Key | Model | Size | Speed |
|-----|-------|------|-------|
| `qwen2.5-7b` | Qwen2.5 7B Instruct Q5 | ~5.5GB | Slow |
| `qwen2.5-3b` | Qwen2.5 3B Instruct Q4 | ~2GB | Medium |
| `tinyllama` | TinyLlama 1.1B Q4 | ~0.7GB | Fast |