# Flask Local LLM Server

A local LLM server with an OpenAI-compatible REST API, powered by HuggingFace Transformers.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

For GPU support (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run the server (starts with TinyLlama by default ~1.1GB)
```bash
python app.py
```

Switch model via environment variable:
```bash
MODEL=phi2 python app.py
```

For gated models (Llama 2/3), set your HuggingFace token:
```bash
HF_TOKEN=hf_xxx MODEL=llama3-8b python app.py
```

---

## API Endpoints

### Health Check
```
GET /health
```

### Chat Completions (OpenAI-compatible)
```
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

### Text Completions
```
POST /v1/completions
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 200,
  "temperature": 0.8
}
```

### Switch Model at Runtime
```
POST /models/switch
Content-Type: application/json

{ "model": "mistral-7b" }
```

---

## Available Models

| Key         | Model                              | Size   | Notes              |
|-------------|-------------------------------------|--------|--------------------|
| `tinyllama` | TinyLlama-1.1B-Chat-v1.0           | ~1.1GB | Default, fast      |
| `phi2`      | microsoft/phi-2                    | ~2.7GB | Great for coding   |
| `qwen-500m` | Qwen1.5-0.5B-Chat                  | ~0.5GB | Smallest option    |
| `mistral-7b`| Mistral-7B-Instruct-v0.2           | ~7GB   | Recommended step up|
| `llama3-8b` | Meta-Llama-3-8B-Instruct           | ~8GB   | Requires HF token  |
| `llama2-13b`| Llama-2-13b-chat-hf                | ~13GB  | Requires HF token  |

---

## Example: Using with Python requests

```python
import requests

response = requests.post("http://localhost:5000/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Explain quantum computing briefly"}],
    "max_tokens": 200,
    "temperature": 0.7
})

print(response.json()["choices"][0]["message"]["content"])
```

## Example: Using with OpenAI SDK (pointing to local server)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="local")

response = client.chat.completions.create(
    model="tinyllama",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```