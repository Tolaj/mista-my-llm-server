"""
Flask LLM Server - Run GGUF models locally via llama.cpp
Models must be placed in ./models/<model_key>/*.gguf
Example:
    models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from llama_cpp import Llama
import threading
import time
import os
from pathlib import Path
from typing import Optional
import json

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "qwen2.5-7b": "Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen2.5-3B-Instruct",
    "tinyllama": "TinyLlama-1.1B",
}

ACTIVE_MODEL_KEY = os.getenv("MODEL", "qwen2.5-3b")

llm: Optional[Llama] = None
model_name = ""
model_path = ""

lock = threading.Lock()

# Performance tuning
N_CTX = 8192
N_THREADS = max(1, os.cpu_count() // 2)
N_GPU_LAYERS = int(os.getenv("GPU_LAYERS", "35"))


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────


def find_gguf_file(model_key: str) -> Path:
    model_folder = MODELS_DIR / model_key

    if not model_folder.exists():
        raise FileNotFoundError(
            f"Model folder not found: {model_folder}\n"
            f"Place GGUF file inside this folder."
        )

    gguf_files = list(model_folder.glob("*.gguf"))

    if not gguf_files:
        raise FileNotFoundError(f"No GGUF file found in {model_folder}")

    return gguf_files[0]


def load_model(model_key: str):

    global llm, model_name, model_path

    gguf_path = find_gguf_file(model_key)

    print("\n" + "=" * 60)
    print(f"Loading model: {model_key}")
    print(f"Path: {gguf_path}")
    print(f"Context: {N_CTX}")
    print(f"Threads: {N_THREADS}")
    print(f"GPU layers: {N_GPU_LAYERS}")
    print("=" * 60 + "\n")

    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

    model_name = model_key
    model_path = str(gguf_path)

    print(f"✅ Model loaded: {model_key}\n")


# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────


def generate_chat(messages, max_tokens=512, temperature=0.7, top_p=0.9, stream=False):

    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "ok",
            "model": model_name,
            "model_path": model_path,
            "loaded": llm is not None,
            "ctx": N_CTX,
            "threads": N_THREADS,
            "gpu_layers": N_GPU_LAYERS,
        }
    )


@app.route("/v1/models", methods=["GET"])
def list_models():

    data = []

    for key in MODELS:
        try:
            path = find_gguf_file(key)
            exists = True
        except:
            path = None
            exists = False

        data.append(
            {
                "id": key,
                "object": "model",
                "owned_by": "local",
                "available": exists,
                "path": str(path) if path else None,
            }
        )

    return jsonify({"object": "list", "data": data})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():

    if llm is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json

    messages = data.get("messages")
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    stream = data.get("stream", False)

    if not messages:
        return jsonify({"error": "messages required"}), 400

    if stream:
        return stream_chat(messages, max_tokens, temperature, top_p)

    start = time.time()

    with lock:
        output = generate_chat(messages, max_tokens, temperature, top_p, stream=False)

    text = output["choices"][0]["message"]["content"]

    elapsed = time.time() - start

    return jsonify(
        {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_time": round(elapsed, 2)},
        }
    )


# ─────────────────────────────────────────────────────────────
# STREAMING
# ─────────────────────────────────────────────────────────────


def stream_chat(messages, max_tokens, temperature, top_p):

    def generate():

        with lock:

            stream = generate_chat(
                messages, max_tokens, temperature, top_p, stream=True
            )

            for chunk in stream:

                delta = chunk["choices"][0]["delta"]

                if "content" in delta:

                    data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [
                            {
                                "delta": {"content": delta["content"]},
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                    }

                    yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ─────────────────────────────────────────────────────────────
# MODEL SWITCH
# ─────────────────────────────────────────────────────────────


@app.route("/models/switch", methods=["POST"])
def switch_model():

    data = request.json
    key = data.get("model")

    if key not in MODELS:
        return jsonify({"error": "Invalid model key"}), 400

    try:
        load_model(key)
        return jsonify({"status": "ok", "model": key})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\nModels dir: {MODELS_DIR}")
    print(f"Active model: {ACTIVE_MODEL_KEY}")

    load_model(ACTIVE_MODEL_KEY)

    print("\nServer running at http://localhost:5001\n")

    app.run(host="0.0.0.0", port=5001, threaded=True)
