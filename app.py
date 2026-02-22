"""
Flask LLM Server - Run HuggingFace models locally via REST API
Models are downloaded once and stored in ./models/<model_key>/
To use a manually downloaded model, place it in ./models/<model_key>/
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
import time
import os
from pathlib import Path
from typing import Optional

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Local folder where models are stored
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    # Lightweight models (~1-2GB) — great for testing
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi2": "microsoft/phi-2",
    "qwen-500m": "Qwen/Qwen1.5-0.5B-Chat",
    # Medium models (~4-8GB)
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",  # requires HF token
    # Large models (~13GB+)
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",  # requires HF token
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",  # requires HF token
}

# ─── Change this to swap models ───────────────────────────────────────────────
ACTIVE_MODEL_KEY = os.getenv("MODEL", "tinyllama")
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Required for gated models (Llama)
# ──────────────────────────────────────────────────────────────────────────────

model = None
tokenizer = None
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model_name = ""


def get_local_model_path(model_key: str) -> Path:
    """Returns the local path where a model is/will be stored."""
    return MODELS_DIR / model_key


def is_model_downloaded(model_key: str) -> bool:
    """Check if the model already exists locally."""
    local_path = get_local_model_path(model_key)
    # Check if folder exists and has model files in it
    if not local_path.exists():
        return False
    files = (
        list(local_path.glob("*.safetensors"))
        + list(local_path.glob("*.bin"))
        + list(local_path.glob("*.pt"))
    )
    return len(files) > 0


def load_model(model_key: str):
    """
    Load model from local folder if available, otherwise download from HuggingFace
    and save it locally for future use.
    """
    global model, tokenizer, model_name

    local_path = get_local_model_path(model_key)

    # Determine source: local folder or HuggingFace
    if is_model_downloaded(model_key):
        source = str(local_path)
        print(f"\n{'='*60}")
        print(f"✅ Found local model: {local_path}")
        print(f"Loading from local folder (no download needed)")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
    else:
        hf_model_id = MODELS.get(model_key)
        if not hf_model_id:
            raise ValueError(
                f"Unknown model key: '{model_key}'.\n"
                f"Available keys: {list(MODELS.keys())}\n"
                f"Or place your model folder at: {local_path}"
            )
        source = hf_model_id
        print(f"\n{'='*60}")
        print(f"Model not found locally at: {local_path}")
        print(f"Downloading from HuggingFace: {hf_model_id}")
        print(f"Will save to: {local_path} for future use")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    model_name = source

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate precision
    load_kwargs = dict(
        token=HF_TOKEN,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)

    if device in ("cpu", "mps"):
        model = model.to(device)

    model.eval()

    # Save to local folder if it was downloaded from HuggingFace
    if not is_model_downloaded(model_key):
        print(f"\n💾 Saving model to {local_path} for future use...")
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_path))
        tokenizer.save_pretrained(str(local_path))
        print(f"✅ Model saved to {local_path}\n")

    print(f"✅ Model loaded successfully: {model_name}\n")
    print(f"📁 Local model folder: {local_path}\n")


def build_prompt(messages: list, system: Optional[str] = None) -> str:
    """Build a chat prompt using the tokenizer's chat template if available."""
    if system:
        messages = [{"role": "system", "content": system}] + messages

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback format
    prompt = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            prompt += f"[SYSTEM] {content}\n"
        elif role == "user":
            prompt += f"[USER] {content}\n[ASSISTANT] "
        elif role == "assistant":
            prompt += f"{content}\n"
    return prompt


def generate_response(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a completion for a given prompt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": model_name,
            "device": device,
            "model_loaded": model is not None,
            "models_dir": str(MODELS_DIR),
            "downloaded_models": [key for key in MODELS if is_model_downloaded(key)],
        }
    )


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": key,
                    "object": "model",
                    "owned_by": "local",
                    "local_path": str(get_local_model_path(key)),
                    "downloaded": is_model_downloaded(key),
                }
                for key in MODELS
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    stream = data.get("stream", False)

    if not messages:
        return jsonify({"error": "messages field is required"}), 400

    system_msg = None
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            chat_messages.append(msg)

    prompt = build_prompt(chat_messages, system=system_msg)

    if stream:
        return stream_chat(prompt, max_tokens, temperature, top_p)

    start = time.time()
    response_text = generate_response(prompt, max_tokens, temperature, top_p)
    elapsed = time.time() - start

    return jsonify(
        {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_time_seconds": round(elapsed, 2),
            },
        }
    )


def stream_chat(prompt: str, max_tokens: int, temperature: float, top_p: float):
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def generate():
        for token in streamer:
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [
                    {"delta": {"content": token}, "index": 0, "finish_reason": None}
                ],
            }
            yield f"data: {jsonify(chunk).get_data(as_text=True)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/v1/completions", methods=["POST"])
def completions():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)

    if not prompt:
        return jsonify({"error": "prompt field is required"}), 400

    response_text = generate_response(prompt, max_tokens, temperature, top_p)

    return jsonify(
        {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "model": model_name,
            "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
        }
    )


@app.route("/models/switch", methods=["POST"])
def switch_model():
    data = request.json
    new_model = data.get("model")
    if not new_model:
        return jsonify({"error": "model field required"}), 400
    try:
        load_model(new_model)
        return jsonify({"status": "ok", "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n📁 Models directory: {MODELS_DIR}")
    print(f"🔍 Active model: {ACTIVE_MODEL_KEY}")
    load_model(ACTIVE_MODEL_KEY)
    print("🚀 Starting Flask LLM Server on http://0.0.0.0:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
