import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

MODEL_PATH = "./models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "./models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

with open("/Users/swapnil/Downloads/Cat03.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

print(f"Image size: {len(image_base64)}")

chat_handler = Qwen25VLChatHandler(
    clip_model_path=MMPROJ_PATH,
    verbose=True,  # see if clip is processing
)

llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 means ALL layers on GPU
    verbose=True,
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ],
    max_tokens=300,
    temperature=0.2,
)

print("\nRESULT:", response["choices"][0]["message"]["content"])
