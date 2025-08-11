from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_path = hf_hub_download(
    repo_id="bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF-Q4_K_M",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    local_dir="../models"
)

print(f"Model downloaded to: {model_path}")

llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
out = llm("Q: 2+2?\nA:", max_tokens=16, temperature=0)
print("Model output:", out["choices"][0]["text"].strip())
