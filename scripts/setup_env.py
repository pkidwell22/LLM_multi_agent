# scripts/setup_env.py
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = BASE_DIR / ".venv"
REQ_FILE = BASE_DIR / "requirements.txt"
MODELS_DIR = BASE_DIR / "models"

def run_cmd(cmd, shell=False):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, check=True, shell=shell)

def main():
    # 1. Create venv if missing
    if not VENV_DIR.exists():
        print("[+] Creating virtual environment...")
        run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("[✓] Virtual environment already exists")

    # 2. Install requirements
    print("[+] Installing requirements...")
    pip_path = VENV_DIR / "Scripts" / "pip" if os.name == "nt" else VENV_DIR / "bin" / "pip"
    run_cmd([str(pip_path), "install", "--upgrade", "pip"])
    run_cmd([str(pip_path), "install", "-r", str(REQ_FILE)])

    # 3. Create models dir
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"[✓] Models directory ready: {MODELS_DIR}")

    # 4. Optional: download GGUF model
    choice = input("Download Qwen2.5-7B-Instruct Q4_K_M GGUF now? (y/N): ").strip().lower()
    if choice == "y":
        model_url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        out_path = MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"
        run_cmd(["curl", "-L", model_url, "-o", str(out_path)])
        print(f"[✓] Downloaded model to {out_path}")

    # 5. Test llama.cpp
    print("[+] Running test_llama.py to verify...")
    run_cmd([str(pip_path).replace("pip", "python"), str(BASE_DIR / "test_llama.py")])

if __name__ == "__main__":
    main()
