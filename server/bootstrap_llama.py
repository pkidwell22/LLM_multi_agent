# server/bootstrap_llama.py
import os, importlib.util, pathlib, sys

def ensure_llama_dlls():
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("llama_cpp")
    if spec and spec.origin:
        libdir = pathlib.Path(spec.origin).parent / "lib"
        if libdir.exists():
            os.add_dll_directory(str(libdir))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ensure_llama_dlls()
