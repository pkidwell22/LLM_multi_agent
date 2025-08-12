from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError

# Each target can list fallback aliases; we try them in order.
TARGETS = [
    # already present, kept for idempotency
    (["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"], "Q4_K_M"),

    # summarizer
    (["bartowski/Mistral-7B-Instruct-v0.3-GGUF"], "Q4_K_M"),

    # contrastive backup
    (["bartowski/Qwen2.5-7B-Instruct-GGUF"], "Q4_K_M"),

    # ultra-fast utility (no "Meta-" prefix; include mirrors)
    ([
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "TheBloke/Llama-3.2-3B-Instruct-GGUF",
        "QuantFactory/Llama-3.2-3B-Instruct-GGUF",
    ], "Q4_K_M"),
]

DEST = Path("models")
DEST.mkdir(parents=True, exist_ok=True)

def pick(files, contains):
    want = contains.lower()
    scored = []
    for f in files:
        name = f.lower()
        if not name.endswith(".gguf"):
            continue
        score = 0
        if want in name: score += 5
        if "instruct" in name or "chat" in name: score += 3
        if "q4_k_m" in name: score += 2
        scored.append((score, f))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]

def try_repo(api: HfApi, repo_id: str, want: str) -> str | None:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        print(f"  !! repo not found: {repo_id}")
        return None
    except HfHubHTTPError as e:
        print(f"  !! HTTP error for {repo_id}: {e}")
        return None
    except Exception as e:
        print(f"  !! unexpected error for {repo_id}: {e}")
        return None
    fname = pick(files, want)
    if not fname:
        print(f"  !! no GGUF file matched in {repo_id}")
        return None
    return fname

def main():
    api = HfApi()
    for repo_aliases, want in TARGETS:
        print(f"\n== {repo_aliases[0]} :: contains='{want}' ==")
        fname = None
        chosen_repo = None
        for repo in repo_aliases:
            fname = try_repo(api, repo, want)
            if fname:
                chosen_repo = repo
                break
        if not fname:
            continue
        out = DEST / Path(fname).name
        if out.exists():
            print(f"  -- exists: {out}")
            continue
        print(f"  -> downloading from {chosen_repo}: {fname}")
        hf_hub_download(
            repo_id=chosen_repo,
            filename=fname,
            local_dir=str(DEST),
            local_dir_use_symlinks=False,
        )
        print(f"  ++ saved: {out}")

if __name__ == "__main__":
    main()
