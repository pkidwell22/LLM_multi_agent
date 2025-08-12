# agents/prompts.py

PROMPTS = {
    "rag_answer": (
        "You are a careful literature researcher. Use ONLY the provided CONTEXT.\n"
        "Cite passages in natural prose (no bracketed numbers). If evidence is thin, say so.\n\n"
        "QUESTION:\n{q}\n\nCONTEXT:\n{context}\n\nAnswer concisely:"
    ),
    "summarize": (
        "Summarize the following literary context in 2–4 sentences, plain prose, no brackets:\n\n{context}"
    ),
    "plan": (
        "Given the user query, propose a short retrieval plan: list key terms, authors, works, and any dates.\n"
        "Keep it to 3–5 bullet points.\n\nQuery:\n{q}"
    ),
    "critic": (
        "Read the draft below. If any claims aren’t supported by the provided context, list what to verify next.\n"
        "Otherwise respond exactly with 'OK'.\n\nDraft:\n{q}\n\nContext:\n{context}"
    ),
}

def build_prompt(role: str, q: str, context: str = "") -> str:
    tpl = PROMPTS.get(role, PROMPTS["rag_answer"])
    return tpl.format(q=q, context=context or "")
