import os
import re

from dotenv import load_dotenv
from groq import Groq

from app.rag.retriever import retrieve, NO_ANSWER, ROLE_LEVEL   

load_dotenv()                     


MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


RESTRICTED_CHECK = os.getenv("RESTRICTED_CHECK", "1") == "1"

SYSTEM = """You are an HR policy assistant for NovaTech Solutions.
Answer the question using ONLY the numbered context passages in the user message.

Rules:
- If the passages do not directly answer the exact question, reply with exactly: NOT_FOUND
  Passages that are only about a related topic do not count as an answer.
- Do not use outside knowledge and do not guess.
- Do not change who or what a passage is about. For example, a list of what an employee must
  put in a request is not a list of what a manager must check.
- Write plain sentences or short bullet points. Do not use tables and do not use the | character.
- Be concise. Copy numbers, days and amounts exactly as written in the passages.
- On the last line write USED: followed by the passage numbers you used, for example: USED: 1, 3
"""


def build_context(hits):
    """Chunks ko [1], [2]... number dekar ek text mein jodta hai."""
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}]\n{h['text']}")
    return "\n\n".join(parts)


def call_llm(system, user):
    """LLM ko call karne ki ek hi jagah. Provider badalna ho toh sirf ye function badlo."""
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY nahi mili. Project folder ki .env file mein daalo.")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    extra = {}
    if MODEL.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"     # kam soch-vichar = tez jawab, kam tokens

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_completion_tokens=2000,           # reasoning model ke soch ke tokens bhi isme ginte hain
        **extra,
    )
    return (completion.choices[0].message.content or "").strip()


def build_citations(hits, used):
    """LLM ke bataye chunk numbers se citation banata hai, stored metadata se (LLM nahi likhta).
    Ek hi section ke kai chunks ho toh ek citation: sabse chhota page se sabse bade end_page tak."""
    groups = {}
    for n in used:
        if 1 <= n <= len(hits):
            m = hits[n - 1]["meta"]
            key = (m["doc_name"], m["section_heading"])
            g = groups.setdefault(key, [m["page"], m["end_page"]])
            g[0] = min(g[0], m["page"])
            g[1] = max(g[1], m["end_page"])

    citations = []
    for (doc, section), (p1, p2) in groups.items():
        pages = f"p.{p1}" if p1 == p2 else f"pp.{p1}-{p2}"
        citations.append(f"{doc}, {section}, {pages}")
    return citations


def top_hit_is_restricted(question, role):
    
    top = retrieve(question, "hr", k=1)        
    return bool(top) and top[0]["meta"]["min_role_level"] > ROLE_LEVEL[role]


def answer(question, role, k=5):
    hits = retrieve(question, role, k)          
    if not hits:
        return {"answer": NO_ANSWER, "citations": []}

    if RESTRICTED_CHECK and top_hit_is_restricted(question, role):
        return {"answer": NO_ANSWER, "citations": []}

    user = f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}"
    text = call_llm(SYSTEM, user)

    if not text or text.upper().startswith("NOT_FOUND"):
        return {"answer": NO_ANSWER, "citations": []}

    used = []
    m = re.search(r"^\s*USED:\s*(.*)$", text, re.I | re.M)
    if m:
        used = [int(n) for n in re.findall(r"\d+", m.group(1))]
        text = text[:m.start()].strip()

    return {"answer": text, "citations": build_citations(hits, used)}


def show(question, role):
    print("=" * 78)
    print(f"[{role.upper()}] {question}")
    r = answer(question, role)
    print(r["answer"])
    if r["citations"]:
        for c in r["citations"]:
            print("  Source:", c)
    elif r["answer"] != NO_ANSWER:
        print("  (no sources reported)")


if __name__ == "__main__":
    show("How many casual leaves do I get in a year?", "employee")
    show("When can a manager reject a leave request?", "employee")
    show("When can a manager reject a leave request?", "manager")
    show("What checks should a manager do before approving a travel request?", "employee")
    show("What checks should a manager do before approving a travel request?", "manager")
    show("What is the weather today?", "employee")
