import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from app.rag.generator import answer
from app.rag.retriever import retrieve, NO_ANSWER, ROLE_LEVEL

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PAUSE = 2   

cases = json.loads((Path(__file__).parent / "questions.json").read_text())


def check(case, r):
    if case["type"] in ("restricted", "out_of_scope"):
        return r["answer"] == NO_ANSWER
    if r["answer"] == NO_ANSWER:
        return False
    
    norm = lambda t: unicodedata.normalize("NFKC", t).lower()
    text = norm(r["answer"])
    words_ok = all(norm(w) in text for w in case.get("must_contain", []))
    sec = case.get("expected_section", "")
    cites_ok = any(sec in c for c in r["citations"]) if sec else True
    return words_ok and cites_ok


score = defaultdict(lambda: [0, 0])      
failures = []
for case in cases:
    for i in range(RUNS):
        r = answer(case["question"], case["role"])
        ok = check(case, r)
        score[case["type"]][0] += ok
        score[case["type"]][1] += 1
        if not ok:
            failures.append((case["id"], case["role"], case["question"], r["answer"][:120], r["citations"]))
        time.sleep(PAUSE)


leaks = 0
for case in cases:
    for h in retrieve(case["question"], case["role"], k=20):
        if h["meta"]["min_role_level"] > ROLE_LEVEL[case["role"]]:
            leaks += 1

print("=" * 70)
print(f"RESULTS ({RUNS} runs per question)")
for t, (p, n) in score.items():
    print(f"  {t:14s} {p}/{n}  ({100 * p / n:.0f}%)")
print(f"  access leaks   {leaks}")
if failures:
    print("\nFAILURES")
    for f in failures:
        print(f"  [{f[0]} {f[1]}] {f[2]}\n     -> {f[3]!r}  {f[4]}")

