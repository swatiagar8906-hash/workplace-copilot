import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "./db"


ROLE_LEVEL = {"employee": 1, "manager": 2, "hr": 3}


MAX_DISTANCE = None
NO_ANSWER = ("I couldn't find this in the policies available to your role. "
             "Please contact HR if you think you need this information.")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient(DB_PATH).get_collection("hr_policies")


def retrieve(query, role, k=5, max_distance=MAX_DISTANCE):
    
    level = ROLE_LEVEL[role]
    q_emb = model.encode([query]).tolist()

    res = col.query(
        query_embeddings=q_emb,
        n_results=k,
        where={"min_role_level": {"$lte": level}},   
    )

    
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if max_distance is not None and dist > max_distance:
            continue                      
        hits.append({"text": text, "meta": meta, "distance": dist})
    return hits           


def show(query, role, k=3):
    print("=" * 78)
    print(f"[{role.upper()}] {query}")
    hits = retrieve(query, role, k)
    if not hits:
        print("  -> NO ANSWER:", NO_ANSWER)
    for h in hits:
        m = h["meta"]
        print(f"  L{m['min_role_level']} | {m['doc_name']} > {m['section_heading']} "
              f"(p{m['page']}-{m['end_page']})  dist={h['distance']:.2f}")


if __name__ == "__main__":
    
    show("How many casual leaves do I get in a year?", "employee")
    show("How many casual leaves do I get in a year?", "manager")

    
    manager_questions = [
        "When can a manager reject a leave request?",
        "What checks should a manager do before approving a travel request?",
        "Who approves exceptions to the WFH policy?",
    ]
    for q in manager_questions:
        show(q, "employee")
        show(q, "manager")

    
    leaks = 0
    for q in manager_questions:
        for h in retrieve(q, "employee", k=20):
            if h["meta"]["min_role_level"] > ROLE_LEVEL["employee"]:
                leaks += 1
    print("=" * 78)
    print("ACCESS CHECK:", "FAILED, leaks =" if leaks else "PASSED, leaks =", leaks)
