import re
import fitz
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICY_DIR = Path("data/policies")
DB_PATH = "./db"

DOCS = {
    "Leave Policy (NT-HR-POL-001, v2026.1).pdf": ("Leave Policy", "leave"),
    "Work From Home Policy (NT-HR-POL-007, v2026.1).pdf": ("WFH Policy", "wfh"),
    "Insurance Policy (NT-HR-POL-006, v2026.1).pdf": ("Insurance Policy", "insurance"),
    "Travel Policy (NT-FIN-POL-001, v2026.1).pdf": ("Travel Policy", "travel"),
    "Employee Handbook (NT-HR-HBK-001, v2026.1).pdf": ("Employee Handbook", "handbook"),
    "Onboarding Documents (NT-HR-ONB-001, v2026.1).pdf": ("Onboarding Guide", "onboarding"),
}

TAG_LEVEL = {"ALL EMPLOYEES": 1, "MANAGER AND HR ONLY": 2, "HR ONLY": 3}
tag_re = re.compile(r"\[(ALL EMPLOYEES|MANAGER AND HR ONLY|HR ONLY)\]", re.I)
heading_re = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.]{2,100}$")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, add_start_index = True)

NOISE_RE = re.compile(r"^(Page \d+ of \d+|.*\(NT-[A-Z]+-[A-Z]+-\d+, v[\d.]+\))$", re.I)


def table_row_to_lines(row):
    cells = [(c or "").strip() for c in row]
    filled = [c for c in cells if c]
    if not filled:
        return []
    if len(filled) == 1:
        return [ln.strip() for ln in filled[0].split("\n") if ln.strip()]
    return [" | ".join(c.replace("\n", " ") for c in cells)]

def page_to_lines(page):
    tables = page.find_tables().tables
    table_rects = [fitz.Rect(t.bbox) for t in tables]

    lines = []
    # blocks: (x0, y0, x1, y1, text, block_no, block_type)
    for b in page.get_text("blocks"):
        rect = fitz.Rect(b[:4])
        if any(rect.intersects(tr) for tr in table_rects):
            continue                      
        for ln in b[4].split("\n"):
            if ln.strip():
                lines.append((b[1], ln.strip()))
    for t in tables:
        for row in t.extract():
            for text in table_row_to_lines(row):
                lines.append((t.bbox[1], text))       

    lines.sort(key=lambda x: x[0])        
    return [text for _, text in lines if not NOISE_RE.match(text)]

TAG_ONLY = re.compile(r"^\[(ALL EMPLOYEES|MANAGER AND HR ONLY|HR ONLY)\]$", re.I)


def merge_wrapped_tags(lines):
    out = []
    for line in lines:
        if out and "[" in out[-1] and "]" not in out[-1]:
            out[-1] = out[-1] + " " + line
        elif out and TAG_ONLY.match(line):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return out

def ingest_pdf(filename, doc_name, category):
    doc = fitz.open(POLICY_DIR / filename)
    chunks = []
    level = 1
    section = "Document Header"
    buffer = []

    def flush():
        nonlocal buffer
        keep = buffer and (section == "Document Header" or len(buffer) > 1)
        if keep:
            text = "\n".join(l for l,_ in buffer)
            starts, pos = [], 0
            for l, p in buffer:
                starts.append((pos, p))
                pos += len(l) + 1            # +1 = \n

            def page_at(offset):
                pg = starts[0][1]
                for s, p in starts:
                    if s <= offset:
                        pg = p
                    else:
                        break
                return pg
            for d in splitter.create_documents([text]):
                s = d.metadata["start_index"]
                e = s + len(d.page_content) -1
                chunks.append({
                    "text": f"{doc_name} > {section}\n{d.page_content}",
                    "meta": {
                        "doc_name": doc_name,
                        "category": category,
                        "page": page_at(s),
                        "end_page": page_at(e),
                        "section_heading": section,
                        "min_role_level": level,
                    },
                })
        buffer = []
    for page_no, page in enumerate(doc, start=1):
        for line in merge_wrapped_tags(page_to_lines(page)):
            if heading_re.match(line):
                flush()                         
                m = tag_re.search(line)
                if m:
                    level = TAG_LEVEL[m.group(1).upper()]
                elif re.match(r"^\d+\.\s",line):
                    print(f"  WARNING p{page_no}: {doc_name} > '{line}' doesn't have a valid tag (level {level} has been inherited)")
                section = tag_re.sub("", line).strip()
            buffer.append((line, page_no))
    flush()
    return chunks



if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(DB_PATH)
    try:
        client.delete_collection("hr_policies")
    except Exception:
        pass
    col = client.create_collection("hr_policies")

    all_chunks = []
    for fname, (name, cat) in DOCS.items():
        cs = ingest_pdf(fname, name, cat)
        print(f"{name}: {len(cs)} chunks")
        all_chunks.extend(cs)

    texts = [c["text"] for c in all_chunks]
    col.add(
        ids=[f"c{i}" for i in range(len(all_chunks))],
        documents=texts,
        embeddings=model.encode(texts, show_progress_bar=True).tolist(),
        metadatas=[c["meta"] for c in all_chunks],
    )

    print("\nTotal chunks:", len(all_chunks))
    for lvl in (1, 2, 3):
        n = sum(1 for c in all_chunks if c["meta"]["min_role_level"] == lvl)
        print(f"  min_role_level={lvl}: {n} chunks")
