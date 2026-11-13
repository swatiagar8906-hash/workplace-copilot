# Workplace Copilot

An AI assistant that answers employee questions from company HR documents (Leave, Work From Home, Insurance, Travel, Employee Handbook, Onboarding) using **Retrieval-Augmented Generation (RAG)** with **section-level role-based access control**. An employee never receives text from a manager-only section, because that text is filtered out before it reaches the LLM.

> **Status:** 🚧 In progress. Ingestion, the role-aware retriever and the Groq-based generator (answers with citations) are built. Evaluation, auth and UI are next. See the [Roadmap](#roadmap).

---

## Why this goes beyond a basic RAG app

A basic RAG app loads PDFs, splits them into chunks and searches. HR documents need more:

- **Not every section is for every employee.** Sections are tagged `[ALL EMPLOYEES]`, `[MANAGER AND HR ONLY]` or `[HR ONLY]`. A manager-only chunk must never reach an employee, even if the question is worded cleverly.
- **Policy numbers live in tables** (leave days, claim limits, hotel caps). Naive text extraction breaks a table into disconnected cells, so "8 days" gets separated from "Casual Leave" and the LLM can answer wrongly.
- **Answers must be verifiable**, so every answer should point to its source (document, section, page).

---

## Architecture

```
PDFs → PyMuPDF (text + tables) → clean lines → heading + tag detection
     → section-wise chunks with metadata → embeddings (bge-small-en-v1.5) → ChromaDB

User (role) → question → retriever (filter: min_role_level <= user level)
            → LLM via Groq (answers only from numbered passages, or NOT_FOUND)
            → answer + citations built by code from chunk metadata
```

---

## Project structure

```
├── app/
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py
│       ├── ingest.py        # PDFs → section-wise chunks → embeddings → ChromaDB
│       ├── retriever.py     # role-filtered search + access-control check
│       └── generator.py     # Groq LLM answer + code-built citations
├── data/
│   └── policies/            # 6 synthetic HR PDFs
├── eval/                    # golden Q&A set (planned)
├── ui/                      # Streamlit app (planned)
├── .env                     # GROQ_API_KEY (not committed)
├── try_models.py            # lists the Groq models your API key can use
├── requirements.txt
└── README.md
```

---

## Key design decisions

### 1. Access control is enforced at retrieval time, not in the prompt
Every chunk is stored with a `min_role_level`. At query time the vector database only returns chunks the user's role may see, so unauthorized text never reaches the LLM. Telling an LLM "don't reveal manager-only information" is prompt-level security and can be bypassed.

| Tag in document | `min_role_level` | Visible to |
|---|---|---|
| `[ALL EMPLOYEES]` | 1 | employee, manager, HR |
| `[MANAGER AND HR ONLY]` | 2 | manager, HR |
| `[HR ONLY]` | 3 | HR |

A user can see a chunk when `chunk.min_role_level <= user_level` (employee = 1, manager = 2, HR = 3).

### 2. Chunking follows sections, not pages
A page can contain both employee and manager sections, and one section can run over several pages. So the unit of chunking is a **section**:

- Tags appear only on main headings (`5. Approval Guidelines [MANAGER AND HR ONLY]`).
- Sub-headings (`5.1`, `5.2`) have no tag and **inherit the level of the section above**, until the next tag appears.
- A new heading closes the current section and starts a new one. Each section is then split into chunks of 800 characters with 150 overlap.

A chunk therefore never mixes two sections or two access levels.

### 3. Table-aware extraction
`page.get_text()` returns every table cell on its own line. The pipeline uses PyMuPDF's `find_tables()` to read tables as rows, skips the broken text blocks that sit inside a table's bounding box, and puts each row back in its correct position on the page as `Casual Leave (CL) | 8 days | All employees | ...`.

### 4. Citations at chunk level
Each chunk stores the page where it starts (`page`) and where it ends (`end_page`). The citation is built by code from stored metadata, not written by the LLM, so it cannot be invented.

### 5. PyMuPDF directly, not a LangChain loader
The access-level logic is my own code and works with any loader. I used PyMuPDF directly because I needed `find_tables()` and the position of each text block (for tables, ordering and header/footer removal). LangChain is still used for the text splitter.

---

## How ingestion works (`app/rag/ingest.py`)

| Step | Function | What it does |
|---|---|---|
| 1 | `page_to_lines(page)` | Reads a page as text blocks with positions, removes text inside tables, adds table rows back, sorts top to bottom, drops page headers/footers |
| 2 | `table_row_to_lines(row)` | Turns a table row into `a \| b \| c`; empty rows are dropped; rows with a single filled cell are treated as text and split into lines |
| 3 | `merge_wrapped_tags(lines)` | Rejoins a tag that wrapped onto the next line (`[MANAGER AND` + `HR ONLY]`) |
| 4 | `ingest_pdf(...)` | Walks the lines. A heading closes the previous section (`flush()`), updates the section name and, if the heading has a tag, the access level. Every other line goes into the current section |
| 5 | `flush()` | Splits the finished section into chunks, attaches metadata, and maps each chunk's start/end position back to a page |
| 6 | main block | Embeds all chunks with `BAAI/bge-small-en-v1.5` and stores them in ChromaDB |

Metadata stored with every chunk:

```python
{
  "doc_name": "Leave Policy",
  "category": "leave",
  "page": 14,
  "end_page": 15,
  "section_heading": "10.3 When Leave May Be Rejected or Deferred",
  "min_role_level": 2,
}
```

The two patterns that drive the logic:

```python
tag_re     = re.compile(r"\[(ALL EMPLOYEES|MANAGER AND HR ONLY|HR ONLY)\]", re.I)
heading_re = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.]{2,100}$")
```

`heading_re` accepts `2.`, `2.1` or `3.4.2` followed by a capital letter, a short text without a full stop, and the end of the line. This stops normal sentences that start with a number (`30 days notice is required...`) from being read as headings.

---

## Problems I ran into and how I fixed them

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | Table cells came out as separate lines (`Casual Leave`, `(CL)`, `8 days`) | `get_text()` has no idea about rows and columns | Use `find_tables()`, skip text blocks that overlap a table's bounding box, re-insert rows as `a \| b \| c` |
| 2 | A section that runs over two pages was split by page | Early version flushed at the end of every page | Sections are closed only when a new heading appears, so page boundaries do not matter |
| 3 | Citation showed the page where a section starts, not where the answer is | One page number was stored per section | Store the page of every line, and compute `page` / `end_page` per chunk from its start and end position (`add_start_index=True`) |
| 4 | Normal lines starting with a number looked like headings | The first heading regex only checked "number + text" | Stricter `heading_re` (capital letter, no full stop, length limit) |
| 5 | **Manager sections were stored at level 1** (a real access-control leak) in the Leave and Travel policies | Long headings wrap in the PDF, so the tag was split across two lines and never matched | `merge_wrapped_tags` joins the tag back before headings are read. Found by listing every manager section with its level after ingestion |
| 6 | **Core Leave Policy content (purpose, definitions, entitlement table) was stored as level 2** under the name "14. Revision History" | On some pages `find_tables()` returned a region covering headings and paragraphs, so they were glued into one long line and never recognised as headings. The Table of Contents had also left a manager tag "stuck" on the last entry, and that level stayed on until the next real heading | `table_row_to_lines` splits single-cell rows back into lines; page headers/footers (`Page 2 of 18`, document id) are filtered out. Found by printing the text of the suspicious chunks |

### Effect on the data

| Run | Total chunks | Level 1 | Level 2 | Note |
|---|---|---|---|---|
| First run | 337 | 324 | 13 | Wrapped tags missed, manager sections leaked into level 1 |
| After tag-merge fix | 330 | 292 | 38 | Manager sections correct, but core Leave content wrongly in level 2 |
| After table/header fix | **356** | **325** | **31** | Manager sections at level 2, employee content at level 1 |

---

## Current results

| Document | Level 1 chunks | Level 2 chunks |
|---|---|---|
| Leave Policy | 61 | 11 |
| Travel Policy | 81 | 10 |
| WFH Policy | 70 | 10 |
| Insurance Policy | 34 | 0 |
| Onboarding Guide | 36 | 0 |
| Employee Handbook | 43 | 0 |

- Level 2 chunks come from the manager sections of the Leave (10.x, 11.x), Travel (15.x, 16.x) and WFH (13.x, 14.x) policies.
- Insurance, Onboarding and the Handbook have no manager-only sections in their body text, so all their chunks are level 1.
- No `[HR ONLY]` section exists yet, so level 3 is empty. I plan to add one more document with HR-only sections to demonstrate all three levels.

**Checks I run after each ingestion** (local scripts, not committed): chunk counts per level, the list of manager sections with their level and page, and the text of any chunk that looks suspicious.

---

## Dataset

Six synthetic documents for a fictional company (NovaTech Solutions Pvt. Ltd.), generated with an LLM using a shared "master context" prompt (so numbers and rules stay consistent) plus one prompt per document. They use numbered headings, tables, cross-references between policies, FAQs, and the section tags above. No real company or person is referenced.

---

## Getting started

```bash
git clone https://github.com/swatiagar8906-hash/workplace-copilot.git
cd workplace-copilot

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put the PDFs in `data/policies/` (the file names are listed in the `DOCS` dictionary in `ingest.py`), then run from the project root:

```bash
python -m app.rag.ingest
```

This prints a chunk count per document, the total, and the number of chunks per access level. The vector database is created in `./db`.

Then try the retriever. It asks the same questions as an employee and as a manager, and finishes with an access check:

```bash
python -m app.rag.retriever
```

Finally the generator. It needs a free [Groq](https://console.groq.com) API key. Create a `.env` file in the project root (it is git-ignored):

```
GROQ_API_KEY=your_key_here
# optional, default is openai/gpt-oss-20b
# GROQ_MODEL=openai/gpt-oss-20b
```

Write each line exactly as `KEY=value`, with no spaces around `=` and no quotes.

```bash
python -m app.rag.generator
```

It asks 6 demo questions (general, manager-only as employee and as manager, out-of-scope) and prints each answer with its sources. Groq model names change over time. If you get a `model_not_found` error, run `python try_models.py` to list the models your key can use, then set `GROQ_MODEL` in `.env`.

---

## Retrieval and access control (built)

`app/rag/retriever.py` embeds the question with the same model used at ingestion and queries ChromaDB with a role filter:

```python
res = col.query(
    query_embeddings=q_emb,
    n_results=k,
    where={"min_role_level": {"$lte": ROLE_LEVEL[role]}},
)
```

Chunks above the user's level are excluded inside the database, so they never reach the LLM.

**Test: same question, different role**

| Question | Employee gets | Manager gets |
|---|---|---|
| When can a manager reject a leave request? | only level 1 (5.3 Approval Timelines, 5.5 Cancellation) | level 1 plus level 2 (10.3 When Leave May Be Rejected or Deferred) |
| What checks should a manager do before approving a travel request? | only level 1 (2.1 Basic Rules, ...) | level 2: 15.2, 15.3, 15.1 |
| Who approves exceptions to the WFH policy? | only level 1 (sections 4, 8, 2.3) | level 2: 14.1 Approval of Exceptions first |

An automatic check asks the 3 manager-type questions as an employee (20 results each) and counts chunks above level 1. Result: **ACCESS CHECK: PASSED, leaks = 0**.

---

## Experiments and findings

**A distance cut-off cannot decide "is there an answer?"**
ChromaDB returns a distance with every chunk (lower means closer). I looked for a limit that separates good questions from bad ones, but the numbers do not allow it:

| Question | Best distance |
|---|---|
| "How many casual leaves do I get in a year?" (answer exists) | 0.72 |
| Employee asking manager-only questions (answer filtered out) | 0.38 to 0.51 |

The question that has the right answer scores worse than the ones that do not. Distance measures how similar the wording and topic look (paragraph chunks look closer to a sentence than a table row does), not whether a chunk contains the answer. A fixed threshold would block the good question and let the others through, so it is not used. The ranking itself is right: the correct chunk (3.1 Casual Leave) came first.

**BGE query prefix.** I tried adding BGE's recommended query instruction (`"Represent this sentence for searching relevant passages: "`) before the question. On 4 test questions (asked as both roles) the top result stayed the same, some lower results got better and some worse, so it is switched off (`QUERY_PREFIX = ""`). I will re-test it on the golden evaluation set.

**Bigger model did not help.** I ran the same 6 demo questions with `openai/gpt-oss-120b` instead of `openai/gpt-oss-20b`. For the employee asking the manager travel-checks question, 120b answered with employee-level text (video call, approval before booking) as if it were the manager's checks, where 20b correctly gave the no-answer message. 120b also gave a shorter manager checklist and added markdown bold. The employee "when can a manager reject" question stayed wrong on both. These are single runs, so they are not proof, but a bigger model is not a fix for the "related topic" problem; 20b stays the default. The golden evaluation set will compare models and prompts with numbers.

---

## Answer generation (built)

`app/rag/generator.py` takes the role-filtered chunks and asks an LLM (Groq, `openai/gpt-oss-20b` by default) to answer from them.

1. **Numbered context.** Chunks are passed as `[1]`, `[2]`, ... and the LLM must end its reply with `USED: 1, 3` (the passages it relied on).
2. **No-answer rule.** If the passages do not directly answer the question, the LLM must reply `NOT_FOUND`. A passage about a related topic does not count. Code then returns one fixed message, defined once in `retriever.py`: *"I couldn't find this in the policies available to your role. Please contact HR if you think you need this information."* The wording is generic on purpose, so it does not confirm that restricted content exists.
3. **Citations are built by code**, not written by the LLM. From the `USED:` numbers, the stored metadata gives document, section and pages. Chunks of the same section are merged into one citation, for example *Leave Policy, 10.3 When Leave May Be Rejected or Deferred, pp.14-15*.
4. **One place for the LLM call** (`call_llm`), so changing the provider means changing one function.

**Test run (6 demo questions)**

| Question | Role | Result |
|---|---|---|
| How many casual leaves do I get in a year? | employee | correct: 8 days, source Leave Policy 2.1 |
| When can a manager reject a leave request? | employee | **wrong kind of answer**: it gave the 2-working-day decision timeline from 5.3 (level 1) instead of the no-answer message |
| Same question | manager | correct, from 10.3 When Leave May Be Rejected or Deferred |
| What checks should a manager do before approving a travel request? | employee | no-answer message (correct) |
| Same question | manager | correct bullet list from 15.2, no table |
| What is the weather today? | employee | no-answer message (correct) |

Access control held at answer level: employee answers cited only level 1 sections, and no manager-only text appeared.

**Quality problems found and what happened after the prompt changes**

- The employee travel question used to describe the employee's own checklist as the manager's checks. After telling the model not to change who a passage is about, it now returns the no-answer message.
- The table with `|` characters is gone (plain bullets now).
- The weather question now returns the no-answer message.
- **Still open:** for the employee version of "When can a manager reject a leave request?", the model answered with related level 1 text (the approval timeline). Nothing restricted leaked, but the answer does not match the question. This is the "related topic" failure the prompt tries to prevent, and a prompt alone does not fully stop it. A bigger model did not fix it (see Experiments). A reranker or an answerability check are the next things to try, measured on the golden evaluation set.

---

## Known limitations

- **Numbered notes can be detected as headings.** A line like `3. The employee must give at least 8 weeks' notice...` can look like a heading, so some section names in citations are long sentences. Access levels are not affected, because these lines inherit the level of the section above.
- **Default level is 1.** Text before the first tag is visible to everyone. A manager-only section that lost its tag would be exposed, so level counts and manager sections are checked after each ingestion.
- **Table of Contents entries look like headings.** Entries that stand alone are skipped, so they produce no chunks. Tags written in the Table of Contents are ignored for the same reason.
- **Page-wide table detection.** `find_tables()` can return a region that also covers normal text; this is handled by splitting single-cell rows, but other PDFs may behave differently.
- **PyMuPDF block grouping is heuristic**, so blocks are split into lines and headings are found with a regex rather than by block boundaries.
- **No relevance cut-off.** When the user's role has no chunk that really answers a question, the retriever still returns the closest allowed chunks (see Experiments). The generator prompt has to refuse in that case.
- **The LLM can answer from related allowed text.** For a restricted question, an employee may get an answer built from nearby level 1 content instead of the no-answer message. This is not a leak (the text is allowed), but it can be misleading. A reranker or an answerability check, measured on the golden set, is the planned mitigation.
- **Free-tier Groq limits and model churn.** Rate limits apply and model names change, so the model is configurable through `GROQ_MODEL`.
- **Answers vary slightly between runs.** The same question can return a slightly different list (for example 8 checklist points in one run and 6 in another), so the evaluation set has to be run several times.
- **No evaluation yet.** Answer quality has only been checked on a handful of demo questions.

---

## Roadmap

- [x] Synthetic HR document set (6 PDFs)
- [x] Ingestion: section-wise chunking, table extraction, role tags, page ranges
- [x] Verify ingestion output (chunk counts, manager sections, levels)
- [x] Retriever with role filter (terminal)
- [x] Access control test (employee vs manager on the same question)
- [x] Generator (Groq) with code-built citations and a no-answer rule (one known miss, see Answer generation)
- [ ] Add an HR-only document to demonstrate level 3
- [ ] Golden evaluation set (30-40 questions: lookups, multi-document, role-restricted, out-of-scope)
- [ ] FastAPI backend with JWT login and admin upload
- [ ] Streamlit chat UI with sources
- [ ] Hybrid search (BM25 + vector), reranker, query rewriting, conversation memory
- [ ] RAGAS evaluation (before/after comparison)
- [ ] Docker, deployment, demo video

---

## What I learned building this

- Why access control belongs in the retrieval filter and not in the prompt
- How a PDF stores text (blocks, bounding boxes, coordinates) and why tables break
- The difference between a LangChain loader and using PyMuPDF directly
- Designing metadata so that one chunk carries everything needed for filtering and citations
- Regular expressions for headings and tags (`match` vs `search`, groups, `re.sub`)
- Why a section is a better chunking unit than a page, and how a level is inherited until the next tag
- Debugging by inspection: when the counts looked wrong, I printed the sections and the text of suspicious chunks instead of guessing
- Embedding distance measures topical similarity, not whether a chunk answers the question. Ranking is useful, a fixed score cut-off is not
- Filtering by metadata inside the vector database (`where` with `$lte`) as the access-control layer
- How to test a security property: the same question under two roles, plus an automatic leak count
- Running a small experiment (query prefix) and keeping the simpler option when the data shows no clear gain
- Why citations should be built by code from metadata and not written by the LLM
- Using a sentinel token (`NOT_FOUND`) plus a `USED:` line to make LLM output easy to parse
- Defining a message once (`NO_ANSWER`) and importing it, so the retriever and generator never disagree
- Access control at retrieval is necessary but not sufficient: the LLM can still misread allowed text, which is why evaluation matters
- Keeping the LLM call in one function and the model in an environment variable, because provider model names change

---

## License

MIT, see [LICENSE](LICENSE).