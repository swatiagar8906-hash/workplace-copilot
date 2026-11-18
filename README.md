# Workplace Copilot

An AI assistant that answers employee questions from company HR documents (Leave, Work From Home, Insurance, Travel, Employee Handbook, Onboarding, HR Internal Guidelines) using **Retrieval-Augmented Generation (RAG)** with **section-level role-based access control**. An employee never receives text from a manager-only section, because that text is filtered out before it reaches the LLM.

> **Status:** 🚧 In progress. Ingestion, the role-aware retriever, the Groq-based generator (answers with citations) and a FastAPI backend (`/ask`) are built. A first evaluation set is running; login (JWT) and the UI are next. See the [Roadmap](#roadmap).

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

FastAPI: POST /ask (question + role) → the pipeline above → JSON answer (login is the next step)
```

---

## Project structure

```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app: /health and /ask
│   └── rag/
│       ├── __init__.py
│       ├── ingest.py        # PDFs → section-wise chunks → embeddings → ChromaDB
│       ├── retriever.py     # role-filtered search + access-control check
│       └── generator.py     # Groq LLM answer + code-built citations
├── data/
│   └── policies/            # 7 synthetic HR PDFs
├── eval/
│   ├── questions.json       # evaluation questions (lookup, manager-only, HR-only, restricted, out-of-scope)
│   └── run_eval.py          # runs every question several times and prints pass rates
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
| 7 | The new HR Internal Guidelines document put its whole **Table of Contents** into a level 1 chunk, including the titles of HR-only sections | Its TOC lines had dotted leaders (`3. Disciplinary Process ........ 3`), so they were not read as headings and ended up in the "Document Header" text | Lines with 4 or more dots in a row are dropped like page headers/footers (`NOISE_RE`). Found with a local script that prints every Document Header chunk |

### Effect on the data

| Run | Total chunks | Level 1 | Level 2 | Note |
|---|---|---|---|---|
| First run | 337 | 324 | 13 | Wrapped tags missed, manager sections leaked into level 1 |
| After tag-merge fix | 330 | 292 | 38 | Manager sections correct, but core Leave content wrongly in level 2 |
| After table/header fix | 356 | 325 | 31 | Manager sections at level 2, employee content at level 1 |
| After adding HR Internal Guidelines and filtering Table of Contents lines | **405** | **330** | **37** | First level 3 content: 38 chunks |

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
| HR Internal Guidelines | 5 | 6 (plus 38 at level 3) |

- Level 2 chunks come from the manager sections of the Leave (10.x, 11.x), Travel (15.x, 16.x) and WFH (13.x, 14.x) policies.
- Insurance, Onboarding and the Handbook have no manager-only sections in their body text, so all their chunks are level 1.
- The seventh document, **HR Internal Guidelines**, has the first `[HR ONLY]` sections (38 chunks at level 3), two manager sections (level 2) and two employee sections (level 1), so all three access levels are now in use.

**Checks I run after each ingestion** (local scripts, not committed): chunk counts per level, the list of manager sections with their level and page, and the text of any chunk that looks suspicious.

---

## Dataset

Seven synthetic documents for a fictional company (NovaTech Solutions Pvt. Ltd.), generated with an LLM using a shared "master context" prompt (so numbers and rules stay consistent) plus one prompt per document. They use numbered headings, tables, cross-references between policies, FAQs, and the section tags above. No real company or person is referenced.

---

## Getting started

```bash
git clone https://github.com/swatiagar8906-hash/workplace-copilot.git
cd workplace-copilot

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put the PDFs in [`data/policies/`](data/policies) (the file names are listed in the `DOCS` dictionary in `ingest.py`), then run from the project root:

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

To run the API:

```bash
python -m uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs. FastAPI builds this page from the code, so every endpoint can be tried there (see [API](#api-built)).

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

**A distance cut-off can detect off-topic questions, though.** I measured the distance of the closest chunk in the whole database (all roles) for different kinds of questions (`check_scope.py`, a local script):

| Kind of question | Closest-chunk distance |
|---|---|
| Policy questions from the evaluation set | 0.11 to 0.72 |
| Work-related but not in the policies ("Who is the CEO?", "Can I bring my dog to the office?", "Is there a gym?") | 0.50 to 0.82 |
| Clearly off-topic ("What is the weather today?", "What is 25 times 17?", "Tell me a joke") | 0.85 to 1.06 |

There is a gap between the policy questions and the off-topic ones, so a cut-off of **0.85** separates them on this set. This does not contradict the finding above: distance can say "this is about something else entirely", but it cannot say "this chunk answers the question". The margin is thin (25 times 17 is at 0.853, the farthest real question at 0.72) and the set is small, so the number is a constant in `generator.py` that can be changed after more testing.

**BGE query prefix.** I tried adding BGE's recommended query instruction (`"Represent this sentence for searching relevant passages: "`) before the question. On 4 test questions (asked as both roles) the top result stayed the same, some lower results got better and some worse, so it is switched off (`QUERY_PREFIX = ""`). I will re-test it on the golden evaluation set.

**Closest-chunk distance gap (why a margin rule does not fix R1).** For every employee-side question I compared the distance of the closest allowed chunk with the closest restricted chunk (`gap` = allowed distance minus restricted distance, so a positive gap means the restricted chunk is closer):

| Question | Type | Gap |
|---|---|---|
| R2, R3 (travel checks, WFH exceptions) | restricted | +0.08 |
| R4, R5 (salary band) | restricted | +0.07 |
| R6, R7 (compa-ratio) | restricted | +0.17 |
| R8 (manager rating deadline) | restricted | -0.05 |
| **R1** (when can a manager reject leave) | restricted | **-0.06** |
| L4 (HR acknowledgement time) | lookup | **-0.06** |
| M3 (manager ratings deadline) | manager-only | -0.07 |
| L3 (maternity WFH) | lookup | -0.32 |

Every question with a positive gap is a restricted question, so the current rule ("refuse if the closest chunk overall is restricted") has no false refusals in this set. But R1 and the lookup L4 have the same gap (-0.06), so no distance threshold can refuse R1 without also refusing L4. This matches the earlier finding that distance measures topic similarity, not whether a chunk answers the question. R1 needs a scorer that reads the question and the chunk together (a cross-encoder reranker) or an LLM answerability check.

**Bigger model did not help.** I ran the same 6 demo questions with `openai/gpt-oss-120b` instead of `openai/gpt-oss-20b`. For the employee asking the manager travel-checks question, 120b answered with employee-level text (video call, approval before booking) as if it were the manager's checks, where 20b correctly gave the no-answer message. 120b also gave a shorter manager checklist and added markdown bold. The employee "when can a manager reject" question stayed wrong on both. These are single runs, so they are not proof, but a bigger model is not a fix for the "related topic" problem; 20b stays the default. The golden evaluation set will compare models and prompts with numbers.

---

## Answer generation (built)

`app/rag/generator.py` takes the role-filtered chunks and asks an LLM (Groq, `openai/gpt-oss-20b` by default) to answer from them.

1. **Numbered context.** Chunks are passed as `[1]`, `[2]`, ... and the LLM must end its reply with `USED: 1, 3` (the passages it relied on).
2. **No-answer rule.** If the passages do not directly answer the question, the LLM must reply `NOT_FOUND`. A passage about a related topic does not count. Code then returns one fixed message, defined once in `retriever.py`: *"I couldn't find this in the policies available to your role. Please contact HR if you think you need this information."* The wording is generic on purpose, so it does not confirm that restricted content exists.
3. **Citations are built by code**, not written by the LLM. From the `USED:` numbers, the stored metadata gives document, section and pages. Chunks of the same section are merged into one citation, for example *Leave Policy, 10.3 When Leave May Be Rejected or Deferred, pp.14-15*.
4. **One place for the LLM call** (`call_llm`), so changing the provider means changing one function.
5. **Restricted-topic check.** Before calling the LLM, the code looks at the single closest chunk in the whole database (all levels). If that chunk is above the user's level, the question is about restricted content and the no-answer message is returned without calling the LLM. Only the chunk's level is read; its text is never used. It is behind a flag (`RESTRICTED_CHECK=0` turns it off) so the evaluation can compare before and after.
6. **Off-topic check.** The same closest-chunk lookup is used first: if even the closest chunk in the whole database is farther than 0.85, the question is not about company policies and a different message is returned: *"I can only answer questions about company policies, such as leave, work from home, travel or insurance etc."* This message never mentions roles or restricted content. A restricted question and a question that is simply not in the documents still get the same no-answer message, so a user cannot tell whether a hidden section exists. Only clearly off-topic questions are told apart, and they are decided on the whole database, so a restricted topic never counts as off-topic.

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
- The weather question returned the no-answer message at the time of this run. Since the off-topic check (item 6) it returns the off-topic message instead.
- **Still open:** for the employee version of "When can a manager reject a leave request?", the model answered with related level 1 text (the approval timeline). Nothing restricted leaked, but the answer does not match the question. This is the "related topic" failure the prompt tries to prevent, and a prompt alone does not fully stop it. A bigger model did not fix it (see Experiments). A reranker or an answerability check are the next things to try, measured on the golden evaluation set.

---

## API (built)

`app/main.py` exposes the pipeline with FastAPI:

| Method and path | What it does | Status codes |
|---|---|---|
| `GET /health` | Returns `{"status": "ok"}`. Tells you the server is running; it does not touch the database or the LLM | 200 |
| `POST /ask` | Body: `{"question": "...", "role": "employee"}`. Returns `{"answer": "...", "citations": [...]}` | 200, 400 (unknown role), 404 (off-topic question), 422 (missing or wrong fields, sent by FastAPI) |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the salary band for grade G5?", "role": "hr"}'
```

FastAPI also serves an interactive page at `/docs` and the machine-readable description at `/openapi.json`, both generated from the code.

**Why the status codes are what they are**

- **404 for off-topic questions** (`{"detail": "I can only answer questions about company policies..."}`). Nothing in any policy is related to the question.
- **200 for restricted questions.** A user asking about a section above their level gets the normal no-answer message with status 200, the same as a question that is not in the documents. A 403 would tell the user that a hidden section exists.
- **400 for an unknown role** (anything except `employee`, `manager`, `hr`).

**Temporary:** the role is sent in the request body, so anyone can claim to be HR. This is only for development. The next step is JWT login, so that the role comes from a signed token and the request no longer carries it.

---

## Evaluation (first run)

`python -m eval.run_eval` asks each question 3 times (the LLM answer changes between runs). Question types:

- **lookup / manager_only / hr_only:** the answer must contain given keywords and cite the expected section.
- **restricted:** the asking role must *not* get the answer, so the reply must be the no-answer message.
- **out_of_scope:** the off-topic message is expected for any role (before the off-topic check was added, the no-answer message was expected).
- **access leaks:** retrieval for every question is also checked for chunks above the asking role's level.

**Results (`openai/gpt-oss-20b`, 3 runs per question)**

| Type | Run 2: 17 questions, no restricted-topic check | Run 3: 20 questions, check on |
|---|---|---|
| lookup | 3/3 (100%) | 12/12 (100%) |
| manager_only | 7/9 (78%) | 8/9 (89%) |
| restricted | 18/24 (75%) | 21/24 (88%) |
| out_of_scope | 9/9 (100%) | 9/9 (100%) |
| hr_only | 6/6 (100%) | 6/6 (100%) |
| access leaks | 0 | 0 |

These runs were made before the off-topic check, so the out_of_scope row was measured against the no-answer message. The evaluation has to be run again with the new check.

(Run 1 is described below: it had an evaluation-script bug that made correct answers fail.) Run 3 added three employee lookup questions on purpose: they sit next to restricted sections, so they would show any false refusals from the new check. None occurred.

HR-only questions asked by an employee or a manager always got the no-answer message (salary band, compa-ratio threshold, manager ratings deadline).

**What the failures turned out to be**

- **Evaluation bug, fixed:** in the first run the model wrote `10 March` with a narrow no-break space, so my keyword match failed on a correct answer (M3, 3 failures; first run was manager_only 5/9 and restricted 19/24). The runner now normalises text, and M3 passes.
- **Incomplete answers (M2):** the manager travel-checks answer is correct and cites 15.2, but in 1 to 2 of 3 runs it left out the budget check.
- **Fixed by the restricted-topic check, R3:** the employee asking "Who approves exceptions to the WFH policy?" used to get an answer built from a level 1 special-cases section (Department Head, HR Business Partner, Head of HR, Finance, IT Security), presented as the general rule. The real approval matrix is in section 14.1, which is manager-only. After the check, R3 passes in all runs. I first marked this question as a wrong test; reading the PDF showed the question is right and the system was wrong.
- **Still failing, R1 (3 of 3 runs):** the employee asking "When can a manager reject a leave request?" gets the 2-working-day timeline from 5.3. The check does not catch it because the closest chunk to this question is the level 1 section 5.3 (distance 0.38), not the manager-only 10.3 (0.43). A distance margin cannot fix it either (see Experiments); a reranker is the planned fix.

Not every restricted failure has the same cause, so the remaining one needs a different fix (see Known limitations).

---

## Known limitations

- **Numbered notes can be detected as headings.** A line like `3. The employee must give at least 8 weeks' notice...` can look like a heading, so some section names in citations are long sentences. Access levels are not affected, because these lines inherit the level of the section above.
- **Default level is 1.** Text before the first tag is visible to everyone. A manager-only section that lost its tag would be exposed, so level counts and manager sections are checked after each ingestion.
- **Table of Contents entries look like headings.** Entries that stand alone are skipped, so they produce no chunks. Tags written in the Table of Contents are ignored for the same reason.
- **Page-wide table detection.** `find_tables()` can return a region that also covers normal text; this is handled by splitting single-cell rows, but other PDFs may behave differently.
- **PyMuPDF block grouping is heuristic**, so blocks are split into lines and headings are found with a regex rather than by block boundaries.
- **No relevance cut-off.** When the user's role has no chunk that really answers a question, the retriever still returns the closest allowed chunks (see Experiments). The generator prompt has to refuse in that case.
- **The LLM can answer from related allowed text.** For a restricted question, an employee may get an answer built from nearby level 1 content instead of the no-answer message. This is not a leak (the text is allowed), but it can be misleading. A reranker or an answerability check, measured on the golden set, is the planned mitigation.
- **No authentication yet.** The role is sent by the client in the request body, so it can be faked. JWT login is the next step.
- **The off-topic cut-off is one fixed number (0.85).** It was measured on about 30 questions and the margin is thin. A real policy question that is worded very differently from the documents could be called off-topic and get the off-topic message.
- **Free-tier Groq limits and model churn.** Rate limits apply and model names change, so the model is configurable through `GROQ_MODEL`.
- **Answers vary slightly between runs.** The same question can return a slightly different list (for example 8 checklist points in one run and 6 in another), so the evaluation set has to be run several times.
- **Small evaluation set.** 20 questions so far; pass rates on a set this small move a lot with one question. The plan is 30-40 questions.
- **Some numbered notes become section names in citations.** For example a WFH citation reads `4. Approval is needed from the Department Head, HR Business Partner, Head of HR,`. Access levels are not affected, but citations look untidy.

---

## Roadmap

- [x] Synthetic HR document set (7 PDFs)
- [x] Ingestion: section-wise chunking, table extraction, role tags, page ranges
- [x] Verify ingestion output (chunk counts, manager sections, levels)
- [x] Retriever with role filter (terminal)
- [x] Access control test (employee vs manager on the same question)
- [x] Generator (Groq) with code-built citations and a no-answer rule (one known miss, see Answer generation)
- [x] Add an HR-only document to demonstrate level 3
- [x] First evaluation set and runner (20 questions, each run 3 times)
- [x] Off-topic detection with its own message (cut-off measured, see Experiments)
- [x] FastAPI backend: `/health` and `/ask` (the role is sent by the client for now)
- [ ] Grow the evaluation set to 30-40 questions (more lookups, multi-document questions)
- [ ] JWT login so the role comes from a token, and admin upload
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
- Why an evaluation set is needed: a failing test can be a bug in the system, in the test, or just run-to-run noise, and the failures have to be read before trusting the score
- Adding a document with HR-only sections to test all three access levels, and checking that a lower role never gets a higher role's answer
- Measuring before choosing a threshold: the same distance that cannot judge "is there an answer" can detect an off-topic question
- Designing responses so they do not leak: a restricted question and a missing one get the same reply, and only clearly off-topic questions are told apart
- Basics of a FastAPI backend: endpoints, request bodies, the auto-generated `/docs` page, and what HTTP status codes (200, 400, 404, 422) mean

---

## License

MIT, see [LICENSE](LICENSE).