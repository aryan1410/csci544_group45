# CSCI544 — NLP Project: LLM Evaluation : Group 45
- Aryan Shah (aryanutp@usc.edu)
- Nidhi Choudhary (nidhicho@usc.edu)
- Khushi Mehta (khushipr@usc.edu)
- Ishrit Chavan (ichavan@usc.edu)
- Aditya Sidham (sidham@usc.edu)

This project evaluates three approaches to factual question answering — a static LLM baseline, a retrieval-augmented generation (RAG) pipeline, and an agentic search system — measuring Token F1, hallucination rate, and latency on a shared question set drawn from real company annual reports (10-K filings) and trivia questions.

---

## System

- **OS:** Windows 11 / macOS (both supported)
- **Python:** 3.11
- **Hardware:** Standard laptop/desktop CPU (no GPU required)
- **Key APIs:** Groq (static LLM), OpenAI (RAG + agent), Tavily/SerpAPI (agent search)

---

## 1. Environment Setup

### 1.1 Clone the repository

```bash
git clone <repo-url>
cd CSCI544
```

### 1.2 Create and activate a virtual environment

**Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Configure API keys

Open a `.env` file at the project root and fill in your keys:

```
GROQ_API_KEY=your_groq_key_here        # required for static LLM
OPENAI_API_KEY=your_openai_key_here    # required for RAG and agent
TAVILY_API_KEY=your_tavily_key_here    # required for agent search
GEMINI_API_KEY=                        # optional — falls back to OpenAI if blank
SERP_API_KEY=                          # optional — enables dual search mode
```

All other values in `.env` (model name, search budgets, timeouts) can be left at their defaults from the example file

---

## 2. Running the Code

All three components share the same virtual environment and the same `.env` file at the root. Run each from its own subdirectory so that relative paths resolve correctly.

---

### 2.1 Static LLM — `static_llm/`

Evaluates a Groq-hosted Llama 3.3 70B model directly, with no retrieval.

**Run:**
```bash
cd static_llm
python llm_eval.py
```

**Output:** `static_llm/eval_results.csv`

The script reads questions from `data/qa_dataset.txt` and writes one result row per question to the CSV.

---

### 2.2 RAG — `rag/`

Builds a FAISS vector index over five company annual reports, then evaluates retrieval-augmented answers against the shared question set.

**Step 1 — Build the knowledge base (run once):**
```bash
cd rag
python knowledge_base_builder.py
```

This downloads TriviaQA and PopQA Q&A pairs from HuggingFace, loads the five company annual reports from `data/`, and embeds everything into a FAISS index at `rag/faiss_trivia_index/`. The TriviaQA/PopQA pairs give RAG coverage over `questions_subset.json`; the annual reports give it coverage over the custom Apple 10-K questions. This step takes several minutes and uses OpenAI embedding credits.

**Step 2 — Run evaluation:**
```bash
python batch_eval.py
```

**Output:** `rag/rag_results.csv`

> `batch_eval.py` must be run from the `rag/` directory because it imports `triviabot` as a local module.

---

### 2.3 Agent — `agent/`

A research agent that plans sub-queries, searches the web with Tavily/SerpAPI, and synthesizes a final answer. Evaluation sends questions to the live backend over HTTP.

**Step 1 — Start the backend (keep this terminal open):**
```bash
cd agent/backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

Wait until you see `Application startup complete` before proceeding.

**Step 2 — Run evaluation (open a new terminal, activate the venv):**
```bash
cd agent
python agent_eval.py
```

**Output:** `agent/agent_eval_results.csv`

> The eval script sends one question at a time and waits for a response (up to 120 s per question). The backend must be running on `localhost:8000` throughout.

---

## 3. How Results Are Generated

All three pipelines are evaluated on the same 50-question set and scored with the same metrics.

### Question set — `data/qa_dataset.txt`

All 50 questions are drawn from **Apple's FY2023 Annual Report (10-K)**, filed with the SEC for the fiscal year ended September 30, 2023. Questions are organized in three tiers designed to produce a measurable scoring gradient across the three pipelines:

| Tier | Questions | Examples | Expected winner |
| --- | --- | --- | --- |
| **Easily searchable** | Q1–15 | CEO name, ticker symbol, OS names, product names, headquarters city | All three (agent catches up here) |
| **Medium difficulty** | Q16–25 | Historical CFO, board chairman, distribution channel %, geographic rankings, segment growth trends | Static LLM (training data) and RAG |
| **Document-specific** | Q26–50 | Exact revenue figures, net income, EPS, CapEx, employee count, store count | RAG only |

The document-specific questions require exact figures in the format used in the filing (e.g., `200,583 million`). A web search typically returns rounded or reformatted values (e.g., `200.6 billion`) which score near zero after normalization, putting the agent at a disadvantage on this tier.

The medium-difficulty tier includes deliberate gotchas for a live web search — for example, Luca Maestri was Apple's CFO during FY2023 but left in January 2024. An agent searching today finds the current CFO (Kevan Parekh) and returns the wrong answer, while the static LLM (trained before the change) and RAG (reading the archived filing) both return the correct historical answer.

### Knowledge base — `data/` company filings

The RAG knowledge base is built from five company annual reports, deliberately chosen to force genuine retrieval rather than simple lookup:

| File | Company | Period |
| --- | --- | --- |
| `apple_fy2023_10k.txt` | Apple Inc. | FY2023 (ended Sep 30, 2023) |
| `microsoft_fy2023_10k.txt` | Microsoft Corporation | FY2023 (ended Jun 30, 2023) |
| `alphabet_fy2022_10k.txt` | Alphabet Inc. | FY2022 (ended Dec 31, 2022) |
| `meta_fy2022_10k.txt` | Meta Platforms, Inc. | FY2022 (ended Dec 31, 2022) |
| `amazon_fy2022_10k.txt` | Amazon.com, Inc. | FY2022 (ended Dec 31, 2022) |

The four non-Apple documents serve as distractors — they share vocabulary (revenue, segment, operating income, CEO) with Apple's filing, so the retriever must use semantic similarity to surface the correct Apple chunks rather than relying on keyword frequency alone. The `qa_dataset.txt` file is intentionally excluded from the knowledge base to prevent data leakage; all answers must come from the source filings.

### Metrics

| Metric | Definition |
| --- | --- |
| **Token F1** | SQuAD-style token overlap between predicted and ground-truth answer, after normalization (lowercase, strip articles and punctuation). |
| **Hallucination rate** | Fraction of answers with F1 below a threshold (0.2 for static LLM and agent; 0.4 for RAG). |
| **Latency** | Wall-clock seconds from request to complete answer. |

Normalization strips all punctuation including commas and decimal points, so `200,583 million` and `200583 million` are equivalent, but `200.6 billion` normalizes to `2006 billion` and does not match.

### Expected scoring gradient

| Pipeline | Strong on | Weak on | Expected Token F1 |
| --- | --- | --- | --- |
| **RAG** | Document-specific figures (retrieves exact text from the filing) | — | Highest |
| **Static LLM** | Easily searchable + medium (encoded in training data) | Exact figures not memorized | Middle |
| **Agent** | Easily searchable (live web search) | Document-specific figures (rounded/reformatted on the web) + historical personnel | Lowest |

### Pipeline details

**Static LLM** sends each question directly to `llama-3.3-70b-versatile` via the Groq API with a short-answer system prompt. No retrieval. Answers rely entirely on knowledge encoded during pre-training.

**RAG** embeds each question with OpenAI `text-embedding-ada-002`, retrieves the top-5 nearest chunks from the FAISS index, re-ranks with an embeddings filter (threshold τ = 0.45, keeping top 3), and passes the retained context to GPT-3.5-turbo with a strict short-answer prompt.

**Agent** uses a plan → search → synthesize loop. Given a question it generates sub-queries, issues up to `MAX_SEARCHES` web searches via Tavily (and optionally SerpAPI in dual-search mode), and synthesizes a final answer from the collected results using GPT-4o-mini.

Results for each pipeline are written to a CSV with one row per question containing the question, predicted answer, ground truth, F1 score, hallucination flag, and latency.
