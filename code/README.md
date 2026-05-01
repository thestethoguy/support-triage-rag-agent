# Support Triage Agent — `code/`

A terminal-based RAG agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** using a local ChromaDB vector store.

---

## Architecture

```
code/
├── ingest.py       # One-time corpus ingestion → ChromaDB
├── router.py       # Classification & routing logic (TriageDecision)
├── generator.py    # Strict RAG retrieval + response generation (Phase 3)
├── main.py         # Batch orchestrator: reads CSV, writes output.csv
├── requirements.txt
├── .env.example
└── README.md       ← you are here
```

**Pipeline overview:**

```
support_tickets.csv
        │
        ▼
  main.py  ──row──►  router.py  (classify + routing decision)
                         │
               ┌─────────┴──────────┐
           ESCALATE               REPLY
               │                    │
     escalation_reason       generator.py
               │           (RAG retrieval + LLM)
               │                    │
               │           [ESCALATE] sentinel?
               │            Yes → override status
               │            No  → use response
               └─────────┬──────────┘
                         │
                   output.csv  +  log.txt
```

---

## Prerequisites

- Python 3.11+
- An OpenAI API key (for embeddings — `text-embedding-3-small` — and the default LLM)
- Optionally an Anthropic API key if you want to use Claude as the LLM

---

## Installation

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
copy .env.example .env          # Windows
# or: cp .env.example .env     # macOS / Linux
# Then edit .env and add your OPENAI_API_KEY
```

---

## Step 1 — Ingest the corpus

Run **once** (or with `--reset` to rebuild):

```bash
# From the code/ directory:
python ingest.py

# Full rebuild from scratch:
python ingest.py --reset

# Custom paths:
python ingest.py --data-dir ../data --db-dir ./chroma_db --chunk-size 800

# Help:
python ingest.py --help
```

This reads all 773 `.md` files from `data/`, splits them into 800-character
overlapping chunks, embeds with OpenAI `text-embedding-3-small`, and saves
the ChromaDB to `./chroma_db`.

**Expected output:**
```
── Step 1 · Load documents ──────────────────────────────────────
  Found 773 markdown files under ..\data
  Loading docs: 100%|████████| 773/773 [00:02<00:00, 350file/s]

── Step 2 · Split into chunks ───────────────────────────────────
  Split 773 docs → ~3500 chunks (size≤800, overlap=120)

── Step 3 · Embed & persist to ChromaDB ─────────────────────────
  Embedding 3500 chunks → .\chroma_db  (this may take a minute…)
  Embedding batches: 100%|████████| 35/35 [01:20<00:00,  2.3s/it]
  ✓ ChromaDB persisted — collection 'support_corpus' contains 3500 vectors

✅  Ingestion complete.
```

---

## Step 2 — Run the agent

```bash
python main.py
```

Output is written to `../support_tickets/output.csv`.

Optional flags:

```
--tickets   path to support_tickets.csv   (default: ../support_tickets/support_tickets.csv)
--output    path to output.csv            (default: ../support_tickets/output.csv)
--db-dir    path to ChromaDB              (default: ./chroma_db)
--provider  openai | anthropic            (default: openai)
--model     LLM model name               (default: gpt-4o-mini)
--verbose   print per-ticket details to stdout
```

---

## Routing logic summary (`router.py`)

| Signal | Action |
|---|---|
| Fraud / stolen card / identity theft | Escalate |
| Prompt injection / system-prompt leak attempt | Escalate |
| Billing dispute / refund demand | Escalate |
| Security vulnerability / account takeover | Escalate |
| Legal / compliance / vendor forms | Escalate |
| Low LLM routing confidence (< 0.65) | Escalate defensively |
| Clear FAQ answerable from corpus | Reply |
| Out-of-scope / invalid harmless request | Reply (explain scope) |

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | Used for embeddings + default LLM |
| `ANTHROPIC_API_KEY` | No | — | Required if `LLM_PROVIDER=anthropic` |
| `LLM_PROVIDER` | No | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name |
| `EMBED_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |

---

## Determinism

- Temperature is set to `0` for all LLM calls.
- ChromaDB uses a fixed collection name and deterministic similarity ranking.
- No random sampling anywhere in the pipeline.
