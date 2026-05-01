"""
main.py — Batch orchestrator for the HackerRank Orchestrate triage agent.

Reads support_tickets.csv, runs every ticket through:
  Router  (classify + route decision)
    └─ REPLY   → Generator (RAG retrieval + LLM response)
                  └─ [ESCALATE] sentinel → override to escalated
    └─ ESCALATE → use escalation_reason directly

Writes:
  • support_tickets/output.csv   — HackerRank prediction format
  • log.txt (per AGENTS.md §5)   — structured chat transcript

Usage:
    python main.py [--tickets PATH] [--output PATH] [--db-dir PATH]
                   [--model NAME] [--verbose]
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ── Load .env (code/ dir first, then repo root) ───────────────────────────────
_HERE      = Path(__file__).parent
_REPO_ROOT = _HERE.parent

load_dotenv(dotenv_path=_HERE / ".env",       override=False)
load_dotenv(dotenv_path=_REPO_ROOT / ".env",  override=False)

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from router    import TicketRouter, TriageDecision
from generator import ResponseGenerator, ESCALATE_SENTINEL

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TICKETS     = _REPO_ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT      = _REPO_ROOT / "support_tickets" / "output.csv"
DEFAULT_DB_DIR      = _HERE / "qdrant_db_v4"
COLLECTION_NAME     = "support_corpus"

# AGENTS.md §2 — log file lives outside the repo in the user's home dir
_LOG_DIR  = Path.home() / "hackerrank_orchestrate"
_LOG_FILE = _LOG_DIR / "log.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers  (AGENTS.md §5.2 format)
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_log_dir() -> None:
    """Create the log directory if it doesn't exist."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    """ISO-8601 timestamp in local time with offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_ticket_log(
    ticket_id: int,
    user_text: str,
    decision: TriageDecision,
    docs_retrieved: int,
    final_response: str,
    action: str,           # "REPLY" | "ESCALATE"
) -> None:
    """Append one ticket's structured block to log.txt.

    Format required by Phase 4 spec:
    ========================================
    TICKET ID: …
    USER TEXT: …
    -- ROUTER DECISION --
    …
    -- RAG RETRIEVAL --
    …
    -- AGENT RESPONSE --
    …
    ========================================
    """
    preview = user_text.strip().replace("\n", " ")[:100]
    if len(user_text.strip()) > 100:
        preview += "..."

    block = (
        "\n"
        "========================================\n"
        f"TICKET ID: {ticket_id}\n"
        f"USER TEXT: {preview}\n"
        "-- ROUTER DECISION --\n"
        f"Product Area: {decision.product_area}\n"
        f"Request Type: {decision.request_type}\n"
        f"Action: {action}\n"
        "-- RAG RETRIEVAL --\n"
        f"Documents Retrieved: {docs_retrieved}\n"
        "-- AGENT RESPONSE --\n"
        f"{final_response.strip()}\n"
        "========================================\n"
    )

    try:
        _ensure_log_dir()
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(block)
    except Exception as exc:
        print(f"  [WARN] Could not write to log: {exc}", file=sys.stderr)


def _append_session_log(tickets_path: Path, provider: str, model: str) -> None:
    """Write a SESSION START entry (AGENTS.md §5.1)."""
    try:
        _ensure_log_dir()
        import subprocess
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(_REPO_ROOT), text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            branch = "unknown"

        entry = (
            f"\n## {_ts()} SESSION START\n\n"
            f"Agent: Antigravity\n"
            f"Repo Root: {_REPO_ROOT}\n"
            f"Branch: {branch}\n"
            f"Worktree: main\n"
            f"Parent Agent: none\n"
            f"Language: py\n"
            f"LLM: {provider}/{model}\n"
            f"Tickets: {tickets_path}\n"
        )
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass   # logging must never crash the agent


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm(model: str):
    """Instantiate a deterministic (temperature=0) Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        max_output_tokens=1024,
        # google_api_key picked up automatically from GOOGLE_API_KEY env var
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core processing logic
# ─────────────────────────────────────────────────────────────────────────────


def _process_ticket(
    ticket_id:  int,
    issue:      str,
    subject:    str,
    company:    str,
    router:     TicketRouter,
    generator:  ResponseGenerator,
    verbose:    bool = False,
) -> dict:
    """Run one ticket through Router → (Generator?) and return output row dict."""

    # Build a unified ticket text for the generator query
    ticket_text = "\n".join(
        p for p in [f"Subject: {subject}", f"Issue: {issue}"] if p.strip()
    )

    # ── Step 1: Router ────────────────────────────────────────────────────────
    decision: TriageDecision = router.route(
        issue=issue, subject=subject, company=company
    )

    if verbose:
        print(
            f"\n  #{ticket_id:02d} [{decision.status.upper()}] "
            f"area={decision.product_area} type={decision.request_type} "
            f"conf={decision.confidence:.0%}"
        )

    # ── Step 2: Route decision ────────────────────────────────────────────────
    docs_retrieved = 0
    final_action   = "ESCALATE"
    final_response: str

    if decision.status == "escalated":
        # Router says ESCALATE — use the router's reason as the response
        final_response = (
            decision.escalation_reason
            or "This issue has been escalated to our support team for review."
        )
        final_action   = "ESCALATE"

    else:
        # Router says REPLY — call the Generator
        gen_result = generator.generate(
            ticket_text=ticket_text,
            product_area=decision.product_area,
            company=company,
        )
        docs_retrieved = gen_result.docs_retrieved

        if gen_result.escalated:
            # Generator's [ESCALATE] sentinel → override
            final_response = (
                "This request requires human review. "
                "Our support team will follow up shortly."
            )
            final_action   = "ESCALATE"
            decision       = decision.model_copy(update={"status": "escalated"})
            if verbose:
                print(f"    ↳ Generator self-escalated: {gen_result.response[:80]}")
        else:
            final_response = gen_result.response
            final_action   = "REPLY"

    # ── Step 3: Log the ticket ────────────────────────────────────────────────
    _append_ticket_log(
        ticket_id=ticket_id,
        user_text=f"{subject or ''} {issue or ''}",
        decision=decision,
        docs_retrieved=docs_retrieved,
        final_response=final_response,
        action=final_action,
    )

    # ── Step 4: Build output row ──────────────────────────────────────────────
    # Map internal action → HackerRank status vocabulary
    hr_status = "replied" if final_action == "REPLY" else "escalated"

    # justification: concise, traceable
    justification_parts = [
        f"Classified as '{decision.request_type}' in '{decision.product_area}' "
        f"(routing confidence: {decision.confidence:.0%})."
    ]
    if hr_status == "escalated":
        justification_parts.append(
            decision.escalation_reason
            or "Escalated due to sensitivity or insufficient corpus coverage."
        )
    else:
        justification_parts.append(
            f"Response grounded in {docs_retrieved} corpus chunk(s)."
        )

    return {
        "status":        hr_status,
        "product_area":  decision.product_area,
        "response":      final_response,
        "justification": " ".join(justification_parts),
        "request_type":  decision.request_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the support triage agent on support_tickets.csv."
    )
    p.add_argument("--tickets",  type=Path, default=DEFAULT_TICKETS,
                   help=f"Input CSV (default: {DEFAULT_TICKETS})")
    p.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output CSV (default: {DEFAULT_OUTPUT})")
    p.add_argument("--db-dir",   type=Path, default=DEFAULT_DB_DIR,
                   help=f"Qdrant store directory (default: {DEFAULT_DB_DIR})")
    p.add_argument("--model",    type=str,
                   default=os.environ.get("LLM_MODEL", "gemini-1.5-flash"),
                   help="Gemini model name (default: gemini-1.5-flash)")
    p.add_argument("--verbose",  action="store_true",
                   help="Print per-ticket routing details to stdout.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Environment checks ────────────────────────────────────────────────────
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "[ERROR] GOOGLE_API_KEY is not set.\n"
            "        Add it to code/.env  →  GOOGLE_API_KEY=AIza...\n"
            "        Free key: https://aistudio.google.com/app/apikey",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.db_dir.exists():
        print(
            f"[ERROR] Qdrant store not found at {args.db_dir}.\n"
            "        Run first:  python ingest.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.tickets.exists():
        print(f"[ERROR] Tickets CSV not found: {args.tickets}", file=sys.stderr)
        sys.exit(1)

    # ── Load Qdrant vector store ──────────────────────────────────────────────
    print(f"\n[1/4] Loading Qdrant store from {args.db_dir} …")
    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(path=str(args.db_dir))
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    count = qdrant_client.count(collection_name=COLLECTION_NAME).count
    print(f"      ✓ {count} vectors loaded.")

    # ── Initialise LLM, Router, Generator ────────────────────────────────────
    print(f"[2/4] Initialising LLM: gemini/{args.model} …")
    llm       = _build_llm(args.model)
    router    = TicketRouter(llm=llm, retriever=retriever)
    generator = ResponseGenerator(llm=llm, vectorstore=vectorstore)

    # ── Load tickets ──────────────────────────────────────────────────────────
    print(f"[3/4] Loading tickets from {args.tickets} …")
    df      = pd.read_csv(args.tickets)
    n_rows  = len(df)
    print(f"      ✓ {n_rows} tickets to process.")

    # ── Session log entry (AGENTS.md §5.1) ───────────────────────────────────
    _append_session_log(args.tickets, "google-gemini", args.model)

    # ── Process tickets ───────────────────────────────────────────────────────
    print(f"[4/4] Processing tickets …\n")
    results: list[dict] = []

    for idx, row in tqdm(
        df.iterrows(),
        total=n_rows,
        desc="Triaging tickets",
        unit="ticket",
        dynamic_ncols=True,
    ):
        issue   = str(row.get("Issue",   "") or "").strip()
        subject = str(row.get("Subject", "") or "").strip()
        company = str(row.get("Company", "") or "").strip()

        result = _process_ticket(
            ticket_id=int(idx) + 1,   # 1-indexed, human-readable
            issue=issue,
            subject=subject,
            company=company,
            router=router,
            generator=generator,
            verbose=args.verbose,
        )
        results.append(result)

    # ── Write output.csv ──────────────────────────────────────────────────────
    OUTPUT_COLS = ["status", "product_area", "response", "justification", "request_type"]
    out_df = pd.DataFrame(results, columns=OUTPUT_COLS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_replied   = (out_df["status"] == "replied").sum()
    n_escalated = (out_df["status"] == "escalated").sum()

    print(f"\n{'='*50}")
    print(f"✅  Batch complete — {n_rows} tickets processed")
    print(f"   Replied:   {n_replied}")
    print(f"   Escalated: {n_escalated}")
    print(f"\n   Output CSV : {args.output}")
    print(f"   Chat log   : {_LOG_FILE}")
    print(f"{'='*50}\n")

    # Append a summary turn to the AGENTS.md log
    try:
        _ensure_log_dir()
        summary_entry = (
            f"\n## {_ts()} Batch run complete\n\n"
            f"User Prompt (verbatim, secrets redacted):\n"
            f"Run triage agent on {args.tickets.name}\n\n"
            f"Agent Response Summary:\n"
            f"Processed {n_rows} tickets. Replied: {n_replied}. "
            f"Escalated: {n_escalated}. Output written to {args.output}.\n\n"
            f"Actions:\n"
            f"* wrote: {args.output}\n"
            f"* appended {n_rows} ticket entries to {_LOG_FILE}\n\n"
            f"Context:\n"
            f"tool=Antigravity\n"
            f"repo_root={_REPO_ROOT}\n"
            f"worktree=main\n"
            f"parent_agent=none\n"
        )
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(summary_entry)
    except Exception:
        pass


if __name__ == "__main__":
    main()
