"""
generator.py — Strict RAG retrieval + response generation engine.

Called by main.py ONLY when the Router decides action = REPLY.

Pipeline per ticket:
  1. Query Qdrant with a domain-metadata filter → top-k chunks
  2. Format chunks into a structured context block
  3. Call the LLM (Google Gemini, injected by main.py) with an ironclad system prompt
  4. If the LLM determines the corpus is insufficient it emits [ESCALATE]
     → caller overrides status to 'escalated'

Public surface:
    generator = ResponseGenerator(llm, vectorstore)
    result    = generator.generate(ticket_text, product_area, domain)
    # result.response      — user-facing text OR starts with "[ESCALATE]"
    # result.docs_retrieved — int, number of chunks used
    # result.escalated      — bool, True if [ESCALATE] was triggered
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ESCALATE_SENTINEL = "[ESCALATE]"
TOP_K = 4  # chunks to retrieve for generation

# Canonical domain name lookup (normalise input company → Qdrant domain tag)
_DOMAIN_NORMALISE: dict[str, str] = {
    "hackerrank": "HackerRank",
    "claude":     "Claude",
    "visa":       "Visa",
}

# ─────────────────────────────────────────────────────────────────────────────
# The Ironclad System Prompt  (matches spec exactly)
# ─────────────────────────────────────────────────────────────────────────────

_GENERATION_SYSTEM_TEMPLATE = """\
You are a strict, factual support agent for {product_area}. \
Your task is to resolve the user's issue using ONLY the provided context documents.

RULES:
- NO HALLUCINATIONS. Do not invent links, policies, features, or phone numbers.
- If the answer cannot be completely and confidently derived from the provided \
context, you MUST abort and output the exact string: \
[ESCALATE] - Information missing from corpus.
- Keep the tone professional, helpful, and concise.

CONTEXT:
{retrieved_context}"""

_GENERATION_USER_TEMPLATE = """\
Support ticket:
{ticket_text}

Provide your response now. Remember: only use information from the CONTEXT above."""

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """Returned by ResponseGenerator.generate() for every REPLY ticket."""
    response: str               # Final text shown to the user (or [ESCALATE] …)
    docs_retrieved: int         # Number of corpus chunks fed to the LLM
    escalated: bool             # True when LLM self-escalated via [ESCALATE]
    source_urls: list[str] = field(default_factory=list)  # for traceability


# ─────────────────────────────────────────────────────────────────────────────
# Helper: domain normalisation
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_domain(company: str) -> Optional[str]:
    """Map raw company string → canonical Qdrant domain tag, or None."""
    key = (company or "").strip().lower()
    return _DOMAIN_NORMALISE.get(key)          # None if unrecognised / "none"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: format retrieved docs into the context block
# ─────────────────────────────────────────────────────────────────────────────


def _format_context(docs: list[Document]) -> str:
    """Render a numbered context block from retrieved LangChain Documents."""
    if not docs:
        return "(No relevant documents found in the corpus.)"

    sections: list[str] = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        title    = m.get("title") or m.get("filename", "?")
        domain   = m.get("domain", "?")
        subpath  = m.get("subpath", "?")
        src_url  = m.get("source_url", "")

        header = f"[Document {i}] {title} | {domain} — {subpath}"
        if src_url:
            header += f"\nSource: {src_url}"

        sections.append(f"{header}\n\n{doc.page_content.strip()}")

    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class ResponseGenerator:
    """RAG-backed response generator for REPLY-routed tickets.

    Args:
        llm:         A LangChain chat model (temperature=0 recommended).
        vectorstore: A LangChain VectorStore backed by Qdrant.
        top_k:       Number of chunks to retrieve per ticket (default 4).
    """

    def __init__(self, llm, vectorstore: VectorStore, top_k: int = TOP_K):
        self._llm         = llm
        self._vectorstore = vectorstore
        self._top_k       = top_k

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        ticket_text: str,
        product_area: str,
        company: str,
    ) -> GenerationResult:
        """Run retrieval + generation for one REPLY ticket.

        Args:
            ticket_text:  The raw ticket body (issue + subject concatenated).
            product_area: Label determined by the router (e.g. 'account_management').
            company:      Raw company field from the CSV ('HackerRank', 'Claude',
                          'Visa', or 'None').

        Returns:
            GenerationResult with .response, .docs_retrieved, .escalated.
        """
        # ── Step 1: Retrieve with domain metadata filter ─────────────────────
        docs = self._retrieve(ticket_text, company)

        # ── Step 2: Format context ────────────────────────────────────────────
        context_block = _format_context(docs)

        # ── Step 3: LLM generation call ───────────────────────────────────────
        system_msg = _GENERATION_SYSTEM_TEMPLATE.format(
            product_area=product_area,
            retrieved_context=context_block,
        )
        user_msg = _GENERATION_USER_TEMPLATE.format(ticket_text=ticket_text.strip())

        try:
            resp    = self._llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ])
            raw_response = resp.content.strip()
        except Exception as exc:
            raw_response = (
                f"{ESCALATE_SENTINEL} - Generation failed: {exc}"
            )

        # ── Step 4: Detect self-escalation ────────────────────────────────────
        escalated = raw_response.startswith(ESCALATE_SENTINEL)

        source_urls = [
            doc.metadata.get("source_url", "")
            for doc in docs
            if doc.metadata.get("source_url")
        ]

        return GenerationResult(
            response=raw_response,
            docs_retrieved=len(docs),
            escalated=escalated,
            source_urls=source_urls,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, company: str) -> list[Document]:
        """Query Qdrant with a domain metadata filter.

        Strategy:
          1. Try domain-filtered similarity search (preferred — authoritative).
          2. If fewer than min_docs returned, top-up with unfiltered results
             to avoid empty context (edge case: unknown company).
        """
        min_docs    = 2   # minimum before we fall back to unfiltered
        domain_tag  = _normalise_domain(company)
        docs: list[Document] = []

        # ── Domain-filtered retrieval ─────────────────────────────────────────
        if domain_tag:
            try:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.domain",
                            match=MatchValue(value=domain_tag),
                        )
                    ]
                )
                docs = self._vectorstore.similarity_search(
                    query,
                    k=self._top_k,
                    filter=qdrant_filter,
                )
            except Exception:
                docs = []   # fall through to unfiltered

        # ── Unfiltered top-up ─────────────────────────────────────────────────
        if len(docs) < min_docs:
            try:
                unfiltered = self._vectorstore.similarity_search(
                    query,
                    k=self._top_k,
                )
                seen     = {d.metadata.get("filepath") for d in docs}
                for d in unfiltered:
                    if d.metadata.get("filepath") not in seen:
                        docs.append(d)
                        seen.add(d.metadata.get("filepath"))
                    if len(docs) >= self._top_k:
                        break
            except Exception:
                pass

        return docs[: self._top_k]
