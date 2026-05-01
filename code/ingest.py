"""
ingest.py — Corpus ingestion pipeline for the HackerRank Orchestrate triage agent.

Reads every .md file in the data/ directory, assigns domain metadata
(HackerRank | Claude | Visa), splits into overlapping chunks, embeds with
local HuggingFace sentence-transformers model (all-MiniLM-L6-v2, no API key),
Qdrant vector store at ./qdrant_db_v4.

Usage:
    python ingest.py                   # ingest with default settings
    python ingest.py --data-dir ../data --db-dir ./qdrant_db_v4 --reset

Run once before launching the agent. Re-run with --reset to rebuild the DB.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ── Load environment variables (.env in project root or code/) ───────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_MAP: dict[str, str] = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}

# Front-matter key → metadata field
FRONTMATTER_FIELDS = {
    "title": "title",
    "source_url": "source_url",
    "last_updated_iso": "last_updated",
    "article_id": "article_id",
}

CHUNK_SIZE = 800          # characters per chunk (not tokens)
CHUNK_OVERLAP = 120       # overlap between consecutive chunks
COLLECTION_NAME = "support_corpus"
EMBED_DIM = 384           # all-MiniLM-L6-v2 output dimension

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def detect_domain(filepath: Path, data_root: Path) -> str:
    """Return the canonical domain name based on which top-level subdirectory
    the file lives under inside data_root."""
    try:
        relative = filepath.relative_to(data_root)
        top_dir = relative.parts[0].lower()
        return DOMAIN_MAP.get(top_dir, "Unknown")
    except ValueError:
        return "Unknown"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-style front matter (--- ... ---) from markdown.
    Returns (metadata_dict, body_without_frontmatter)."""
    metadata: dict = {}
    pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = pattern.match(text)
    if match:
        raw_fm = match.group(1)
        body = text[match.end():]
        for line in raw_fm.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().strip('"').lower()
                val = val.strip().strip('"')
                if key in FRONTMATTER_FIELDS:
                    metadata[FRONTMATTER_FIELDS[key]] = val
    else:
        body = text
    return metadata, body


def build_subpath_label(filepath: Path, data_root: Path) -> str:
    """Return a human-readable sub-path label, e.g. 'hackerrank/assessments'."""
    try:
        relative = filepath.relative_to(data_root)
        parts = relative.parts
        # parts[0] = domain dir, parts[-1] = filename; middle parts = category
        if len(parts) >= 3:
            return "/".join(parts[:-1])  # e.g. hackerrank/assessments
        return str(relative.parent)
    except ValueError:
        return str(filepath.parent)


def load_documents(data_root: Path) -> list[Document]:
    """Walk data_root and load every .md file as a LangChain Document
    with rich metadata."""
    md_files = sorted(data_root.rglob("*.md"))
    documents: list[Document] = []

    print(f"  Found {len(md_files)} markdown files under {data_root}")

    for filepath in tqdm(md_files, desc="  Loading docs", unit="file"):
        try:
            raw = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"  [WARN] Could not read {filepath}: {exc}", file=sys.stderr)
            continue

        fm_meta, body = parse_frontmatter(raw)
        domain = detect_domain(filepath, data_root)
        subpath = build_subpath_label(filepath, data_root)

        # Build the metadata dict that every chunk will carry
        metadata: dict = {
            "domain": domain,
            "subpath": subpath,
            "filename": filepath.name,
            "filepath": str(filepath),
            # front-matter fields (may be empty strings if absent)
            "title": fm_meta.get("title", ""),
            "source_url": fm_meta.get("source_url", ""),
            "last_updated": fm_meta.get("last_updated", ""),
            "article_id": fm_meta.get("article_id", ""),
        }

        documents.append(Document(page_content=body.strip(), metadata=metadata))

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split preferring paragraph → sentence → word boundaries
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
        add_start_index=True,  # adds chunk_start_index to metadata
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split {len(documents)} docs → {len(chunks)} chunks "
          f"(size≤{chunk_size}, overlap={chunk_overlap})")
    return chunks


def build_qdrant(
    chunks: list[Document],
    db_dir: Path,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> QdrantVectorStore:
    """Embed chunks and persist to a local file-based Qdrant store.

    Uses the local HuggingFace all-MiniLM-L6-v2 model (no API key required).
    Batches in groups of 20 to stay inside the free-tier rate limit.

    Design rule: ONE QdrantClient and ONE QdrantVectorStore are created before
    the loop.  The loop calls add_documents() only — never from_documents() —
    to guarantee no second connection is opened while the first is still live.
    """
    if reset and db_dir.exists():
        import shutil
        print(f"  [reset] Deleting existing DB at {db_dir} …")
        shutil.rmtree(db_dir)

    db_dir.mkdir(parents=True, exist_ok=True)

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # ── Open ONE client for the entire ingestion run ──────────────────────────
    # Local file-based Qdrant allows only one connection at a time; we must
    # never open a second one (from_documents() would do exactly that).
    client = QdrantClient(path=str(db_dir))

    # Create the collection once if it doesn't exist yet
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"  Created Qdrant collection '{collection_name}'")
    else:
        print(f"  Reusing existing Qdrant collection '{collection_name}'")

    # ── Build the vector store wrapper ONCE, using the existing client ────────
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    print(f"  Embedding {len(chunks)} chunks → {db_dir}  (this may take a few minutes…)")

    # Batch in groups of 20 to respect the free-tier rate limit
    # (all-MiniLM-L6-v2 runs locally — no rate-limit, but batching keeps memory
    #  usage predictable for the full 9 000+ chunk corpus)
    BATCH = 20

    for i in tqdm(range(0, len(chunks), BATCH), desc="  Embedding batches"):
        batch = chunks[i : i + BATCH]
        vectorstore.add_documents(batch)   # single open connection — no lock

    count = client.count(collection_name=collection_name).count
    print(f"  ✓ Qdrant store persisted — collection '{collection_name}' "
          f"contains {count} vectors")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    here = Path(__file__).parent
    default_data = (here.parent / "data").resolve()
    default_db = (here / "qdrant_db_v4").resolve()

    parser = argparse.ArgumentParser(
        description="Ingest the support corpus into a local Qdrant vector store."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help=f"Root directory of the support corpus (default: {default_data})",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=default_db,
        help=f"Where to persist the Qdrant store (default: {default_db})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Max characters per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Qdrant store before ingesting (full rebuild).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate API key early
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "[ERROR] GOOGLE_API_KEY is not set.\n"
            "  Add it to your .env file or export it in your shell.\n"
            "  Get a free key at: https://aistudio.google.com/app/apikey",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.data_dir.exists():
        print(
            f"[ERROR] data directory not found: {args.data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n── Step 1 · Load documents ──────────────────────────────────────")
    documents = load_documents(args.data_dir)
    if not documents:
        print("[ERROR] No documents loaded — check --data-dir.", file=sys.stderr)
        sys.exit(1)

    print("\n── Step 2 · Split into chunks ───────────────────────────────────")
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

    print("\n── Step 3 · Embed & persist to Qdrant ───────────────────────────")
    build_qdrant(chunks, args.db_dir, args.collection, reset=args.reset)

    print("\n✅  Ingestion complete.")
    print(f"   Vector store: {args.db_dir}")
    print("   Run main.py to start the triage agent.\n")


if __name__ == "__main__":
    main()
