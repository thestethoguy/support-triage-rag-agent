# Multi-Domain Support Triage Agent 🤖

A modular, LLM-powered Retrieval-Augmented Generation (RAG) pipeline designed to intelligently route and resolve high-volume customer support tickets across multiple domains (Visa, Claude, HackerRank).

## 🚀 The Architecture

This agent utilizes a hybrid approach to balance speed, cost, and reliability:
* **The Brain:** Generates final responses using **Google Gemini 1.5 Flash**, chosen for its massive context window, high-speed generation, and cost-efficiency.
* **The Retrieval Engine:** Powered by a local **Qdrant Vector Database**.
* **The Embeddings:** Utilizes local open-source `all-MiniLM-L6-v2` HuggingFace embeddings. (Pivoted from cloud embeddings to local models to achieve 100% reliability, zero API latency, and bypass 404/rate-limit errors).

## 🛡️ Safety & Guardrails (Zero Hallucinations)

In customer support, hallucinations are unacceptable. This agent implements a strict **Domain-Scoped Metadata Filtering** system. 
1. A dual-layer router (Regex heuristics + LLM) classifies the incoming ticket.
2. The vector database is queried *strictly* within that specific domain's metadata tag. 
3. If the retrieved context does not contain the answer, the LLM is explicitly prompted to output a sentinel value (`[ESCALATE]`), which intercepts the generation and routes the ticket to a human agent.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM:** Google Gemini 1.5 Flash (`langchain-google-genai`)
* **Embeddings:** HuggingFace `sentence-transformers`
* **Vector Store:** Qdrant (Local instance)
* **Framework:** LangChain

## 💡 Lessons Learned
During development, dealing with Windows file-locking exceptions (`AlreadyLocked` on local vector stores) required refactoring the ingestion logic to guarantee single-instance connections and versioned directory states. Furthermore, migrating from deprecated cloud embedding endpoints to robust local transformers ensured the pipeline remained highly available.
