"""
CalorAI nutrition RAG engine.

A self-contained, dependency-light retrieval stack that grounds every calorie
number the agent produces in a curated nutrition knowledge base:

    query ─► text views ─► BM25 (lexical) ─┐
                       ├─► phonetic char-n-gram (fuzzy) ─┼─► RRF fusion ─► feature reranker
                       └─► dense embeddings (optional) ──┘        │
                                                                   ▼
                                       calibrated decision: grounded | ambiguous | no_match
                                       + profile-aware disambiguation + citations

Public entry point: :func:`backend.rag.service.get_rag`.
"""
