# Phase 2 — Personal Knowledge Engine Architecture

## 1. Objective

Transform the existing RAG-based Personal Knowledge Engine into a single-agent knowledge assistant.

The agent will decide which knowledge-search tools to call. Those tools will reuse the existing retrieval pipeline. The final answer must remain grounded in retrieved evidence, preserve citations, and abstain when the evidence is insufficient.

This phase does not introduce a multi-agent architecture.

## 2. Current architecture — Phase 1

The current question-answering flow is:

1. User submits a question.
2. `KnowledgeAnswer` receives the question.
3. `RetrievalPipeline` performs hybrid retrieval and reranking.
4. `context_builder.py` formats retrieved chunks.
5. `GeminiAnswerGenerator` generates an answer.
6. The system returns an answer with citations or an insufficient-evidence response.

The retrieval pipeline combines BM25 and semantic retrieval through Reciprocal Rank Fusion, followed by cross-encoder reranking.

## 3. Target architecture — Phase 2

The proposed flow is:

1. User submits a question.
2. A single agent receives the question.
3. The agent selects one or more available knowledge-search tools.
4. The tools invoke the existing retrieval pipeline.
5. Retrieved evidence is collected and deduplicated.
6. Existing context-building and generation components produce the answer.
7. Citation validation and insufficient-evidence handling are preserved.
8. The answer is returned to the user.

The agent is responsible for deciding where to search. It is not an independent source of factual knowledge.

## 4. Design principles

### Reuse existing retrieval

Do not duplicate BM25, semantic retrieval, hybrid fusion, or reranking inside the agent.

### Separate decisions from execution

The agent decides which tool to call. Each tool performs a defined search operation and returns structured results.

### Preserve evidence provenance

Retrieved evidence must retain its chunk ID and source information so that citations can be traced back to the original document.

### Ground every answer

The generator must answer using retrieved evidence. Unsupported claims must not be presented as facts.

### Preserve abstention

If evidence is missing or insufficient, the system must communicate that limitation rather than invent an answer.

### Bound agent execution

The agent must have limits on tool calls and execution so that it cannot enter an uncontrolled loop.

## 5. Proposed knowledge tools

Initial tool candidates:

- `search_projects`
- `search_skills`
- `search_experience`
- `search_notes`

Potential later tools:

- `search_education`
- `search_certifications`
- `search_activities`

Tool availability must reflect the actual ingested documents. A tool should not imply that a category of information exists when the knowledge base does not contain supporting documents.

## 6. Document and chunk metadata

The current `Document` model contains source and metadata fields.

The current `DocumentChunk` model does not preserve these fields. Chunking therefore loses document provenance.

Planned improvement:

- Preserve source information in every chunk.
- Preserve document metadata in every chunk.
- Add a document category where appropriate.
- Persist the enriched chunks in JSONL.
- Make category-aware retrieval possible without duplicating the retrieval pipeline.

The existing combined personal knowledge-base document contains multiple topics. It should not be treated as if each chunk necessarily belongs to one specialized category. Section-level categorization may be considered later if needed.

## 7. Ingestion considerations

The current `ingest.py` command processes one input file per run. Its default input is `data/ayush-more-knowledge-base.md`.

The data directory also contains separate Markdown files for projects, skills, education, activities, and work experience. The current command does not automatically ingest all these files.

A future ingestion improvement must define how these documents are discovered, categorized, chunked, and saved.

## 8. Existing components to reuse

- `RetrievalPipeline`
- `HybridRetriever`
- `BM25Retriever`
- `QdrantRetriever`
- `CrossEncoderReranker`
- `context_builder.py`
- `GeminiAnswerGenerator`
- Existing answer and citation models
- Existing API response schemas where possible

## 9. Proposed future package structure

```text
src/knowledge_engine/
    agent/
        __init__.py
        runner.py
        prompts.py
        registry.py
        tools.py
```

The agent package will coordinate existing components rather than replace them.

## 10. Known technical debt and risks

- Chunking currently drops document source and metadata.
- The default ingestion command processes only one source file.
- The combined knowledge-base document contains multiple topics.
- Existing citation validation confirms chunk-ID membership, but this alone does not prove that a citation supports a particular claim.
- The chunk overlap fallback in `chunking.py` needs a regression test and correction.
- The Gemini generator contains a possible API-key logging issue that must be addressed before production use.
- The README's evaluation and processed-file names should be checked against the actual repository.

These issues should be handled in the appropriate implementation milestones, without unnecessarily changing the retrieval baseline.

## 11. Evaluation principles

Phase 2 must preserve the Phase 1 retrieval baseline and add agent-specific evaluation.

Agent evaluation should measure:

- Correct tool selection.
- Retrieval quality.
- Evidence coverage.
- Citation validity.
- Answer grounding.
- Correct abstention.
- Bounded tool execution.
- Multi-turn conversation behavior.

## 12. Phase 1 baseline

Latest reported test result:

- 37 tests passed.
- Hybrid + reranker Precision@5: 0.252.
- Hybrid + reranker Recall@5: 1.000.
- Hybrid + reranker MRR: 0.872.

These are the baseline results to compare against during Phase 2.

## 13. Phase 2 completion goal

The completed system should behave as a single-agent knowledge assistant that can select appropriate search tools, retrieve and compare evidence, preserve citations, answer from supported information, and abstain when evidence is insufficient.