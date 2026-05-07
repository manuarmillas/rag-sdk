# RAG SDK

A modular, type-safe SDK for building Retrieval-Augmented Generation (RAG) pipelines in TypeScript. Hexagonal architecture with pluggable adapters for embeddings, vector stores, and LLM generation.

## Packages

| Package | Description |
|---------|-------------|
| `@rag-sdk/core` | Core ports, pipelines (ingest, query, generate), error types, and zero-dep chunkers |
| `@rag-sdk/embedding` | Embedding provider adapters (OpenAI) |
| `@rag-sdk/store` | Vector store adapters — Memory (zero deps), Qdrant, Pinecone, pgvector |
| `@rag-sdk/generator` | LLM generation adapters (OpenAI) |
| `@rag-sdk/reranker` | Reranker adapters (Cohere) |
| `@rag-sdk/chunker` | Semantic chunking (requires EmbeddingProvider) |

## Installation

```bash
# Minimal — embeddings + in-memory store
pnpm add @rag-sdk/core @rag-sdk/embedding @rag-sdk/store

# With generation
pnpm add @rag-sdk/generator

# Production vector stores (optional peer dependencies)
pnpm add @qdrant/js-client-rest   # for Qdrant
pnpm add @pinecone-database/pinecone  # for Pinecone
pnpm add pg                        # for pgvector
```

## Quickstart

### Retrieve

```ts
import { rag } from '@rag-sdk/core';
import { createOpenAI } from '@rag-sdk/embedding';
import { createMemoryStore } from '@rag-sdk/store';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
});

await sdk.ingest([{ content: 'Hello world', metadata: { source: 'test' } }]);
const result = await sdk.query('hello');
console.log(result.results);
```

### Generate (RAG → Answer)

```ts
import { createOpenAIGenerator } from '@rag-sdk/generator';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  generator: createOpenAIGenerator({ apiKey: process.env.OPENAI_API_KEY }),
});

await sdk.ingest([{ content: 'RAG SDK is a TypeScript toolkit for RAG pipelines.' }]);
const result = await sdk.generate('What is RAG SDK?');
console.log(result.answer);
```

### Rerank (Retrieve → Rerank)

```ts
import { createCohereReranker } from '@rag-sdk/reranker';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  reranker: createCohereReranker({ apiKey: process.env.COHERE_API_KEY }),
});

await sdk.ingest([{ content: 'RAG SDK supports reranking for better relevance.' }]);
const result = await sdk.query('What improves relevance?', { topK: 10, rerank: { topN: 3 } });
console.log(result.results); // top 3 reranked results
```

### Semantic Chunking

```ts
import { SemanticChunker } from '@rag-sdk/chunker';

const chunker = new SemanticChunker(provider, { chunkSize: 500, threshold: 0.5 });

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  chunker,
});
```

### Markdown Chunking

```ts
import { MarkdownChunker } from '@rag-sdk/core';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  chunker: new MarkdownChunker({ chunkSize: 1000, overlap: 200 }),
});
```

### Production Store (Qdrant)

```ts
import { createQdrantStore } from '@rag-sdk/store';

const store = createQdrantStore({
  url: 'http://localhost:6333',
  collectionName: 'my-docs',
  dimensions: 1536,
});
```

## Architecture

```
@rag-sdk/core        — ports + pipelines + errors + default chunkers
    ↑
    ├── @rag-sdk/embedding   — EmbeddingProvider adapters
    ├── @rag-sdk/store       — VectorStore adapters (memory, Qdrant, Pinecone, pgvector)
    ├── @rag-sdk/generator   — Generator adapters (OpenAI LLM)
    ├── @rag-sdk/reranker    — Reranker adapters (Cohere)
    └── @rag-sdk/chunker     — SemanticChunker (uses EmbeddingProvider port)
```

All adapter packages use optional peer dependencies — you only install the vendor SDKs you need.
