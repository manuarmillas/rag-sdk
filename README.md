# RAG SDK

A modular, type-safe SDK for building Retrieval-Augmented Generation (RAG) pipelines in TypeScript. Hexagonal architecture with pluggable adapters for embeddings, vector stores, and LLM generation.

## Packages

| Package | Description |
|---------|-------------|
| `@rag-sdk/core` | Core ports, pipelines (ingest, query, generate), error types, and zero-dep chunkers |
| `@rag-sdk/embedding` | Embedding provider adapters (OpenAI) |
| `@rag-sdk/store` | Vector store adapters — Memory (zero deps), Qdrant, Pinecone, pgvector |
| `@rag-sdk/generator` | LLM generation adapters (OpenAI) |

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
    └── @rag-sdk/generator   — Generator adapters (OpenAI LLM)
```

All adapter packages use optional peer dependencies — you only install the vendor SDKs you need.
