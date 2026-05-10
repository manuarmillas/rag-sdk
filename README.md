# RAG SDK

A modular, type-safe SDK for building Retrieval-Augmented Generation (RAG) pipelines in TypeScript. Hexagonal architecture with pluggable adapters for embeddings, vector stores, and LLM generation.

## Packages

| Package | Description |
|---------|-------------|
| `@rag-sdk/core` | Core ports, pipelines (ingest, query, generate, hybrid search), error types, and zero-dep chunkers |
| `@rag-sdk/embedding` | Embedding provider adapters (OpenAI, Ollama, Cohere, VoyageAI, Local) |
| `@rag-sdk/store` | Vector store adapters — Memory (zero deps), Qdrant, Pinecone, pgvector |
| `@rag-sdk/generator` | LLM generation adapters (OpenAI, Ollama, Cohere) |
| `@rag-sdk/reranker` | Reranker adapters (Cohere, Local cross-encoder) |
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

### Stream Generate (RAG → Streaming Answer)

```ts
const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  generator: createOpenAIGenerator({ apiKey: process.env.OPENAI_API_KEY }),
});

await sdk.ingest([{ content: 'RAG SDK supports real-time streaming of generated answers.' }]);

for await (const token of sdk.generateStream('How does streaming work?')) {
  process.stdout.write(token);
}
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

### Hybrid Search (Vector + Keyword)

```ts
const keywordSearcher = {
  id: 'my-keyword-index',
  async keywordSearch(text, options) {
    // Your keyword search implementation (e.g. Meilisearch, Elasticsearch, SQLite FTS)
    return [{ id: '1', score: 1, content: '...', metadata: {} }];
  },
};

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  keywordSearcher,
});

await sdk.ingest([{ content: 'RAG SDK supports hybrid vector + keyword search.' }]);
const result = await sdk.query('hybrid search', { topK: 10, hybrid: { rrfK: 60 } });
console.log(result.results); // fused and re-ranked via Reciprocal Rank Fusion
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

## V3 Quickstart — Local & Multi-Provider

### Local Reranker (free, offline, zero API keys)

```ts
import { createLocalReranker } from '@rag-sdk/reranker';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  reranker: createLocalReranker(), // 80MB model downloads on first use
});

// Same API — swaps adapter without app code changes
const result = await sdk.query('relevant docs', { topK: 10, rerank: { topN: 3 } });
```

### Ollama Embedding (local, zero deps)

```ts
import { createOllamaEmbedding } from '@rag-sdk/embedding';

const sdk = rag({
  provider: createOllamaEmbedding({ model: 'nomic-embed-text' }),
  store: createMemoryStore({ dimensions: 768 }),
});
// Requires Ollama running at localhost:11434
```

### Ollama Generator (local, streaming)

```ts
import { createOllamaEmbedding } from '@rag-sdk/embedding';
import { createOllamaGenerator } from '@rag-sdk/generator';

const sdk = rag({
  provider: createOllamaEmbedding({ model: 'nomic-embed-text' }),
  store: createMemoryStore({ dimensions: 768 }),
  generator: createOllamaGenerator({ model: 'llama3.2' }),
});

const result = await sdk.generate('What is RAG?');
console.log(result.answer);

// Streaming
for await (const token of sdk.generateStream('Explain step by step')) {
  process.stdout.write(token);
}
```

### Cohere Embedding + Generator

```ts
import { createCohereEmbedding } from '@rag-sdk/embedding';
import { createCohereGenerator } from '@rag-sdk/generator';

const sdk = rag({
  provider: createCohereEmbedding({ apiKey: process.env.COHERE_API_KEY }),
  store: createMemoryStore({ dimensions: 1024 }),
  generator: createCohereGenerator({ apiKey: process.env.COHERE_API_KEY }),
});
```

### VoyageAI Embedding

```ts
import { createVoyageEmbedding } from '@rag-sdk/embedding';

const sdk = rag({
  provider: createVoyageEmbedding({ apiKey: process.env.VOYAGE_API_KEY }),
  store: createMemoryStore({ dimensions: 512 }),
});
```

### Local Embeddings (free, offline)

```ts
import { createLocalEmbedding } from '@rag-sdk/embedding';

const sdk = rag({
  provider: createLocalEmbedding(), // 80MB model, CPU-only, no API keys
  store: createMemoryStore({ dimensions: 384 }),
});
```

## Architecture

```
@rag-sdk/core        — ports + pipelines + errors + default chunkers
    ↑
    ├── @rag-sdk/embedding   — EmbeddingProvider: OpenAI, Ollama, Cohere, VoyageAI, Local
    ├── @rag-sdk/store       — VectorStore: Memory, Qdrant, Pinecone, pgvector
    ├── @rag-sdk/generator   — Generator: OpenAI, Ollama, Cohere
    ├── @rag-sdk/reranker    — Reranker: Cohere, Local (cross-encoder)
    └── @rag-sdk/chunker     — SemanticChunker (uses EmbeddingProvider port)
```

All adapter packages use optional peer dependencies — you only install the vendor SDKs you need.
