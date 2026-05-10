# RAG SDK

A modular, type-safe SDK for building Retrieval-Augmented Generation (RAG) pipelines in TypeScript. Hexagonal architecture with pluggable adapters for embeddings, vector stores, LLM generation, and reranking — all behind stable ports.

```ts
// 5 lines, swap any adapter, zero core changes
const sdk = rag({
  provider: createLocalEmbedding(),
  store: createMemoryStore({ dimensions: 384 }),
  generator: createOllamaGenerator({ model: 'llama3.2' }),
  reranker: createLocalReranker(),
});
```

## Packages

| Package | Description | Adapters |
|---------|-------------|----------|
| `@rag-sdk/core` | Ports, pipelines, errors, default chunkers | `MarkdownChunker`, `RecursiveChunker` |
| `@rag-sdk/embedding` | `EmbeddingProvider` adapters | OpenAI, Ollama, Cohere, VoyageAI, Local |
| `@rag-sdk/store` | `VectorStore` adapters | Memory (zero deps), Qdrant, Pinecone, pgvector |
| `@rag-sdk/generator` | `Generator` adapters | OpenAI, Ollama, Cohere |
| `@rag-sdk/reranker` | `Reranker` adapters | Cohere, Local cross-encoder |
| `@rag-sdk/chunker` | Advanced chunking | `SemanticChunker` |

## Installation

```bash
# Core + embedding + in-memory store (minimal)
pnpm add @rag-sdk/core @rag-sdk/embedding @rag-sdk/store

# Generator (RAG → answer)
pnpm add @rag-sdk/generator

# Reranker (post-retrieval quality)
pnpm add @rag-sdk/reranker

# Optional peer dependencies — install only what you use:
pnpm add openai                          # for OpenAI embedding / generation
pnpm add cohere-ai                       # for Cohere embedding / generation / reranker
pnpm add voyageai                        # for VoyageAI embedding
pnpm add @huggingface/transformers       # for local embedding / reranker (CPU, free)
pnpm add @qdrant/js-client-rest          # for Qdrant store
pnpm add @pinecone-database/pinecone     # for Pinecone store
pnpm add pg                              # for pgvector store
```

All vendor SDKs are **optional peer dependencies** — install only what you need. Missing peers produce clear runtime errors with install hints.

## Provider Comparison

### Embedding Providers

| Provider | Dimensions | Model | Type | Install |
|----------|-----------|-------|------|---------|
| **Local** | 384 | `all-MiniLM-L6-v2` | CPU, free | `@huggingface/transformers` |
| **OpenAI** | 1536 / 3072 | `text-embedding-3-small/large` | Cloud, paid | `openai` |
| **Ollama** | configurable | `nomic-embed-text`, any Ollama model | Local, free | None (REST API) |
| **Cohere** | 1024 | `embed-english-v3.0` | Cloud, paid | `cohere-ai` |
| **VoyageAI** | 512 / 1024 | `voyage-3-lite / voyage-3` | Cloud, paid | `voyageai` |

### Generator Providers

| Provider | Model | Type | Streaming | Install |
|----------|-------|------|-----------|---------|
| **OpenAI** | `gpt-4o-mini`, `gpt-4o` | Cloud, paid | ✅ SSE | `openai` |
| **Ollama** | `llama3.2`, `mistral`, any GGUF | Local, free | ✅ SSE | None (REST API) |
| **Cohere** | `command-r-plus`, `command-r` | Cloud, paid | ✅ | `cohere-ai` |

### Reranker Providers

| Provider | Model | Type | Install |
|----------|-------|------|---------|
| **Local** | `ms-marco-MiniLM-L-6-v2` (80MB) | CPU, free | `@huggingface/transformers` |
| **Cohere** | `rerank-english-v3.0` | Cloud, paid | `cohere-ai` |

> ℹ️ Full comparison with pricing, quality trade-offs, and when to use each provider: see [docs/providers.md](docs/providers.md).

## Quickstart

### Retrieve (embeddings + search)

```ts
import { rag } from '@rag-sdk/core';
import { createOpenAI } from '@rag-sdk/embedding';
import { createMemoryStore } from '@rag-sdk/store';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
});

await sdk.ingest([
  { content: 'RAG SDK is a TypeScript toolkit for RAG pipelines.' },
]);
const result = await sdk.query('What is RAG SDK?');
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

const result = await sdk.generate('What is RAG SDK?');
console.log(result.answer);
```

### Stream Generate

```ts
for await (const token of sdk.generateStream('Explain step by step')) {
  process.stdout.write(token);
}
```

### Rerank (post-retrieval quality boost)

```ts
import { createCohereReranker } from '@rag-sdk/reranker';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  reranker: createCohereReranker({ apiKey: process.env.COHERE_API_KEY }),
});

const result = await sdk.query('relevant docs', {
  topK: 10,
  rerank: { topN: 3 },
});
```

### 100% Local RAG (CPU only, no APIs, no servers)

```ts
import { createLocalEmbedding } from '@rag-sdk/embedding';
import { createLocalReranker } from '@rag-sdk/reranker';

const sdk = rag({
  provider: createLocalEmbedding(),    // 80MB model, auto-downloads
  store: createMemoryStore({ dimensions: 384 }),
  reranker: createLocalReranker(),     // 80MB model, cross-encoder
});

await sdk.ingest([{ content: 'Your documents here...' }]);
const result = await sdk.query('your question', { topK: 5, rerank: { topN: 3 } });
```

### Ollama Workflow (local, streaming)

```ts
import { createOllamaEmbedding } from '@rag-sdk/embedding';
import { createOllamaGenerator } from '@rag-sdk/generator';

const sdk = rag({
  provider: createOllamaEmbedding({ model: 'nomic-embed-text' }),
  store: createMemoryStore({ dimensions: 768 }),
  generator: createOllamaGenerator({ model: 'llama3.2' }),
});

// Synchronous generation
const result = await sdk.generate('What is RAG?');

// Token-by-token streaming
for await (const token of sdk.generateStream('Explain step by step')) {
  process.stdout.write(token);
}
```

### Cohere Stack

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

### Hybrid Search (Vector + Keyword)

```ts
const keywordSearcher = {
  id: 'my-index',
  async keywordSearch(text, options) {
    return [{ id: '1', score: 0.9, content: '...', metadata: {} }];
  },
};

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
  keywordSearcher,
});

const result = await sdk.query('hybrid search', {
  topK: 10,
  hybrid: { rrfK: 60 },
});
```

### Chunking

```ts
// Markdown-aware (zero deps, in core)
import { MarkdownChunker } from '@rag-sdk/core';
const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 200 });

// Semantic (requires an EmbeddingProvider)
import { SemanticChunker } from '@rag-sdk/chunker';
const chunker = new SemanticChunker(provider, { chunkSize: 500, threshold: 0.5 });

const sdk = rag({ provider, store, chunker });
```

### Production Stores

```ts
import { createQdrantStore } from '@rag-sdk/store';

const store = createQdrantStore({
  url: 'http://localhost:6333',
  collectionName: 'my-docs',
  dimensions: 1536,
});
// Also available: createPineconeStore(), createPgVectorStore()
```

### Typed Metadata

```ts
interface DocMetadata { author: string; category: string; tags: string[]; }

const sdk = rag<DocMetadata>({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore<DocMetadata>({ dimensions: 1536 }),
});

await sdk.ingest([{
  content: '...',
  metadata: { author: 'Jane', category: 'docs', tags: ['rag'] },
}]);

// Filter by typed metadata — type-checked at compile time
const result = await sdk.query('rag', { filter: { category: 'docs' } });
```

## Architecture

```
                          ┌──────────────────────┐
                          │    @rag-sdk/core      │
                          │  Ports + Pipelines    │
                          │  Errors + Chunkers    │
                          └──────┬───────┬───────┘
                                 │       │
          ┌──────────────────────┼───────┼──────────────────────┐
          │                      │       │                      │
          ▼                      ▼       ▼                      ▼
┌─────────────────┐   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ @rag-sdk/       │   │ @rag-sdk/    │ │ @rag-sdk/    │ │ @rag-sdk/    │
│ embedding       │   │ store        │ │ generator    │ │ reranker     │
│                 │   │              │ │              │ │              │
│ • OpenAI        │   │ • Memory     │ │ • OpenAI     │ │ • Cohere     │
│ • Ollama        │   │ • Qdrant     │ │ • Ollama     │ │ • Local      │
│ • Cohere        │   │ • Pinecone   │ │ • Cohere     │ │   (cross-enc)│
│ • VoyageAI      │   │ • pgvector   │ │              │ └──────────────┘
│ • Local (CPU)   │   └──────────────┘ └──────────────┘
└─────────────────┘
```

**Key principle**: all adapter packages implement **ports** defined in `@rag-sdk/core`. The core never imports from adapter packages — adapters depend on ports, never the reverse. Adding a new provider means adding a file to an existing package. Zero core changes. This is Clean / Hexagonal Architecture.

## Error Handling

```ts
import { ProviderError, ConfigurationError, DimensionMismatchError } from '@rag-sdk/core';

try {
  await sdk.ingest(documents);
} catch (err) {
  if (err instanceof ProviderError) {
    // e.g. "Provider openai failed during embed"
    console.error(`${err.providerId} failed during ${err.operation}`, err.cause);
  } else if (err instanceof DimensionMismatchError) {
    // Embedding dimension doesn't match store — misconfiguration
    console.error(err.message);
  }
}
```

## Further Reading

| Resource | Description |
|----------|-------------|
| [`examples/v3-full-rag.ts`](examples/v3-full-rag.ts) | Full RAG pipeline demo (8 sections, runs out of the box) |
| [`examples/basic-usage.ts`](examples/basic-usage.ts) | Per-feature V2 usage examples |
| [`docs/providers.md`](docs/providers.md) | Provider comparison: dimensions, pricing, when to use |
| [`docs/packages.md`](docs/packages.md) | Per-package API reference with all adapter signatures |

## Design Decisions

- **Factory functions, not classes**: `createXxx()` returns plain object literals. No `new`, no `extends`, no inheritance chains.
- **Structural typing**: adapters satisfy ports by shape, not by `implements` keyword. TypeScript verifies at compile time.
- **Optional peer dependencies**: adapters declare vendor SDKs as optional peers. Install only what you use.
- **Lazy initialization**: local adapters load models on first use, not at factory creation. First call downloads ~80MB to `~/.cache/huggingface/`.
- **Error wrapping**: all provider calls wrapped in `ProviderError(providerId, operation, cause)`. Consistent error surface regardless of underlying SDK.
