# Package API Reference

Factory signatures, config interfaces, and ports for every public export in RAG SDK V3.

## `@rag-sdk/core`

### Main Entry

```ts
import { rag } from '@rag-sdk/core';

function rag<M extends Metadata = Metadata>(config: RagConfig<M>): RagSDK<M>;
```

### `RagConfig<M>`

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `provider` | `EmbeddingProvider` | ✅ | — |
| `store` | `VectorStore<M>` | ✅ | — |
| `generator` | `Generator<M>` | ❌ | — |
| `reranker` | `Reranker<M>` | ❌ | — |
| `keywordSearcher` | `KeywordSearcher<M>` | ❌ | — |
| `chunker` | `Chunker` | ❌ | `RecursiveChunker` |
| `namespace` | `string` | ❌ | — |

### `RagSDK<M>`

```ts
interface RagSDK<M extends Metadata = Metadata> {
  ingest(documents: Document<M>[], options?: IngestOptions): Promise<void>;
  query(text: string, options?: QueryOptions<M>): Promise<QueryResult<M>>;
  generate(request: GenerateRequest<M>, options?: GenerateOptions): Promise<GenerationResult<M>>;
  generateStream(request: GenerateRequest<M>, options?: GenerateOptions): AsyncGenerator<string, void, undefined>;
}
```

### Chunkers

```ts
// Recursive character splitter — default
new RecursiveChunker({ chunkSize?: number; overlap?: number });

// Markdown-aware — splits on headers, code blocks, paragraphs
new MarkdownChunker({ chunkSize?: number; overlap?: number });
```

### Errors

```ts
RagSdkError          // Base — { code: string; message: string; cause?: unknown }
ConfigurationError   // Invalid config at construction
ValidationError      // Runtime input validation
ProviderError        // { providerId: string; operation: string; cause?: unknown }
StoreError           // Vector store failure
ChunkingError        // Chunking pipeline failure
DimensionMismatchError // Embedding/Store dimension mismatch
BatchError           // Batch processing failure
```

---

## `@rag-sdk/embedding`

All factories return `EmbeddingProvider`.

```ts
interface EmbeddingProvider {
  readonly id: string;
  readonly modelId: string;
  readonly dimensions: number;
  readonly maxBatchSize?: number;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}
```

### `createOpenAI(config)`

```ts
interface CreateOpenAIConfig {
  apiKey: string;
  model?: string;          // default: 'text-embedding-3-small'
  dimensions?: number;     // default: 1536
  baseURL?: string;
  organization?: string;
}
```

> Requires: `pnpm add openai` (optional peer)

### `createOllamaEmbedding(config)`

```ts
interface OllamaEmbeddingConfig {
  model: string;           // e.g. 'nomic-embed-text'
  baseURL?: string;        // default: 'http://localhost:11434'
  dimensions?: number;     // auto-detected from model
}
```

> Requires: Ollama running (no npm dep)

### `createCohereEmbedding(config)`

```ts
interface CohereEmbeddingConfig {
  apiKey: string;
  model?: string;          // default: 'embed-english-v3.0'
  inputType?: string;      // 'search_document' | 'search_query' | 'classification' | 'clustering'
}
```

> Requires: `pnpm add cohere-ai` (optional peer)

### `createVoyageEmbedding(config)`

```ts
interface VoyageEmbeddingConfig {
  apiKey: string;
  model?: string;          // default: 'voyage-3-lite' (512d)
  inputType?: string;      // 'document' | 'query'
}
```

> Requires: `pnpm add voyageai` (optional peer)

### `createLocalEmbedding(config?)`

```ts
interface LocalEmbeddingConfig {
  model?: string;          // default: 'Xenova/all-MiniLM-L6-v2' (384d, 80MB)
  dimensions?: number;     // default: 384
  batchSize?: number;      // default: 32
}
```

> Requires: `pnpm add @huggingface/transformers` (optional peer)
> Model auto-downloads to `~/.cache/huggingface/` on first `embed()` call

---

## `@rag-sdk/generator`

All factories return `Generator<M>`.

```ts
interface Generator<M extends Metadata = Metadata> {
  readonly id: string;
  readonly modelId: string;
  generate(request: GenerateRequest<M>, options?: GenerateOptions): Promise<GenerationResult<M>>;
  generateStream?(request: GenerateRequest<M>, options?: GenerateOptions): AsyncGenerator<string, void, undefined>;
}
```

### `createOpenAIGenerator(config)`

```ts
interface CreateOpenAIGeneratorConfig {
  apiKey: string;
  model?: string;          // default: 'gpt-4o-mini'
  temperature?: number;
  maxTokens?: number;
  baseURL?: string;
}
```

> Requires: `pnpm add openai` (optional peer)

### `createOllamaGenerator(config)`

```ts
interface OllamaGeneratorConfig {
  model: string;           // e.g. 'llama3.2'
  baseURL?: string;        // default: 'http://localhost:11434'
  temperature?: number;
  maxTokens?: number;
}
```

> Requires: Ollama running (no npm dep)
> Streaming format: newline-delimited JSON (SSE-compatible)

### `createCohereGenerator(config)`

```ts
interface CohereGeneratorConfig {
  apiKey: string;
  model?: string;          // default: 'command-r-plus'
  temperature?: number;
  maxTokens?: number;
}
```

> Requires: `pnpm add cohere-ai` (optional peer)

---

## `@rag-sdk/reranker`

All factories return `Reranker<M>`.

```ts
interface Reranker<M extends Metadata = Metadata> {
  readonly id: string;
  readonly modelId?: string;
  rerank(query: string, results: SearchResult<M>[], options?: RerankOptions): Promise<SearchResult<M>[]>;
}

interface RerankOptions {
  topN?: number;
}
```

### `createCohereReranker(config)`

```ts
interface CreateCohereRerankerConfig {
  apiKey: string;
  model?: string;          // default: 'rerank-english-v3.0'
}
```

> Requires: `pnpm add cohere-ai` (optional peer)

### `createLocalReranker(config?)`

```ts
interface LocalRerankerConfig {
  model?: string;          // default: 'Xenova/ms-marco-MiniLM-L-6-v2' (80MB)
  batchSize?: number;      // default: 32
}
```

> Requires: `pnpm add @huggingface/transformers` (optional peer)
> Format: `query [SEP] document` cross-encoder
> Model auto-downloads to `~/.cache/huggingface/` on first `rerank()` call

---

## `@rag-sdk/store`

All factories return `VectorStore<M>`.

```ts
interface VectorStore<M extends Metadata = Metadata> {
  readonly id?: string;
  readonly dimensions?: number;
  upsert(vectors: Vector<M>[], options?: { namespace?: string }): Promise<void>;
  query(embedding: number[], options: QueryOptions): Promise<SearchResult<M>[]>;
  delete(ids: string[], options?: { namespace?: string }): Promise<void>;
}
```

### `createMemoryStore(config)`

```ts
function createMemoryStore<M extends Metadata = Metadata>(config: {
  dimensions: number;
}): VectorStore<M>;
```

> Zero dependencies — pure in-memory arrays

### `createQdrantStore(config)`

```ts
async function createQdrantStore<M extends Metadata = Metadata>(config: {
  url: string;
  collectionName: string;
  dimensions: number;
}): Promise<VectorStore<M>>;
```

> Requires: `pnpm add @qdrant/js-client-rest` (optional peer)

### `createPineconeStore(config)`

```ts
async function createPineconeStore<M extends Metadata = Metadata>(config: {
  apiKey: string;
  indexName: string;
  dimensions: number;
}): Promise<VectorStore<M>>;
```

> Requires: `pnpm add @pinecone-database/pinecone` (optional peer)

### `createPgVectorStore(config)`

```ts
async function createPgVectorStore<M extends Metadata = Metadata>(config: {
  pool: pg.Pool;
  tableName?: string;
  dimensions: number;
}): Promise<VectorStore<M>>;
```

> Requires: `pnpm add pg` (optional peer)
> The SDK never owns the connection pool — you pass an existing `pg.Pool`

---

## `@rag-sdk/chunker`

### `SemanticChunker`

```ts
class SemanticChunker {
  constructor(provider: EmbeddingProvider, config?: {
    chunkSize?: number;    // default: 500
    threshold?: number;    // default: 0.5 — cosine similarity breakpoint
  });
}
```

> Splits text where semantic similarity drops below threshold
> Requires an `EmbeddingProvider` for computing similarities

---

## Port Hierarchy

```
EmbeddingProvider        — text → embedding (NOT generic, by design)
Generator<M>             — context + query → answer (with optional streaming)
Reranker<M>              — query + candidates → reranked candidates
VectorStore<M>           — upsert + query + delete
KeywordSearcher<M>       — keyword → candidates
Chunker                  — document → chunks
```

All ports with `<M>` carry metadata through the pipeline (`Document<M> → Chunk<M> → Vector<M> → SearchResult<M>`). `EmbeddingProvider` is intentionally non-generic — it operates on raw text, metadata is irrelevant at the embedding level.
