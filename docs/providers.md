# Provider Guide

Choosing the right provider for your RAG pipeline. All providers implement the same ports — swap without changing application code.

## Embedding Providers

### Comparison Matrix

| Provider | Dimensions | Max Batch | Model | Speed | Cost | Offline | Install |
|----------|-----------|-----------|-------|-------|------|---------|---------|
| **Local** (`all-MiniLM-L6-v2`) | 384 | ~64 | ONNX (80MB) | ★★☆ CPU | Free ✅ | ✅ | `@huggingface/transformers` |
| **OpenAI** (`text-embedding-3-small`) | 1536 | 2048 | Cloud | ★★★ | $0.02/M tokens | ❌ | `openai` |
| **OpenAI** (`text-embedding-3-large`) | 3072 | 2048 | Cloud | ★★★ | $0.13/M tokens | ❌ | `openai` |
| **Ollama** (`nomic-embed-text`) | 768 | — | Local GPU/CPU | ★★☆ | Free ✅ | ✅ | None (REST API) |
| **Ollama** (`mxbai-embed-large`) | 1024 | — | Local GPU/CPU | ★★☆ | Free ✅ | ✅ | None (REST API) |
| **Cohere** (`embed-english-v3.0`) | 1024 | 96 | Cloud | ★★★ | $0.10/M tokens | ❌ | `cohere-ai` |
| **VoyageAI** (`voyage-3-lite`) | 512 | 128 | Cloud | ★★★ | $0.06/M tokens | ❌ | `voyageai` |
| **VoyageAI** (`voyage-3`) | 1024 | 128 | Cloud | ★★★ | $0.12/M tokens | ❌ | `voyageai` |

### When to Use Each

**Local (`createLocalEmbedding`)**
- Prototyping, testing, CI pipelines — no API keys, no network
- Privacy-sensitive data — embeddings never leave your machine
- Serverless / edge — runs on WASM, no Python or GPU needed
- Quality: lower than cloud providers (384d vs 1536d+), but sufficient for semantic search on small/medium datasets
- Trade-off: first call downloads 80MB model (~2-5s), ~200MB RAM

**OpenAI (`createOpenAI`)**
- Best quality-to-cost ratio for general English text
- Largest ecosystem, most tooling
- Dimensions: `text-embedding-3-small` (1536d, cheap) or `text-embedding-3-large` (3072d, best quality)
- Trade-off: requires API key, data leaves your environment

**Ollama (`createOllamaEmbedding`)**
- Run ANY embedding model locally (pull from Ollama registry)
- Zero npm deps — pure REST API to `localhost:11434`
- Models: `nomic-embed-text` (768d), `mxbai-embed-large` (1024d), `bge-large` (1024d)
- Trade-off: requires Ollama installed and running, model management is manual

**Cohere (`createCohereEmbedding`)**
- Excellent for multilingual and specialized domains
- `inputType` parameter: `search_document`, `search_query`, `classification`, `clustering`
- Trade-off: lower batch limit (96), separate from OpenAI ecosystem

**VoyageAI (`createVoyageEmbedding`)**
- Optimized for retrieval quality — consistently ranks high in MTEB benchmarks
- `voyage-3-lite` (512d) is the best cost/quality ratio among cloud providers
- Trade-off: smaller ecosystem, fewer integrations

---

## Generator Providers

| Provider | Default Model | Streaming | Max Context | Cost | Offline | Install |
|----------|--------------|-----------|-------------|------|---------|---------|
| **OpenAI** | `gpt-4o-mini` | ✅ SSE | 128K | $0.15/1M tokens | ❌ | `openai` |
| **Ollama** | `llama3.2` | ✅ SSE | model-dependent | Free ✅ | ✅ | None (REST API) |
| **Cohere** | `command-r-plus` | ✅ | 128K | $2.50/1M tokens | ❌ | `cohere-ai` |

### When to Use Each

**OpenAI (`createOpenAIGenerator`)**
- Best quality for general-purpose generation
- `gpt-4o-mini` for cost-sensitive, `gpt-4o` for complex reasoning
- Trade-off: most expensive at scale, data leaves your environment

**Ollama (`createOllamaGenerator`)**
- Run LLMs locally — Llama 3.2, Mistral, Gemma, Phi, DeepSeek, Qwen
- Streaming via SSE, token-by-token
- Zero npm deps — pure REST API
- Trade-off: requires Ollama installed, quality depends on model choice, needs GPU for decent speed on >7B models

**Cohere (`createCohereGenerator`)**
- `command-r-plus` optimized for RAG workflows — handles long contexts well
- Good at following system prompts and structured output
- Trade-off: most expensive per-token, smaller model selection than OpenAI

---

## Reranker Providers

| Provider | Model | Speed | Cost | Offline | Install |
|----------|-------|-------|------|---------|---------|
| **Local** | `ms-marco-MiniLM-L-6-v2` (80MB) | ★★☆ CPU | Free ✅ | ✅ | `@huggingface/transformers` |
| **Cohere** | `rerank-english-v3.0` | ★★★ | $2/1K searches | ❌ | `cohere-ai` |

### When to Use Each

**Local (`createLocalReranker`)**
- Privacy-first — query + documents never leave your machine
- Good enough for most use cases (~85-90% of Cohere quality)
- Lazy loading: model downloads on first `rerank()` call
- Batch size configurable (default 32) — larger = faster, more RAM
- Trade-off: first call has ~2-5s download + ~500ms warmup, ~200MB RAM per loaded model

**Cohere (`createCohereReranker`)**
- Best quality — industry-leading reranking accuracy
- No cold start — instant first call
- Trade-off: requires API key, network call latency, cost per search

---

## Provider Combinations — Recommended Stacks

### MVP / Prototyping (zero cost, zero setup)
```ts
provider:  createLocalEmbedding()           // 384d, CPU
generator: createOllamaGenerator(...)        // requires Ollama
reranker:  createLocalReranker()             // CPU
store:     createMemoryStore(...)
```

### Production (cloud, best quality)
```ts
provider:  createOpenAI({ apiKey, model: 'text-embedding-3-large' })   // 3072d
generator: createOpenAIGenerator({ apiKey, model: 'gpt-4o-mini' })     // 128K ctx
reranker:  createCohereReranker({ apiKey })                            // best quality
store:     createQdrantStore({ url, collectionName, dimensions: 3072 })
```

### Privacy-First (fully offline)
```ts
provider:  createOllamaEmbedding({ model: 'mxbai-embed-large' })       // 1024d, local
generator: createOllamaGenerator({ model: 'llama3.2' })                // local
reranker:  createLocalReranker()                                        // CPU
store:     createMemoryStore(...)
```

### Cost-Optimized
```ts
provider:  createVoyageEmbedding({ apiKey, model: 'voyage-3-lite' })   // 512d, $0.06/M
generator: createOpenAIGenerator({ apiKey, model: 'gpt-4o-mini' })     // cheap gen
reranker:  createLocalReranker()                                        // free rerank
store:     createQdrantStore(...)
```

---

## Mix and Match

All providers implement the same ports. Mix any embedding with any generator with any reranker:

```ts
// VoyageAI embeddings + Ollama local generation + Cohere cloud reranker
const sdk = rag({
  provider: createVoyageEmbedding({ apiKey: 'voy-xxx' }),
  generator: createOllamaGenerator({ model: 'llama3.2' }),
  reranker: createCohereReranker({ apiKey: 'cohere-xxx' }),
  store: createMemoryStore({ dimensions: 1024 }),
});
```

The SDK doesn't care — it talks to ports, not implementations.
