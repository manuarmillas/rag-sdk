/**
 * RAG SDK — Usage Examples (V3)
 * ================================
 *
 * Demonstrates the full V2 + V3 API surface:
 *   - Ingest, query, generate, stream
 *   - Local reranker (CPU, free) and Cohere reranker
 *   - Ollama, Cohere, VoyageAI, and local embedding providers
 *   - Markdown and semantic chunking
 *   - Hybrid search (vector + keyword)
 *   - Production stores (Qdrant, Pinecone, pgvector)
 *   - Custom metadata and error handling
 *
 * Requirements:
 *   pnpm add @rag-sdk/core @rag-sdk/embedding @rag-sdk/store
 *   pnpm add @rag-sdk/generator @rag-sdk/reranker @rag-sdk/chunker
 */

import { rag, MarkdownChunker, ConfigurationError, DimensionMismatchError } from '@rag-sdk/core';
import { createOpenAI, createOllamaEmbedding, createCohereEmbedding, createVoyageEmbedding, createLocalEmbedding } from '@rag-sdk/embedding';
import { createMemoryStore, createQdrantStore } from '@rag-sdk/store';
import { createOpenAIGenerator, createOllamaGenerator } from '@rag-sdk/generator';
import { createCohereReranker, createLocalReranker } from '@rag-sdk/reranker';
import { SemanticChunker } from '@rag-sdk/chunker';

// ---------------------------------------------------------------------------
// 1. Basic setup — ingest + query (retrieve-only)
// ---------------------------------------------------------------------------

async function basicRetrieve() {
  const sdk = rag({
    provider: createOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'text-embedding-3-small',
    }),
    store: createMemoryStore({ dimensions: 1536 }),
  });

  // Ingest documents
  await sdk.ingest([
    { content: 'RAG SDK is a TypeScript toolkit for building RAG pipelines.', metadata: { source: 'docs' } },
    { content: 'It supports multiple vector stores: Qdrant, Pinecone, pgvector, and in-memory.', metadata: { source: 'docs' } },
    { content: 'The SDK follows hexagonal architecture with pluggable adapters.', metadata: { source: 'blog' } },
  ]);

  // Retrieve relevant chunks
  const result = await sdk.query('What vector stores does RAG SDK support?', { topK: 2 });
  console.log('🔍 Retrieve results:');
  for (const r of result.results) {
    console.log(`  [${r.score.toFixed(3)}] ${r.content}`);
  }
}

// ---------------------------------------------------------------------------
// 2. RAG generation — retrieve + generate answer
// ---------------------------------------------------------------------------

async function basicGenerate() {
  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    generator: createOpenAIGenerator({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o-mini',
    }),
  });

  await sdk.ingest([
    { content: 'TypeScript adds static type checking to JavaScript.', metadata: { source: 'docs' } },
    { content: 'TypeScript compiles to plain JavaScript that runs anywhere.', metadata: { source: 'docs' } },
  ]);

  const answer = await sdk.generate('What is TypeScript?');
  console.log('🤖 Answer:', answer.answer);
  console.log('   Tokens used:', answer.usage);
}

// ---------------------------------------------------------------------------
// 3. Streaming generation — token-by-token
// ---------------------------------------------------------------------------

async function streamingGenerate() {
  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    generator: createOpenAIGenerator({ apiKey: process.env.OPENAI_API_KEY! }),
  });

  await sdk.ingest([
    { content: 'Deno is a modern runtime for JavaScript and TypeScript.', metadata: { source: 'docs' } },
  ]);

  console.log('🌊 Streaming answer:');
  for await (const token of sdk.generateStream('Explain Deno in one sentence.')) {
    process.stdout.write(token);
  }
  console.log(); // newline
}

// ---------------------------------------------------------------------------
// 4. Markdown chunking — heading-aware splitting
// ---------------------------------------------------------------------------

async function markdownChunking() {
  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    chunker: new MarkdownChunker(), // zero-dependency, ships with core
  });

  await sdk.ingest([
    {
      content: `# Getting Started
RAG SDK is easy to set up.

## Installation
Run \`pnpm add @rag-sdk/core\` to install.

## Configuration
Create a provider and store, then call \`rag()\`.

### Environment Variables
Set \`OPENAI_API_KEY\` in your environment.`,
      metadata: { source: 'readme' },
    },
  ]);

  const result = await sdk.query('How do I configure the SDK?');
  for (const r of result.results) {
    console.log('📝 Markdown chunk:', r.metadata.heading, '→', r.content.slice(0, 80));
  }
}

// ---------------------------------------------------------------------------
// 5. Semantic chunking — embedding-based breakpoints
// ---------------------------------------------------------------------------

async function semanticChunking() {
  const provider = createOpenAI({ apiKey: process.env.OPENAI_API_KEY! });

  const sdk = rag({
    provider,
    store: createMemoryStore({ dimensions: 1536 }),
    chunker: new SemanticChunker(provider, { chunkSize: 500, threshold: 0.6 }),
  });

  await sdk.ingest([
    {
      content: `Quantum computing leverages qubits. Unlike classical bits, qubits can exist
in superposition states, enabling parallel computation. Quantum gates manipulate
these states to perform algorithms. Shor's algorithm factors large numbers efficiently.`,
      metadata: { source: 'research' },
    },
  ]);

  const result = await sdk.query('How does quantum computing work?');
  console.log('🧠 Semantic chunk:', result.results[0]?.content.slice(0, 100));
}

// ---------------------------------------------------------------------------
// 6. Hybrid search — vector + keyword combined
// ---------------------------------------------------------------------------

async function hybridSearch() {
  // A KeywordSearcher can be any object implementing the port.
  // Here we show a toy example; in production you'd use a full-text index.
  const toyKeywordSearcher = {
    id: 'toy-keyword',
    // eslint-disable-next-line @typescript-eslint/require-await
    async keywordSearch(text: string, _options: Record<string, unknown>) {
      // In production: delegate to Elasticsearch, Meilisearch, or Qdrant payload index.
      // This toy returns a mock result for demonstration.
      if (text.toLowerCase().includes('bolt')) {
        return [{
          id: 'k1',
          content: 'Usain Bolt holds the 100m world record at 9.58 seconds.',
          score: 0.9,
          metadata: { source: 'sports' },
          documentId: 'doc-1',
          chunkIndex: 0,
        }];
      }
      return [];
    },
  };

  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    keywordSearcher: toyKeywordSearcher,
  });

  await sdk.ingest([
    { content: 'Usain Bolt is a Jamaican sprinter who won eight Olympic gold medals.', metadata: { source: 'sports' } },
    { content: 'The 100-meter dash world record is one of the most prestigious in athletics.', metadata: { source: 'sports' } },
  ]);

  // Hybrid search merges vector + keyword results via Reciprocal Rank Fusion (RRF)
  const result = await sdk.query("Who is Bolt and what's his record?", {
    topK: 3,
    hybrid: { enabled: true, vectorWeight: 1, keywordWeight: 1 },
  });
  console.log('🔀 Hybrid results:');
  for (const r of result.results) {
    console.log(`  [${r.score.toFixed(3)}] ${r.content}`);
  }
}

// ---------------------------------------------------------------------------
// 7. Reranking — post-retrieval relevance boost
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function reranking() {
  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    reranker: createCohereReranker({
      apiKey: process.env.COHERE_API_KEY!,
      model: 'rerank-english-v3.0',
    }),
  });

  await sdk.ingest([
    { content: 'Docker containers package applications with their dependencies.', metadata: { source: 'devops' } },
    { content: 'Kubernetes orchestrates containerized applications across clusters.', metadata: { source: 'devops' } },
    { content: 'Virtual machines emulate entire operating systems with a hypervisor.', metadata: { source: 'devops' } },
  ]);

  // Reranker reorders results by relevance, not just cosine similarity
  const result = await sdk.query('How do containers differ from VMs?', { topK: 3, rerank: { topN: 2 } });
  console.log('🎯 Reranked results (score = reranker relevance):');
  for (const r of result.results) {
    console.log(`  [${r.score.toFixed(3)}] ${r.content}`);
  }
}

// ---------------------------------------------------------------------------
// 8. Production store — Qdrant
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function productionQdrant() {
  // Requires: pnpm add @qdrant/js-client-rest
  // Requires: a running Qdrant instance (default: http://localhost:6333)

  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: await createQdrantStore({
      url: process.env.QDRANT_URL ?? 'http://localhost:6333',
      collectionName: 'my-documents',
      dimensions: 1536,
    }),
  });

  await sdk.ingest([
    { content: 'Qdrant is a vector similarity search engine written in Rust.', metadata: { source: 'docs' } },
  ]);

  const result = await sdk.query('What is Qdrant?');
  console.log('🗄️  Qdrant result:', result.results[0]?.content);
}

// ---------------------------------------------------------------------------
// 9. Production store — pgvector (with existing pg.Pool)
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-unused-vars, @typescript-eslint/require-await
async function productionPgVector() {
  // Requires: pnpm add pg
  // Requires: a running PostgreSQL instance with pgvector extension

  // import pg from 'pg';
  // const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

  // const sdk = rag({
  //   provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
  //   store: createPgVectorStore({
  //     pool,               // user manages connection lifecycle
  //     tableName: 'embeddings',
  //     dimensions: 1536,
  //   }),
  // });
  //
  // await sdk.ingest([{ content: 'pgvector brings vector search to PostgreSQL.' }]);
  // await pool.end();  // user closes the pool

  console.log('📦 pgvector: uncomment the code above and provide a pg.Pool to use.');
}

// ---------------------------------------------------------------------------
// 10. Error handling
// ---------------------------------------------------------------------------

async function errorHandling() {
  // Without a generator, generate() throws ConfigurationError
  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    // generator NOT configured
  });

  try {
    await sdk.generate('Hello');
  } catch (err) {
    if (err instanceof ConfigurationError) {
      console.log('❌ Expected:', err.message);
      // → "Generator is required for generate()"
    }
  }
}

// ---------------------------------------------------------------------------
// 11. Custom metadata
// ---------------------------------------------------------------------------

type MyMetadata = Record<string, unknown> & {
  source: string;
  author?: string;
  tags?: string[];
};

async function customMetadata() {
  const sdk = rag<MyMetadata>({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore<MyMetadata>({ dimensions: 1536 }),
  });

  await sdk.ingest([
    {
      content: 'TypeScript generics enable type-safe, reusable components.',
      metadata: { source: 'blog', author: 'Jane Doe', tags: ['typescript', 'generics'] },
    },
  ]);

  // Filter by metadata
  const result = await sdk.query('generics', {
    filter: { author: 'Jane Doe' },
  });

  console.log('🏷️  Filtered by author:', result.results.length, 'results');
}

// ---------------------------------------------------------------------------
// Run all examples (comment out those requiring external services)
// ---------------------------------------------------------------------------

// ─── V3: Local reranker (CPU, free) ─────────────────────────────────────────
async function localReranker() {
  const sdk = rag({
    provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
    store: createMemoryStore({ dimensions: 1536 }),
    reranker: createLocalReranker(), // 80MB cross-encoder, CPU
  });

  await sdk.ingest([{ content: 'Document about machine learning and neural networks.' }]);
  const result = await sdk.query('neural networks', { topK: 10, rerank: { topN: 3 } });
  console.log('🧠 Local reranker (top result):', result.results[0]?.content);
}

// ─── V3: Ollama embedding + generator ──────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function ollamaWorkflow() {
  const sdk = rag({
    provider: createOllamaEmbedding({ model: 'nomic-embed-text' }),
    store: createMemoryStore({ dimensions: 768 }),
    generator: createOllamaGenerator({ model: 'llama3.2' }),
  });

  await sdk.ingest([{ content: 'RAG SDK supports local-first inference with Ollama.' }]);
  const result = await sdk.generate('What does RAG SDK support?');
  console.log('🦙 Ollama answer:', result.answer);
}

// ─── V3: Local embedding + local reranker (100% offline) ────────────────────
async function fullyLocal() {
  const sdk = rag({
    provider: createLocalEmbedding(),
    store: createMemoryStore({ dimensions: 384 }),
    reranker: createLocalReranker(),
  });

  await sdk.ingest([{ content: 'RAG SDK runs fully offline with no API keys.' }]);
  const result = await sdk.query('offline', { topK: 5, rerank: { topN: 3 } });
  console.log('🏠 Fully local result:', result.results[0]?.content);
}

// ─── V3: Cohere embedding ──────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function cohereEmbedding() {
  const sdk = rag({
    provider: createCohereEmbedding({ apiKey: process.env.COHERE_API_KEY! }),
    store: createMemoryStore({ dimensions: 1024 }),
  });

  await sdk.ingest([{ content: 'Cohere provides high-quality multilingual embeddings.' }]);
  const result = await sdk.query('multilingual');
  console.log('🔗 Cohere embedding result:', result.results[0]?.content);
}

// ─── V3: VoyageAI embedding ────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function voyageEmbedding() {
  const sdk = rag({
    provider: createVoyageEmbedding({ apiKey: process.env.VOYAGE_API_KEY! }),
    store: createMemoryStore({ dimensions: 512 }),
  });

  await sdk.ingest([{ content: 'VoyageAI optimizes embeddings for retrieval quality.' }]);
  const result = await sdk.query('retrieval quality');
  console.log('🚢 Voyage embedding result:', result.results[0]?.content);
}

// ─── V3: Dimension mismatch error ──────────────────────────────────────────
async function dimError() {
  try {
    const sdk = rag({
      provider: createLocalEmbedding(),        // 384d
      store: createMemoryStore({ dimensions: 1536 }), // expects 1536d
    });
    await sdk.ingest([{ content: 'This will throw a dimension mismatch error.' }]);
  } catch (err) {
    if (err instanceof DimensionMismatchError) {
      console.log('⚠️  Caught dimension mismatch:', err.message.slice(0, 60));
    }
  }
}

// ─── Main ──────────────────────────────────────────────────────────────────

async function main() {
  console.log('=== RAG SDK V3 — Usage Examples ===\n');

  await basicRetrieve();
  await basicGenerate();
  await streamingGenerate();
  await markdownChunking();
  await semanticChunking();
  await hybridSearch();
  // await reranking();            // requires COHERE_API_KEY
  // await productionQdrant();     // requires Qdrant running locally
  // await productionPgVector();   // requires PostgreSQL + pgvector
  await errorHandling();
  await customMetadata();

  // ── V3 features (local / no setup needed) ──
  await localReranker();        // requires @huggingface/transformers
  await fullyLocal();           // requires @huggingface/transformers
  await dimError();             // no setup needed

  // ── V3 features (require external services) ──
  // await ollamaWorkflow();    // requires Ollama running
  // await cohereEmbedding();   // requires COHERE_API_KEY
  // await voyageEmbedding();   // requires VOYAGE_API_KEY
}

main().catch(console.error);
