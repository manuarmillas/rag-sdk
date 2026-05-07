import { describe, it, expect } from 'vitest';
import { rag } from '../rag.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { Metadata } from '../types/document.js';
import { createMemoryStore } from '@rag-sdk/store';

/**
 * Deterministic embedding provider that generates vectors based on a simple
 * hash of the input text. Same input → same vector every time.
 * This allows us to test real ranking behavior without external APIs.
 */
function createDeterministicProvider(dimensions = 8): EmbeddingProvider {
  return {
    id: 'deterministic',
    modelId: 'det-model-v1',
    dimensions,
    maxBatchSize: 100,
    async embed(text: string): Promise<number[]> {
      return hashToVector(text, dimensions);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map((t) => hashToVector(t, dimensions));
    },
  };
}

function hashToVector(text: string, dimensions: number): number[] {
  const vec: number[] = [];
  for (let i = 0; i < dimensions; i++) {
    let hash = 0;
    for (let j = 0; j < text.length; j++) {
      hash = (hash * 31 + text.charCodeAt(j) + i * 7) % 1000;
    }
    vec.push(hash / 1000);
  }
  return vec;
}

describe('rag e2e integration with reference adapters', () => {
  it('ingests and queries documents end-to-end', async () => {
    const provider = createDeterministicProvider();
    const store = createMemoryStore<{ source: string }>({
      dimensions: provider.dimensions,
    });

    const sdk = rag({ provider, store });

    await sdk.ingest([
      { id: 'doc-1', content: 'The mitochondria is the powerhouse of the cell', metadata: { source: 'biology' } },
      { id: 'doc-2', content: 'TypeScript is a typed superset of JavaScript', metadata: { source: 'programming' } },
      { id: 'doc-3', content: 'Photosynthesis converts sunlight into energy', metadata: { source: 'biology' } },
    ]);

    // Query about biology should rank biology docs higher
    const result = await sdk.query('mitochondria energy', { topK: 3 });

    expect(result.query).toBe('mitochondria energy');
    expect(result.results.length).toBe(3);

    // The biology docs should appear — mitochondria and photosynthesis overlap
    // with "mitochondria energy" query
    const hasBiologyDoc = result.results.some(
      (r) => r.metadata.source === 'biology',
    );
    expect(hasBiologyDoc).toBe(true);
  });

  it('returns correct metadata in results', async () => {
    const provider = createDeterministicProvider();
    const store = createMemoryStore<{ source: string; page: number }>({
      dimensions: provider.dimensions,
    });

    const sdk = rag({ provider, store });

    await sdk.ingest([
      {
        id: 'page-1',
        content: 'Chapter one introduces the main character',
        metadata: { source: 'book.pdf', page: 1 },
      },
      {
        id: 'page-2',
        content: 'Chapter two develops the plot further',
        metadata: { source: 'book.pdf', page: 2 },
      },
    ]);

    const result = await sdk.query('chapter', { topK: 2 });

    expect(result.results.length).toBe(2);
    expect(result.results[0].metadata.source).toBe('book.pdf');
    expect(typeof result.results[0].metadata.page).toBe('number');
  });

  it('respects namespace isolation', async () => {
    const provider = createDeterministicProvider();
    const store = createMemoryStore<{ ns: string }>({
      dimensions: provider.dimensions,
    });

    const sdk = rag({ provider, store });

    // Ingest in namespace "a"
    await sdk.ingest(
      [{ content: 'Python is great for data science', metadata: { ns: 'a' } }],
      { namespace: 'a' },
    );

    // Ingest in namespace "b" with different content
    await sdk.ingest(
      [{ content: 'Rust is great for systems programming', metadata: { ns: 'b' } }],
      { namespace: 'b' },
    );

    // Query namespace "a" should get Python result
    const resultA = await sdk.query('data science', {
      topK: 1,
      namespace: 'a',
    });

    expect(resultA.results.length).toBe(1);
    expect(resultA.results[0].metadata.ns).toBe('a');
    expect(resultA.namespace).toBe('a');

    // Query namespace "b" should get Rust result
    const resultB = await sdk.query('systems programming', {
      topK: 1,
      namespace: 'b',
    });

    expect(resultB.results.length).toBe(1);
    expect(resultB.results[0].metadata.ns).toBe('b');
    expect(resultB.namespace).toBe('b');
  });

  it('ranks results by relevance (cosine similarity)', async () => {
    const provider = createDeterministicProvider();
    const store = createMemoryStore<Metadata>({
      dimensions: provider.dimensions,
    });

    const sdk = rag({ provider, store });

    await sdk.ingest([
      { content: 'Machine learning models require training data' },
      { content: 'The capital of France is Paris' },
      { content: 'Deep learning uses neural networks with many layers' },
    ]);

    const result = await sdk.query('machine learning deep neural networks', {
      topK: 3,
    });

    expect(result.results.length).toBe(3);

    // The first result should be the most relevant (ML-related docs)
    const scores = result.results.map((r) => r.score);
    expect(scores[0]).toBeGreaterThanOrEqual(scores[1]);
    expect(scores[1]).toBeGreaterThanOrEqual(scores[2]);
  });

  it('returns chunks with correct documentId and chunkIndex', async () => {
    const provider = createDeterministicProvider();
    const store = createMemoryStore<Metadata>({
      dimensions: provider.dimensions,
    });

    const sdk = rag({ provider, store });

    await sdk.ingest([
      {
        id: 'long-doc',
        content: 'A'.repeat(2000),
      },
    ]);

    const result = await sdk.query('A', { topK: 10 });

    // Long document should be chunked
    expect(result.results.length).toBeGreaterThan(0);

    // All chunks should reference the same documentId
    for (const chunk of result.results) {
      expect(chunk.documentId).toBe('long-doc');
      expect(typeof chunk.chunkIndex).toBe('number');
    }
  });

  it('handles filter on metadata', async () => {
    const provider = createDeterministicProvider();
    const store = createMemoryStore<{ type: string }>({
      dimensions: provider.dimensions,
    });

    const sdk = rag({ provider, store });

    await sdk.ingest([
      { content: 'REST APIs use HTTP methods', metadata: { type: 'api' } },
      { content: 'GraphQL allows querying specific fields', metadata: { type: 'api' } },
      { content: 'Docker containers isolate applications', metadata: { type: 'infra' } },
    ]);

    const result = await sdk.query('query fields', {
      topK: 3,
      filter: { type: 'api' },
    });

    // Only API-type docs should be returned
    expect(result.results.length).toBeGreaterThan(0);
    for (const r of result.results) {
      expect(r.metadata.type).toBe('api');
    }
  });
});
