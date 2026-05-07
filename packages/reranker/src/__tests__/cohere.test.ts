import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createCohereReranker } from '../cohere.js';
import { ProviderError } from '@rag-sdk/core';
import type { SearchResult } from '@rag-sdk/core';

// Mock the cohere-ai module
vi.mock('cohere-ai', () => {
  return {
    CohereClient: vi.fn(),
  };
});

import { CohereClient } from 'cohere-ai';

interface MockRerankArgs {
  query: string;
  documents: string[];
  model: string;
  topN?: number;
}

let lastRerankCall: MockRerankArgs | undefined;

function createMockClient(opts?: { failRerank?: boolean }) {
  const mockRerank = vi.fn(async () => ({
    results: [
      { index: 1, relevanceScore: 0.98 },
      { index: 0, relevanceScore: 0.75 },
    ],
  }));

  (CohereClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
    rerank: vi.fn(async (...args: unknown[]) => {
      lastRerankCall = args[0] as MockRerankArgs;
      if (opts?.failRerank) throw new Error('rerank error');
      return mockRerank();
    }),
  }));

  return { mockRerank };
}

describe('createCohereReranker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    lastRerankCall = undefined;
  });

  it('returns a Reranker with correct id and modelId', () => {
    createMockClient();
    const reranker = createCohereReranker({ model: 'rerank-multilingual-v3.0' });
    expect(reranker.id).toBe('cohere');
    expect(reranker.modelId).toBe('rerank-multilingual-v3.0');
  });

  it('uses default model when not specified', () => {
    createMockClient();
    const reranker = createCohereReranker();
    expect(reranker.modelId).toBe('rerank-english-v3.0');
  });

  it('reorders results based on Cohere response', async () => {
    createMockClient();
    const reranker = createCohereReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first document', metadata: {} },
      { id: 'doc-2', score: 0.8, content: 'second document', metadata: {} },
    ];

    const reordered = await reranker.rerank('test query', results);

    expect(reordered.length).toBe(2);
    expect(reordered[0].id).toBe('doc-2');
    expect(reordered[1].id).toBe('doc-1');
  });

  it('replaces scores with Cohere relevance scores', async () => {
    createMockClient();
    const reranker = createCohereReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
      { id: 'doc-2', score: 0.8, content: 'second', metadata: {} },
    ];

    const reordered = await reranker.rerank('test query', results);

    expect(reordered[0].score).toBe(0.98);
    expect(reordered[1].score).toBe(0.75);
  });

  it('preserves all result fields through reranking', async () => {
    createMockClient();
    const reranker = createCohereReranker();

    const results: SearchResult[] = [
      {
        id: 'chunk-1',
        score: 0.9,
        content: 'hello world',
        metadata: { source: 'test' },
        documentId: 'doc-1',
        chunkIndex: 0,
        namespace: 'ns-a',
      },
    ];

    const reordered = await reranker.rerank('test query', results);

    expect(reordered.length).toBe(1);
    const r = reordered[0];
    expect(r.id).toBe('chunk-1');
    expect(r.content).toBe('hello world');
    expect(r.metadata).toEqual({ source: 'test' });
    expect(r.documentId).toBe('doc-1');
    expect(r.chunkIndex).toBe(0);
    expect(r.namespace).toBe('ns-a');
  });

  it('passes topN to the Cohere API', async () => {
    createMockClient();
    const reranker = createCohereReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
      { id: 'doc-2', score: 0.8, content: 'second', metadata: {} },
    ];

    await reranker.rerank('test query', results, { topN: 1 });

    expect(lastRerankCall).toBeDefined();
    expect(lastRerankCall!.topN).toBe(1);
  });

  it('does not pass topN when undefined', async () => {
    createMockClient();
    const reranker = createCohereReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
    ];

    await reranker.rerank('test query', results);

    expect(lastRerankCall).toBeDefined();
    expect(lastRerankCall!.topN).toBeUndefined();
  });

  it('returns empty array when input results are empty', async () => {
    createMockClient();
    const reranker = createCohereReranker();

    const reordered = await reranker.rerank('test query', []);

    expect(reordered).toEqual([]);
  });

  it('wraps errors in ProviderError', async () => {
    createMockClient({ failRerank: true });
    const reranker = createCohereReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
    ];

    await expect(
      reranker.rerank('test query', results),
    ).rejects.toThrow(ProviderError);
  });
});
