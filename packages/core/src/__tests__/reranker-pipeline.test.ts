import { describe, it, expect, vi } from 'vitest';
import { queryPipeline } from '../pipeline/query.js';
import { ProviderError } from '../errors.js';
import type { Metadata, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { QueryOptions } from '../types/store.js';
import type { Reranker, RerankOptions } from '../types/reranker.js';

function createMockProvider(opts?: { dimensions?: number }): EmbeddingProvider {
  const dims = opts?.dimensions ?? 3;
  return {
    id: 'mock-provider',
    modelId: 'mock-model',
    dimensions: dims,
    async embed(text: string): Promise<number[]> {
      return Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map((text) =>
        Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i),
      );
    },
  };
}

function createMockStore<M extends Metadata = Metadata>(opts?: { dimensions?: number }) {
  return {
    dimensions: opts?.dimensions,
    upsert: vi.fn(async () => {}),
    query: vi.fn(
      async (_embedding: number[], _options: QueryOptions) =>
        [] as SearchResult<M>[],
    ),
    delete: vi.fn(async () => {}),
  };
}

function createMockReranker<M extends Metadata = Metadata>(
  opts?: {
    reverse?: boolean;
    fail?: boolean;
    scores?: number[];
  },
): Reranker<M> {
  return {
    id: 'mock-reranker',
    modelId: 'rerank-model',
    rerank: vi.fn(
      async (
        _query: string,
        results: SearchResult<M>[],
        options?: RerankOptions,
      ): Promise<SearchResult<M>[]> => {
        if (opts?.fail) throw new Error('rerank failed');

        let reordered = opts?.reverse ? [...results].reverse() : [...results];

        if (opts?.scores) {
          reordered = reordered.map((r, i) => ({
            ...r,
            score: opts.scores![i] ?? r.score,
          }));
        }

        const topN = options?.topN;
        if (topN !== undefined && topN >= 0 && reordered.length > topN) {
          reordered = reordered.slice(0, topN);
        }

        return reordered;
      },
    ),
  };
}

describe('queryPipeline with reranker', () => {
  it('reorders results when reranker is provided', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
      { id: '2', score: 0.8, content: 'second', metadata: {} },
      { id: '3', score: 0.7, content: 'third', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker({ reverse: true });

    const result = await queryPipeline('test', { topK: 3 }, {
      provider,
      store,
      reranker,
    });

    expect(reranker.rerank).toHaveBeenCalledTimes(1);
    expect(reranker.rerank).toHaveBeenCalledWith(
      'test',
      mockResults,
      undefined,
    );
    expect(result.results.map((r) => r.id)).toEqual(['3', '2', '1']);
  });

  it('replaces scores with reranker scores', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
      { id: '2', score: 0.8, content: 'second', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker({ scores: [0.99, 0.95] });

    const result = await queryPipeline('test', { topK: 2 }, {
      provider,
      store,
      reranker,
    });

    expect(result.results[0].score).toBe(0.99);
    expect(result.results[1].score).toBe(0.95);
  });

  it('trims results to topN when specified', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
      { id: '2', score: 0.8, content: 'second', metadata: {} },
      { id: '3', score: 0.7, content: 'third', metadata: {} },
      { id: '4', score: 0.6, content: 'fourth', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker();

    const result = await queryPipeline(
      'test',
      { topK: 4, rerank: { topN: 2 } },
      { provider, store, reranker },
    );

    expect(result.results.length).toBe(2);
    expect(result.results[0].id).toBe('1');
    expect(result.results[1].id).toBe('2');
  });

  it('preserves all result fields through reranking', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
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
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker({ scores: [0.99] });

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
      reranker,
    });

    expect(result.results.length).toBe(1);
    const r = result.results[0];
    expect(r.id).toBe('chunk-1');
    expect(r.content).toBe('hello world');
    expect(r.metadata).toEqual({ source: 'test' });
    expect(r.documentId).toBe('doc-1');
    expect(r.chunkIndex).toBe(0);
    expect(r.namespace).toBe('ns-a');
  });

  it('skips reranking when no reranker is provided', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
    });

    expect(result.results).toEqual(mockResults);
  });

  it('wraps reranker errors in ProviderError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker({ fail: true });

    await expect(
      queryPipeline('test', { topK: 1 }, { provider, store, reranker }),
    ).rejects.toThrow(ProviderError);
  });

  it('re-throws existing ProviderError without wrapping', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker();
    const existingError = new ProviderError('mock-reranker', 'rerank', new Error('network'));
    (reranker.rerank as ReturnType<typeof vi.fn>).mockRejectedValue(existingError);

    await expect(
      queryPipeline('test', { topK: 1 }, { provider, store, reranker }),
    ).rejects.toThrow(existingError);
  });

  it('forwards rerank options to reranker', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'first', metadata: {} },
      { id: '2', score: 0.8, content: 'second', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const reranker = createMockReranker();

    await queryPipeline(
      'test',
      { topK: 2, rerank: { topN: 1 } },
      { provider, store, reranker },
    );

    expect(reranker.rerank).toHaveBeenCalledWith(
      'test',
      mockResults,
      { topN: 1 },
    );
  });
});
