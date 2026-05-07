import { describe, it, expect, vi } from 'vitest';
import { queryPipeline } from '../pipeline/query.js';
import { StoreError } from '../errors.js';
import type { Metadata, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { QueryOptions } from '../types/store.js';
import type { KeywordSearcher, HybridOptions } from '../types/searcher.js';

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

function createMockKeywordSearcher<M extends Metadata = Metadata>(
  results: SearchResult<M>[],
  opts?: { fail?: boolean },
): KeywordSearcher<M> {
  return {
    id: 'mock-keyword',
    keywordSearch: vi.fn(async (_text: string, _options: QueryOptions) => {
      if (opts?.fail) throw new Error('keyword search failed');
      return results;
    }),
  };
}

describe('queryPipeline with keywordSearcher', () => {
  it('fuses vector and keyword results with RRF', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'vector first', metadata: {} },
      { id: '2', score: 0.8, content: 'vector second', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordResults: SearchResult[] = [
      { id: '2', score: 0.95, content: 'keyword first', metadata: { source: 'kw' } },
      { id: '3', score: 0.85, content: 'keyword second', metadata: {} },
    ];
    const keywordSearcher = createMockKeywordSearcher(keywordResults);

    const result = await queryPipeline('test', { topK: 2 }, {
      provider,
      store,
      keywordSearcher,
    });

    expect(keywordSearcher.keywordSearch).toHaveBeenCalledTimes(1);
    expect(keywordSearcher.keywordSearch).toHaveBeenCalledWith(
      'test',
      expect.objectContaining({ topK: 2 }),
    );

    // All three unique IDs should be present
    const ids = result.results.map((r) => r.id);
    expect(ids).toContain('1');
    expect(ids).toContain('2');
    expect(ids).toContain('3');

    // ID '2' appears in both lists, so it should have the highest fused score
    expect(result.results[0].id).toBe('2');

    // Content from vector result should be preserved for duplicate ID
    expect(result.results[0].content).toBe('vector second');
  });

  it('uses default RRF weights and rrfK', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: 'a', score: 0.9, content: 'a', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordResults: SearchResult[] = [
      { id: 'b', score: 0.8, content: 'b', metadata: {} },
    ];
    const keywordSearcher = createMockKeywordSearcher(keywordResults);

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
      keywordSearcher,
    });

    expect(result.results.length).toBe(2);
    // Default weights are 1, rrfK is 60
    // a: 1/(60+1) = 0.01639...
    // b: 1/(60+1) = 0.01639...
    // Both equal, so order is stable (a first because it was inserted first in map iteration)
    expect(result.results[0].score).toBeCloseTo(1 / 61, 5);
    expect(result.results[1].score).toBeCloseTo(1 / 61, 5);
  });

  it('respects custom weights and rrfK', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: 'a', score: 0.9, content: 'a', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordResults: SearchResult[] = [
      { id: 'b', score: 0.8, content: 'b', metadata: {} },
    ];
    const keywordSearcher = createMockKeywordSearcher(keywordResults);

    const hybrid: HybridOptions = {
      vectorWeight: 2,
      keywordWeight: 0.5,
      rrfK: 10,
    };

    const result = await queryPipeline('test', { topK: 1, hybrid }, {
      provider,
      store,
      keywordSearcher,
    });

    // a: 2/(10+1) = 0.1818...
    // b: 0.5/(10+1) = 0.04545...
    expect(result.results[0].id).toBe('a');
    expect(result.results[0].score).toBeCloseTo(2 / 11, 5);
    expect(result.results[1].score).toBeCloseTo(0.5 / 11, 5);
  });

  it('skips keyword search when hybrid.enabled is false', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'vector', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordSearcher = createMockKeywordSearcher([
      { id: '2', score: 0.8, content: 'keyword', metadata: {} },
    ]);

    const result = await queryPipeline(
      'test',
      { topK: 1, hybrid: { enabled: false } },
      { provider, store, keywordSearcher },
    );

    expect(keywordSearcher.keywordSearch).not.toHaveBeenCalled();
    expect(result.results).toEqual(vectorResults);
  });

  it('skips keyword search when keywordSearcher is not provided', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'vector', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
    });

    expect(result.results).toEqual(vectorResults);
  });

  it('wraps keyword search errors in StoreError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([]);

    const keywordSearcher = createMockKeywordSearcher([], { fail: true });

    await expect(
      queryPipeline('test', { topK: 1 }, { provider, store, keywordSearcher }),
    ).rejects.toThrow(StoreError);
  });

  it('preserves all fields from vector result on duplicate id', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      {
        id: 'dup',
        score: 0.9,
        content: 'vector content',
        metadata: { source: 'vector' },
        documentId: 'doc-1',
        chunkIndex: 0,
        namespace: 'ns-v',
      },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordResults: SearchResult[] = [
      {
        id: 'dup',
        score: 0.8,
        content: 'keyword content',
        metadata: { source: 'keyword' },
        documentId: 'doc-2',
        chunkIndex: 1,
        namespace: 'ns-k',
      },
    ];
    const keywordSearcher = createMockKeywordSearcher(keywordResults);

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
      keywordSearcher,
    });

    expect(result.results.length).toBe(1);
    const r = result.results[0];
    expect(r.id).toBe('dup');
    expect(r.content).toBe('vector content');
    expect(r.metadata).toEqual({ source: 'vector' });
    expect(r.documentId).toBe('doc-1');
    expect(r.chunkIndex).toBe(0);
    expect(r.namespace).toBe('ns-v');
  });

  it('uses keyword result fields when id is unique to keyword list', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: 'v-only', score: 0.9, content: 'vector', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordResults: SearchResult[] = [
      {
        id: 'k-only',
        score: 0.8,
        content: 'keyword',
        metadata: { source: 'kw' },
        documentId: 'doc-k',
        chunkIndex: 5,
        namespace: 'ns-k',
      },
    ];
    const keywordSearcher = createMockKeywordSearcher(keywordResults);

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
      keywordSearcher,
    });

    const kOnly = result.results.find((r) => r.id === 'k-only');
    expect(kOnly).toBeDefined();
    expect(kOnly!.content).toBe('keyword');
    expect(kOnly!.metadata).toEqual({ source: 'kw' });
    expect(kOnly!.documentId).toBe('doc-k');
    expect(kOnly!.chunkIndex).toBe(5);
    expect(kOnly!.namespace).toBe('ns-k');
  });

  it('passes namespace and filter to keyword search', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([]);

    const keywordSearcher = createMockKeywordSearcher([]);

    await queryPipeline(
      'test',
      { topK: 5, filter: { source: 'web' }, namespace: 'ns-1' },
      { provider, store, keywordSearcher, defaultNamespace: 'default-ns' },
    );

    expect(keywordSearcher.keywordSearch).toHaveBeenCalledWith(
      'test',
      expect.objectContaining({
        topK: 5,
        filter: { source: 'web' },
        namespace: 'ns-1',
      }),
    );
  });
});

describe('rrfFusion unit', () => {
  it('ranks higher when result appears in both lists', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const vectorResults: SearchResult[] = [
      { id: 'only-vector', score: 0.9, content: 'v', metadata: {} },
      { id: 'both', score: 0.8, content: 'v', metadata: {} },
    ];
    store.query.mockResolvedValue(vectorResults);

    const keywordResults: SearchResult[] = [
      { id: 'both', score: 0.95, content: 'k', metadata: {} },
      { id: 'only-keyword', score: 0.85, content: 'k', metadata: {} },
    ];
    const keywordSearcher = createMockKeywordSearcher(keywordResults);

    const result = await queryPipeline('test', { topK: 2 }, {
      provider,
      store,
      keywordSearcher,
    });

    // both should be first because it has contributions from both lists
    expect(result.results[0].id).toBe('both');
    // Score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    expect(result.results[0].score).toBeCloseTo(1 / 62 + 1 / 61, 5);
  });

  it('deduplicates by id and preserves vector content', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([
      { id: '1', score: 0.9, content: 'from vector', metadata: {} },
    ]);

    const keywordSearcher = createMockKeywordSearcher([
      { id: '1', score: 0.8, content: 'from keyword', metadata: {} },
    ]);

    const result = await queryPipeline('test', { topK: 1 }, {
      provider,
      store,
      keywordSearcher,
    });

    expect(result.results.length).toBe(1);
    expect(result.results[0].content).toBe('from vector');
  });
});
