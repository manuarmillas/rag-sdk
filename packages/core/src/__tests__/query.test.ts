import { describe, it, expect, vi } from 'vitest';
import { queryPipeline } from '../pipeline/query.js';
import {
  ValidationError,
  ProviderError,
  DimensionMismatchError,
  StoreError,
} from '../errors.js';
import type { Metadata, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { QueryOptions } from '../types/store.js';

function createMockProvider(
  opts?: {
    dimensions?: number;
    failEmbed?: boolean;
  },
): EmbeddingProvider {
  const dims = opts?.dimensions ?? 3;
  return {
    id: 'mock',
    modelId: 'mock-model',
    dimensions: dims,
    async embed(text: string): Promise<number[]> {
      if (opts?.failEmbed) throw new Error('embed failed');
      return Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map((text) =>
        Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i),
      );
    },
  };
}

function createMockStore<M extends Metadata = Metadata>(
  opts?: { dimensions?: number },
) {
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

describe('queryPipeline', () => {
  it('embeds the query text', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const embedSpy = vi.fn(provider.embed.bind(provider));
    const providerWithSpy = { ...provider, embed: embedSpy };
    const store = createMockStore({ dimensions: 3 });

    await queryPipeline('hello world', {}, { provider: providerWithSpy, store });

    expect(embedSpy).toHaveBeenCalledWith('hello world');
  });

  it('calls store.query with the embedding and passes topK, filter, namespace', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    await queryPipeline(
      'hello',
      { topK: 10, filter: { source: 'test' }, namespace: 'ns-1' },
      { provider, store },
    );

    expect(store.query).toHaveBeenCalledTimes(1);
    const [embedding, options] = store.query.mock.calls[0];
    expect(embedding).toEqual(
      Array.from({ length: 3 }, (_, i) => 'hello'.charCodeAt(0) + i),
    );
    expect(options.topK).toBe(10);
    expect(options.filter).toEqual({ source: 'test' });
    expect(options.namespace).toBe('ns-1');
  });

  it('returns QueryResult with original query and results', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'hello', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const result = await queryPipeline('hello', {}, { provider, store });

    expect(result.query).toBe('hello');
    expect(result.results).toBe(mockResults);
  });

  it('uses defaultNamespace from deps when options.namespace is not provided', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    const result = await queryPipeline(
      'hello',
      {},
      { provider, store, defaultNamespace: 'default-ns' },
    );

    expect(store.query.mock.calls[0][1].namespace).toBe('default-ns');
    expect(result.namespace).toBe('default-ns');
  });

  it('throws ValidationError for empty query text', async () => {
    const provider = createMockProvider();
    const store = createMockStore();

    await expect(queryPipeline('', {}, { provider, store })).rejects.toThrow(
      ValidationError,
    );
    await expect(
      queryPipeline('   ', {}, { provider, store }),
    ).rejects.toThrow(ValidationError);
  });

  it('throws ValidationError for invalid topK values', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    await expect(queryPipeline('hello', { topK: 0 }, { provider, store })).rejects.toThrow(ValidationError);
    await expect(queryPipeline('hello', { topK: -1 }, { provider, store })).rejects.toThrow(ValidationError);
    await expect(queryPipeline('hello', { topK: NaN }, { provider, store })).rejects.toThrow(ValidationError);
    await expect(queryPipeline('hello', { topK: Infinity }, { provider, store })).rejects.toThrow(ValidationError);
    await expect(queryPipeline('hello', { topK: 1.5 }, { provider, store })).rejects.toThrow(ValidationError);
  });

  it('accepts valid topK value', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    await expect(queryPipeline('hello', { topK: 1 }, { provider, store })).resolves.toBeDefined();
  });

  it('wraps provider errors in ProviderError', async () => {
    const provider = createMockProvider({ failEmbed: true });
    const store = createMockStore();

    await expect(queryPipeline('hello', {}, { provider, store })).rejects.toThrow(
      ProviderError,
    );
  });

  it('wraps store errors in StoreError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query = vi.fn((_embedding: number[], _options: QueryOptions): Promise<SearchResult<Metadata>[]> => {
      throw new Error('store down');
    });

    await expect(queryPipeline('hello', {}, { provider, store })).rejects.toThrow(
      StoreError,
    );
  });

  it('throws DimensionMismatchError when embedding dimensions differ from provider', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    provider.embed = async () => [1, 2];
    const store = createMockStore({ dimensions: 3 });

    await expect(queryPipeline('hello', {}, { provider, store })).rejects.toThrow(
      DimensionMismatchError,
    );
  });

  it('throws DimensionMismatchError when embedding dimensions differ from store', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 4 });

    await expect(queryPipeline('hello', {}, { provider, store })).rejects.toThrow(
      DimensionMismatchError,
    );
  });
});
