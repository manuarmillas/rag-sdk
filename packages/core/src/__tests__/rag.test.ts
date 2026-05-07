import { describe, it, expect, vi } from 'vitest';
import { rag } from '../rag.js';
import {
  ConfigurationError,
  ValidationError,
  DimensionMismatchError,
  BatchError,
  ProviderError,
} from '../errors.js';
import type { Metadata, Vector, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { VectorStore, QueryOptions } from '../types/store.js';

function createMockProvider(
  opts?: {
    dimensions?: number;
    maxBatchSize?: number;
    failEmbed?: boolean;
  },
): EmbeddingProvider {
  const dimensions = opts?.dimensions ?? 3;
  return {
    id: 'mock',
    modelId: 'mock-model',
    dimensions,
    maxBatchSize: opts?.maxBatchSize,
    async embed(text: string): Promise<number[]> {
      if (opts?.failEmbed) throw new Error('embed failed');
      return Array.from(
        { length: dimensions },
        (_, i) => text.charCodeAt(0) + i,
      );
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      if (opts?.failEmbed) throw new Error('embedBatch failed');
      return texts.map((text) =>
        Array.from({ length: dimensions }, (_, i) => text.charCodeAt(0) + i),
      );
    },
  };
}

function createMockStore<M extends Metadata = Metadata>(
  opts?: { dimensions?: number },
): VectorStore<M> {
  const vectors = new Map<string, Vector<M>>();
  return {
    dimensions: opts?.dimensions,
    async upsert(
      items: Vector<M>[],
      options?: { namespace?: string },
    ): Promise<void> {
      const ns = options?.namespace ?? 'default';
      for (const v of items) {
        vectors.set(`${ns}:${v.id}`, v);
      }
    },
    async query(
      embedding: number[],
      options: QueryOptions,
    ): Promise<SearchResult<M>[]> {
      const ns = options.namespace ?? 'default';
      const results: SearchResult<M>[] = [];
      for (const [key, v] of vectors) {
        if (!key.startsWith(`${ns}:`)) continue;
        const score = embedding.reduce(
          (sum, val, i) => sum + val * v.values[i],
          0,
        );
        results.push({
          id: v.id,
          score,
          content: v.content,
          metadata: v.metadata,
          documentId: v.documentId,
          chunkIndex: v.chunkIndex,
          namespace: ns,
        });
      }
      results.sort((a, b) => b.score - a.score);
      return results.slice(0, options.topK ?? 5);
    },
    async delete(
      ids: string[],
      options?: { namespace?: string },
    ): Promise<void> {
      const ns = options?.namespace ?? 'default';
      for (const id of ids) {
        vectors.delete(`${ns}:${id}`);
      }
    },
  };
}

describe('rag() factory', () => {
  it('creates SDK with valid config', () => {
    const sdk = rag({
      provider: createMockProvider(),
      store: createMockStore(),
    });
    expect(sdk.ingest).toBeInstanceOf(Function);
    expect(sdk.query).toBeInstanceOf(Function);
  });

  it('throws ConfigurationError when provider is missing', () => {
    expect(() =>
      rag({
        provider: undefined as unknown as EmbeddingProvider,
        store: createMockStore(),
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError when store is missing', () => {
    expect(() =>
      rag({
        provider: createMockProvider(),
        store: undefined as unknown as VectorStore,
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError for invalid dimensions', () => {
    expect(() =>
      rag({
        provider: { ...createMockProvider(), dimensions: 0 },
        store: createMockStore(),
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError when overlap >= chunkSize', () => {
    expect(() =>
      rag({
        provider: createMockProvider(),
        store: createMockStore(),
        chunk: { chunkSize: 100, overlap: 100 },
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError for chunkSize <= 0', () => {
    expect(() =>
      rag({
        provider: createMockProvider(),
        store: createMockStore(),
        chunk: { chunkSize: 0 },
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError for negative overlap', () => {
    expect(() =>
      rag({
        provider: createMockProvider(),
        store: createMockStore(),
        chunk: { overlap: -1 },
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError for empty separators', () => {
    expect(() =>
      rag({
        provider: createMockProvider(),
        store: createMockStore(),
        chunk: { separators: [] },
      }),
    ).toThrow(ConfigurationError);
  });

  it('throws ConfigurationError for invalid maxBatchSize', () => {
    expect(() =>
      rag({
        provider: { ...createMockProvider(), maxBatchSize: 0 },
        store: createMockStore(),
      }),
    ).toThrow(ConfigurationError);
  });
});

describe('rag integration', () => {
  it('ingests documents and queries them', async () => {
    const sdk = rag({
      provider: createMockProvider({ dimensions: 3 }),
      store: createMockStore({ dimensions: 3 }),
    });

    await sdk.ingest([
      { id: 'doc-1', content: 'hello world', metadata: { source: 'test' } },
    ]);

    const result = await sdk.query('hello', { topK: 1 });
    expect(result.query).toBe('hello');
    expect(result.results.length).toBeGreaterThan(0);
    expect(result.results[0].content).toContain('hello');
  });

  it('uses namespace override on ingest and query', async () => {
    const sdk = rag({
      provider: createMockProvider({ dimensions: 3 }),
      store: createMockStore({ dimensions: 3 }),
      namespace: 'default-ns',
    });

    await sdk.ingest([{ content: 'alpha' }], { namespace: 'custom-ns' });
    const result = await sdk.query('alpha', {
      namespace: 'custom-ns',
      topK: 5,
    });
    expect(result.namespace).toBe('custom-ns');
    expect(result.results.length).toBeGreaterThan(0);
  });

  it('throws ValidationError on empty query', async () => {
    const sdk = rag({
      provider: createMockProvider(),
      store: createMockStore(),
    });
    await expect(sdk.query('')).rejects.toThrow(ValidationError);
  });

  it('throws ValidationError on empty documents', async () => {
    const sdk = rag({
      provider: createMockProvider(),
      store: createMockStore(),
    });
    await expect(sdk.ingest([])).rejects.toThrow(ValidationError);
  });

  it('throws ValidationError on blank document content', async () => {
    const sdk = rag({
      provider: createMockProvider(),
      store: createMockStore(),
    });
    await expect(sdk.ingest([{ content: '   ' }])).rejects.toThrow(
      ValidationError,
    );
  });

  it('splits batches according to maxBatchSize', async () => {
    const provider = createMockProvider({ dimensions: 3, maxBatchSize: 2 });
    const embedBatchSpy = vi.fn(provider.embedBatch.bind(provider));
    const providerWithSpy = { ...provider, embedBatch: embedBatchSpy };

    const sdk = rag({
      provider: providerWithSpy,
      store: createMockStore(),
    });

    await sdk.ingest([
      { content: 'a'.repeat(500) },
    ]);

    expect(embedBatchSpy).toHaveBeenCalled();
    for (const call of embedBatchSpy.mock.calls) {
      expect(call[0].length).toBeLessThanOrEqual(2);
    }
  });

  it('throws DimensionMismatchError when embedding dimensions differ from provider', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const originalEmbedBatch = provider.embedBatch;
    provider.embedBatch = async (texts: string[]): Promise<number[][]> => {
      const result = await originalEmbedBatch(texts);
      return result.map((r: number[]) => r.slice(0, 2));
    };

    const sdk = rag({
      provider,
      store: createMockStore({ dimensions: 3 }),
    });

    await expect(sdk.ingest([{ content: 'hello' }])).rejects.toThrow(
      DimensionMismatchError,
    );
  });

  it('throws ProviderError when provider fails', async () => {
    const provider = createMockProvider({ failEmbed: true });
    const sdk = rag({
      provider,
      store: createMockStore(),
    });

    await expect(sdk.query('test')).rejects.toThrow(ProviderError);
  });

  it('throws BatchError when embedBatch returns wrong count', async () => {
    const provider = createMockProvider();
    provider.embedBatch = async () => [];

    const sdk = rag({
      provider,
      store: createMockStore(),
    });

    await expect(sdk.ingest([{ content: 'hello world' }])).rejects.toThrow(
      BatchError,
    );
  });
});
