import { describe, it, expect, vi } from 'vitest';
import { ingestPipeline } from '../pipeline/ingest.js';
import {
  ValidationError,
  ProviderError,
  DimensionMismatchError,
  BatchError,
  StoreError,
  ChunkingError,
  ConfigurationError,
} from '../errors.js';
import type { Metadata, Chunk, Vector } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { QueryOptions } from '../types/store.js';
import type { Chunker } from '../chunker/types.js';

function createMockProvider(
  opts?: {
    dimensions?: number;
    maxBatchSize?: number;
    failEmbedBatch?: boolean;
  },
): EmbeddingProvider {
  const dims = opts?.dimensions ?? 3;
  return {
    id: 'mock',
    modelId: 'mock-model',
    dimensions: dims,
    maxBatchSize: opts?.maxBatchSize,
    async embed(text: string): Promise<number[]> {
      return Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      if (opts?.failEmbedBatch) throw new Error('batch failed');
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
    upsert: vi.fn(
      async (_vectors: Vector<M>[], _options?: { namespace?: string }) => {},
    ),
    query: vi.fn(
      async (_embedding: number[], _options: QueryOptions) =>
        [] as import('../types/document.js').SearchResult<M>[],
    ),
    delete: vi.fn(async (_ids: string[], _options?: { namespace?: string }) => {}),
  };
}

function createMockChunker<M extends Metadata = Metadata>(
  chunksPerDoc = 1,
): Chunker<M> {
  return {
    chunk(documents) {
      const chunks: Chunk<M>[] = [];
      for (let d = 0; d < documents.length; d++) {
        const doc = documents[d];
        for (let i = 0; i < chunksPerDoc; i++) {
          chunks.push({
            id: `chunk-${doc.id ?? d}-${i}`,
            content: doc.content,
            documentId: doc.id ?? `doc-${d}`,
            chunkIndex: i,
            metadata: (doc.metadata ?? {}) as M,
            startChar: 0,
            endChar: doc.content.length,
          });
        }
      }
      return chunks;
    },
  };
}

describe('ingestPipeline', () => {
  it('chunks documents correctly and upserts vectors matching chunks', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(2);

    await ingestPipeline(
      [
        { id: 'doc-1', content: 'hello' },
        { id: 'doc-2', content: 'world' },
      ],
      {},
      { provider, store, chunker },
    );

    expect(store.upsert).toHaveBeenCalledTimes(1);
    const vectors = store.upsert.mock.calls[0][0];
    expect(vectors).toHaveLength(4);
    expect(vectors[0].documentId).toBe('doc-1');
    expect(vectors[0].chunkIndex).toBe(0);
    expect(vectors[1].chunkIndex).toBe(1);
    expect(vectors[2].documentId).toBe('doc-2');
    expect(vectors[2].chunkIndex).toBe(0);
  });

  it('splits batches by maxBatchSize and calls embedBatch for each batch', async () => {
    const provider = createMockProvider({ dimensions: 3, maxBatchSize: 2 });
    const embedBatchSpy = vi.fn(provider.embedBatch.bind(provider));
    const providerWithSpy = { ...provider, embedBatch: embedBatchSpy };
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);

    await ingestPipeline(
      [
        { content: 'a' },
        { content: 'b' },
        { content: 'c' },
        { content: 'd' },
        { content: 'e' },
      ],
      {},
      { provider: providerWithSpy, store, chunker },
    );

    expect(embedBatchSpy).toHaveBeenCalledTimes(3);
    expect(embedBatchSpy.mock.calls[0][0]).toHaveLength(2);
    expect(embedBatchSpy.mock.calls[1][0]).toHaveLength(2);
    expect(embedBatchSpy.mock.calls[2][0]).toHaveLength(1);
  });

  it('calls store.upsert with correct vectors and namespace from options', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);

    await ingestPipeline(
      [{ id: 'doc-1', content: 'hello', metadata: { source: 'test' } }],
      { namespace: 'ns-1' },
      { provider, store, chunker },
    );

    expect(store.upsert).toHaveBeenCalledTimes(1);
    const vectors = store.upsert.mock.calls[0][0];
    const options = store.upsert.mock.calls[0][1];
    expect(vectors[0].id).toBe('chunk-doc-1-0');
    expect(vectors[0].content).toBe('hello');
    expect(vectors[0].metadata).toEqual({ source: 'test' });
    expect(vectors[0].values).toHaveLength(3);
    expect(vectors[0].namespace).toBe('ns-1');
    expect(options).toEqual({ namespace: 'ns-1' });
  });

  it('uses defaultNamespace from deps when options.namespace is not provided', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);

    await ingestPipeline(
      [{ content: 'hello' }],
      {},
      { provider, store, chunker, defaultNamespace: 'default-ns' },
    );

    expect(store.upsert.mock.calls[0][1]).toEqual({ namespace: 'default-ns' });
    expect(store.upsert.mock.calls[0][0][0].namespace).toBe('default-ns');
  });

  it('throws DimensionMismatchError when embedding dimensions differ from provider', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const originalEmbedBatch = provider.embedBatch;
    provider.embedBatch = async (texts: string[]) => {
      const result = await originalEmbedBatch(texts);
      return result.map((r) => r.slice(0, 2));
    };
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);

    await expect(
      ingestPipeline([{ content: 'hello' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws DimensionMismatchError when embedding dimensions differ from store', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 4 });
    const chunker = createMockChunker(1);

    await expect(
      ingestPipeline([{ content: 'hello' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws ValidationError for empty documents array', async () => {
    const provider = createMockProvider();
    const store = createMockStore();
    const chunker = createMockChunker();

    await expect(
      ingestPipeline([], {}, { provider, store, chunker }),
    ).rejects.toThrow(ValidationError);
  });

  it('throws ValidationError for blank document content', async () => {
    const provider = createMockProvider();
    const store = createMockStore();
    const chunker = createMockChunker();

    await expect(
      ingestPipeline([{ content: '   ' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(ValidationError);
  });

  it('wraps provider errors in ProviderError', async () => {
    const provider = createMockProvider({ failEmbedBatch: true });
    const store = createMockStore();
    const chunker = createMockChunker();

    await expect(
      ingestPipeline([{ content: 'hello' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(ProviderError);
  });

  it('throws BatchError when embedBatch returns wrong count', async () => {
    const provider = createMockProvider();
    provider.embedBatch = async () => [];
    const store = createMockStore();
    const chunker = createMockChunker(1);

    await expect(
      ingestPipeline([{ content: 'hello' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(BatchError);
  });

  it('wraps store errors in StoreError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.upsert = vi.fn((_vectors: Vector<Metadata>[], _options?: { namespace?: string }): Promise<void> => {
      throw new Error('store down');
    });
    const chunker = createMockChunker(1);

    await expect(
      ingestPipeline([{ content: 'hello' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(StoreError);
  });

  it('wraps chunker errors in ChunkingError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker: Chunker<Metadata> = {
      chunk() {
        throw new Error('chunker broke');
      },
    };

    await expect(
      ingestPipeline([{ id: 'doc-1', content: 'hello' }], {}, { provider, store, chunker }),
    ).rejects.toThrow(ChunkingError);
  });

  it('throws ConfigurationError for invalid chunk options at runtime', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);

    await expect(
      ingestPipeline(
        [{ content: 'hello' }],
        { chunk: { chunkSize: 0 } },
        { provider, store, chunker },
      ),
    ).rejects.toThrow(ConfigurationError);
  });

  it('re-ingest same doc yields same chunk IDs', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);

    await ingestPipeline(
      [{ id: 'doc-1', content: 'hello world' }],
      {},
      { provider, store, chunker },
    );
    const firstCall = store.upsert.mock.calls[0][0];

    await ingestPipeline(
      [{ id: 'doc-1', content: 'hello world' }],
      {},
      { provider, store, chunker },
    );
    const secondCall = store.upsert.mock.calls[1][0];

    expect(firstCall.map((v: Vector<Metadata>) => v.id)).toEqual(
      secondCall.map((v: Vector<Metadata>) => v.id),
    );
  });

  it('mutating original metadata after ingest does not affect stored data', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const chunker = createMockChunker(1);
    const meta = { source: 'original' };

    await ingestPipeline(
      [{ id: 'doc-1', content: 'hello', metadata: meta }],
      {},
      { provider, store, chunker },
    );

    meta.source = 'mutated';
    const vectors = store.upsert.mock.calls[0][0];
    expect(vectors[0].metadata.source).toBe('original');
  });
});
