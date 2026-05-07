import { describe, it, expect, vi } from 'vitest';
import { createQdrantStore } from '../qdrant.js';
import { ConfigurationError, DimensionMismatchError, StoreError } from '@rag-sdk/core';
import type { Vector } from '@rag-sdk/core';

const mockUpsert = vi.fn();
const mockSearch = vi.fn();
const mockDelete = vi.fn();

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    upsert: mockUpsert,
    search: mockSearch,
    delete: mockDelete,
  })),
}));

describe('createQdrantStore', () => {
  it('creates a valid VectorStore', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
      dimensions: 3,
    });
    expect(store).toBeDefined();
    expect(store.dimensions).toBe(3);
    expect(typeof store.upsert).toBe('function');
    expect(typeof store.query).toBe('function');
    expect(typeof store.delete).toBe('function');
  });

  it('upsert delegates to Qdrant client', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
      dimensions: 3,
    });
    const vectors: Vector[] = [
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
    ];
    await store.upsert(vectors, { namespace: 'ns' });
    expect(mockUpsert).toHaveBeenCalledWith('test', {
      points: [
        {
          id: 'v1',
          vector: [1, 0, 0],
          payload: {
            content: 'a',
            metadata: {},
            documentId: 'd1',
            chunkIndex: 0,
            namespace: 'ns',
          },
        },
      ],
    });
  });

  it('query delegates to Qdrant client and maps results', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
      dimensions: 3,
    });
    mockSearch.mockResolvedValueOnce([
      {
        id: 'v1',
        score: 0.95,
        payload: {
          content: 'hello',
          metadata: { tag: 'red' },
          documentId: 'd1',
          chunkIndex: 0,
        },
      },
    ]);
    const results = await store.query([1, 0, 0], { topK: 5, namespace: 'ns' });
    expect(mockSearch).toHaveBeenCalledWith('test', {
      vector: [1, 0, 0],
      limit: 5,
      filter: undefined,
      with_payload: true,
    });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('v1');
    expect(results[0].score).toBe(0.95);
    expect(results[0].content).toBe('hello');
    expect(results[0].metadata).toEqual({ tag: 'red' });
    expect(results[0].documentId).toBe('d1');
    expect(results[0].chunkIndex).toBe(0);
  });

  it('delete delegates to Qdrant client', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
    });
    await store.delete(['v1', 'v2']);
    expect(mockDelete).toHaveBeenCalledWith('test', {
      points: ['v1', 'v2'],
    });
  });

  it('throws DimensionMismatchError on upsert with wrong dimensions', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
      dimensions: 3,
    });
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 2], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws DimensionMismatchError on query with wrong dimensions', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
      dimensions: 3,
    });
    await expect(store.query([1, 0], { topK: 5 })).rejects.toThrow(
      DimensionMismatchError,
    );
  });

  it('wraps Qdrant errors in StoreError', async () => {
    const store = await createQdrantStore({
      url: 'http://localhost:6333',
      collectionName: 'test',
    });
    mockUpsert.mockRejectedValueOnce(new Error('network failure'));
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(StoreError);
  });
});

describe('requirePeer', () => {
  it('throws ConfigurationError when @qdrant/js-client-rest is missing', async () => {
    // Simulate missing module by mocking import to fail
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClient).mockImplementationOnce(() => {
      throw new Error('Module not found');
    });

    // Actually, we can't easily mock the module import itself after it's loaded.
    // Instead, we test the runtime helper directly.
    const { requirePeer } = await import('../runtime.js');

    // Test the error path by calling requirePeer with a nonexistent package
    await expect(
      requirePeer('this-package-does-not-exist-12345', 'Missing package'),
    ).rejects.toThrow(ConfigurationError);
  });
});
