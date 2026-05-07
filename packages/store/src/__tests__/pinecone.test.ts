import { describe, it, expect, vi } from 'vitest';
import { createPineconeStore } from '../pinecone.js';
import { ConfigurationError, DimensionMismatchError, StoreError } from '@rag-sdk/core';
import type { Vector } from '@rag-sdk/core';

const mockUpsert = vi.fn();
const mockQuery = vi.fn();
const mockDeleteMany = vi.fn();
const mockNamespace = vi.fn().mockReturnValue({
  upsert: mockUpsert,
  query: mockQuery,
  deleteMany: mockDeleteMany,
});

vi.mock('@pinecone-database/pinecone', () => ({
  Pinecone: vi.fn().mockImplementation(() => ({
    index: vi.fn().mockReturnValue({
      upsert: mockUpsert,
      query: mockQuery,
      deleteMany: mockDeleteMany,
      namespace: mockNamespace,
    }),
  })),
}));

describe('createPineconeStore', () => {
  it('creates a valid VectorStore', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
      dimensions: 3,
    });
    expect(store).toBeDefined();
    expect(store.dimensions).toBe(3);
    expect(typeof store.upsert).toBe('function');
    expect(typeof store.query).toBe('function');
    expect(typeof store.delete).toBe('function');
  });

  it('upsert delegates to Pinecone index', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
      dimensions: 3,
    });
    const vectors: Vector[] = [
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
    ];
    await store.upsert(vectors, { namespace: 'ns' });
    expect(mockNamespace).toHaveBeenCalledWith('ns');
    expect(mockUpsert).toHaveBeenCalledWith([
      {
        id: 'v1',
        values: [1, 0, 0],
        metadata: {
          content: 'a',
          documentId: 'd1',
          chunkIndex: 0,
        },
      },
    ]);
  });

  it('query delegates to Pinecone index and maps results', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
      dimensions: 3,
    });
    mockQuery.mockResolvedValueOnce({
      matches: [
        {
          id: 'v1',
          score: 0.92,
          metadata: {
            content: 'hello',
            tag: 'red',
            documentId: 'd1',
            chunkIndex: 0,
          },
        },
      ],
    });
    const results = await store.query([1, 0, 0], { topK: 5, namespace: 'ns' });
    expect(mockNamespace).toHaveBeenCalledWith('ns');
    expect(mockQuery).toHaveBeenCalledWith({
      vector: [1, 0, 0],
      topK: 5,
      filter: undefined,
      includeMetadata: true,
    });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('v1');
    expect(results[0].score).toBe(0.92);
    expect(results[0].content).toBe('hello');
    expect(results[0].metadata).toEqual({ tag: 'red' });
    expect(results[0].documentId).toBe('d1');
    expect(results[0].chunkIndex).toBe(0);
  });

  it('delete delegates to Pinecone index', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
    });
    await store.delete(['v1', 'v2'], { namespace: 'ns' });
    expect(mockNamespace).toHaveBeenCalledWith('ns');
    expect(mockDeleteMany).toHaveBeenCalledWith(['v1', 'v2']);
  });

  it('throws DimensionMismatchError on upsert with wrong dimensions', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
      dimensions: 3,
    });
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 2], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws DimensionMismatchError on query with wrong dimensions', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
      dimensions: 3,
    });
    await expect(store.query([1, 0], { topK: 5 })).rejects.toThrow(
      DimensionMismatchError,
    );
  });

  it('wraps Pinecone errors in StoreError', async () => {
    const store = await createPineconeStore({
      apiKey: 'test-key',
      indexName: 'test-index',
    });
    mockUpsert.mockRejectedValueOnce(new Error('network failure'));
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(StoreError);
  });
});

describe('requirePeer for pinecone', () => {
  it('throws ConfigurationError for missing package', async () => {
    const { requirePeer } = await import('../runtime.js');
    await expect(
      requirePeer('this-package-does-not-exist-12345', 'Missing package'),
    ).rejects.toThrow(ConfigurationError);
  });
});
