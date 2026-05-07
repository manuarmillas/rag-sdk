import { describe, it, expect } from 'vitest';
import { createMemoryStore } from '../memory-store.js';
import type { Vector } from '@rag-sdk/core';
import { DimensionMismatchError } from '@rag-sdk/core';

describe('InMemoryVectorStore', () => {
  it('upsert stores vectors correctly', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    const vectors: Vector[] = [
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      { id: 'v2', values: [0, 1, 0], content: 'b', metadata: {}, documentId: 'd1', chunkIndex: 1 },
    ];
    await store.upsert(vectors);
    const results = await store.query([1, 0, 0], { topK: 5 });
    expect(results).toHaveLength(2);
    expect(results[0].id).toBe('v1');
    expect(results[0].score).toBeCloseTo(1);
  });

  it('returns ranked results by cosine similarity', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      { id: 'v2', values: [0, 1, 0], content: 'b', metadata: {}, documentId: 'd1', chunkIndex: 1 },
      { id: 'v3', values: [1, 1, 0], content: 'c', metadata: {}, documentId: 'd1', chunkIndex: 2 },
    ]);
    const results = await store.query([1, 0, 0], { topK: 5 });
    expect(results[0].id).toBe('v1');
    expect(results[1].id).toBe('v3');
    expect(results[2].id).toBe('v2');
  });

  it('limits results by topK', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      { id: 'v2', values: [0, 1, 0], content: 'b', metadata: {}, documentId: 'd1', chunkIndex: 1 },
    ]);
    const results = await store.query([1, 0, 0], { topK: 1 });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('v1');
  });

  it('applies metadata filter', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: { tag: 'red' }, documentId: 'd1', chunkIndex: 0 },
      { id: 'v2', values: [0, 1, 0], content: 'b', metadata: { tag: 'blue' }, documentId: 'd1', chunkIndex: 1 },
    ]);
    const results = await store.query([1, 1, 0], { topK: 5, filter: { tag: 'blue' } });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('v2');
  });

  it('deletes vectors by ID', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      { id: 'v2', values: [0, 1, 0], content: 'b', metadata: {}, documentId: 'd1', chunkIndex: 1 },
    ]);
    await store.delete(['v1']);
    const results = await store.query([1, 0, 0], { topK: 5 });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('v2');
  });

  it('isolates namespaces', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert(
      [{ id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 }],
      { namespace: 'ns-1' },
    );
    await store.upsert(
      [{ id: 'v2', values: [0, 1, 0], content: 'b', metadata: {}, documentId: 'd1', chunkIndex: 0 }],
      { namespace: 'ns-2' },
    );
    const results1 = await store.query([1, 0, 0], { topK: 5, namespace: 'ns-1' });
    expect(results1).toHaveLength(1);
    expect(results1[0].id).toBe('v1');

    const results2 = await store.query([1, 0, 0], { topK: 5, namespace: 'ns-2' });
    expect(results2).toHaveLength(1);
    expect(results2[0].id).toBe('v2');
  });

  it('returns empty array when store is empty', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    const results = await store.query([1, 0, 0], { topK: 5 });
    expect(results).toEqual([]);
  });

  it('throws DimensionMismatchError on first upsert with wrong dimensions', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 2], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws DimensionMismatchError on second upsert with wrong dimensions', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
    ]);
    await expect(
      store.upsert([
        { id: 'v2', values: [1, 2], content: 'b', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws DimensionMismatchError on query with wrong dimensions', async () => {
    const store = createMemoryStore({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
    ]);
    await expect(
      store.query([1, 0], { topK: 5 }),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('mutating returned metadata does not affect internal store', async () => {
    const store = createMemoryStore<{ source: string }>({ dimensions: 3 });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: { source: 'test' }, documentId: 'd1', chunkIndex: 0 },
    ]);
    const results = await store.query([1, 0, 0], { topK: 5 });
    results[0].metadata.source = 'mutated';

    const results2 = await store.query([1, 0, 0], { topK: 5 });
    expect(results2[0].metadata.source).toBe('test');
  });
});
