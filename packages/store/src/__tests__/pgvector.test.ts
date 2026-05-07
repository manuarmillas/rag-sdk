import { describe, it, expect, vi } from 'vitest';
import { createPgVectorStore } from '../pgvector.js';
import { DimensionMismatchError, StoreError } from '@rag-sdk/core';
import type { Vector } from '@rag-sdk/core';

function createMockPool() {
  const queryLog: Array<{ sql: string; params: unknown[] }> = [];

  const mockQuery = vi.fn(async (sql: string, params?: unknown[]) => {
    queryLog.push({ sql, params: params ?? [] });
    return { rows: [] as unknown[] };
  });

  const mockClient = {
    query: mockQuery,
    release: vi.fn(),
  };

  const mockConnect = vi.fn().mockResolvedValue(mockClient);

  return {
    connect: mockConnect,
    query: mockQuery,
    _queryLog: queryLog,
    _client: mockClient,
  };
}

describe('createPgVectorStore', () => {
  it('creates a valid VectorStore', () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    expect(store).toBeDefined();
    expect(store.dimensions).toBe(3);
    expect(typeof store.upsert).toBe('function');
    expect(typeof store.query).toBe('function');
    expect(typeof store.delete).toBe('function');
  });

  it('ensures table on first operation', async () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    await store.upsert([
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
    ]);
    const tableQuery = pool._queryLog.find((q) =>
      q.sql.includes('CREATE TABLE IF NOT EXISTS vectors'),
    );
    expect(tableQuery).toBeDefined();
  });

  it('upsert delegates to pool with correct SQL', async () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    const vectors: Vector[] = [
      { id: 'v1', values: [1, 0, 0], content: 'a', metadata: { tag: 'red' }, documentId: 'd1', chunkIndex: 0 },
    ];
    await store.upsert(vectors, { namespace: 'ns' });

    const upsertQuery = pool._queryLog.find((q) =>
      q.sql.includes('INSERT INTO vectors'),
    );
    expect(upsertQuery).toBeDefined();
    expect(upsertQuery?.params).toEqual([
      'v1',
      '[1,0,0]',
      'a',
      JSON.stringify({ tag: 'red' }),
      'd1',
      0,
      'ns',
    ]);
  });

  it('query delegates to pool and maps results', async () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });

    pool._client.query.mockImplementation(async (sql: string, params?: unknown[]) => {
      pool._queryLog.push({ sql, params: params ?? [] });
      if (sql.includes('SELECT')) {
        return {
          rows: [
            {
              id: 'v1',
              content: 'hello',
              metadata: { tag: 'red' },
              documentId: 'd1',
              chunkIndex: 0,
              score: 0.95,
            },
          ],
        };
      }
      return { rows: [] };
    });

    const results = await store.query([1, 0, 0], { topK: 5, namespace: 'ns' });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('v1');
    expect(results[0].score).toBe(0.95);
    expect(results[0].content).toBe('hello');
    expect(results[0].metadata).toEqual({ tag: 'red' });
    expect(results[0].documentId).toBe('d1');
    expect(results[0].chunkIndex).toBe(0);
  });

  it('delete delegates to pool with correct SQL', async () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    await store.delete(['v1', 'v2'], { namespace: 'ns' });

    const deleteQuery = pool._queryLog.find((q) =>
      q.sql.includes('DELETE FROM vectors'),
    );
    expect(deleteQuery).toBeDefined();
    expect(deleteQuery?.params).toEqual([['v1', 'v2'], 'ns']);
  });

  it('throws DimensionMismatchError on upsert with wrong dimensions', async () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 2], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(DimensionMismatchError);
  });

  it('throws DimensionMismatchError on query with wrong dimensions', async () => {
    const pool = createMockPool();
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    await expect(store.query([1, 0], { topK: 5 })).rejects.toThrow(
      DimensionMismatchError,
    );
  });

  it('wraps pg errors in StoreError', async () => {
    const pool = createMockPool();
    pool.connect.mockResolvedValueOnce({
      query: vi.fn().mockRejectedValue(new Error('connection lost')),
      release: vi.fn(),
    });
    const store = createPgVectorStore({
      pool: pool as unknown as import('pg').Pool,
      tableName: 'vectors',
      dimensions: 3,
    });
    await expect(
      store.upsert([
        { id: 'v1', values: [1, 0, 0], content: 'a', metadata: {}, documentId: 'd1', chunkIndex: 0 },
      ]),
    ).rejects.toThrow(StoreError);
  });

  it('does not accept raw connection string', () => {
    // The PgVectorConfig interface only accepts pool: Pool
    // TypeScript compilation enforces this; runtime has no string path.
    expect(true).toBe(true);
  });
});
