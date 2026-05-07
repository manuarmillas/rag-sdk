import type { Pool } from 'pg';
import type {
  Metadata,
  Vector,
  SearchResult,
  VectorStore,
  QueryOptions,
} from '@rag-sdk/core';
import { DimensionMismatchError, StoreError } from '@rag-sdk/core';

export interface PgVectorConfig {
  pool: Pool;
  tableName: string;
  dimensions: number;
  id?: string;
}

interface PgVectorRow {
  id: string;
  content: string;
  metadata: unknown;
  documentId: string;
  chunkIndex: number;
  score: number;
}

export function createPgVectorStore<
  M extends Metadata = Metadata,
>(config: PgVectorConfig): VectorStore<M> {
  const { pool, tableName, dimensions } = config;
  let initialized = false;

  async function ensureTable(): Promise<void> {
    if (initialized) return;
    const client = await pool.connect();
    try {
      await client.query('CREATE EXTENSION IF NOT EXISTS vector');
      await client.query(`
        CREATE TABLE IF NOT EXISTS ${tableName} (
          id TEXT PRIMARY KEY,
          embedding vector(${dimensions}),
          content TEXT NOT NULL,
          metadata JSONB DEFAULT '{}',
          document_id TEXT,
          chunk_index INT,
          namespace TEXT DEFAULT 'default'
        )
      `);
      initialized = true;
    } catch (err) {
      throw new StoreError(
        'STORE_ERROR',
        `PgVector table setup failed: ${err instanceof Error ? err.message : String(err)}`,
        err,
      );
    } finally {
      client.release();
    }
  }

  return {
    id: config.id,
    dimensions,

    async upsert(
      vectors: Vector<M>[],
      options?: { namespace?: string },
    ): Promise<void> {
      await ensureTable();
      const namespace = options?.namespace ?? 'default';

      for (const vector of vectors) {
        if (vector.values.length !== dimensions) {
          throw new DimensionMismatchError(
            'DIMENSION_MISMATCH',
            `Vector dimension ${vector.values.length} does not match expected ${dimensions}`,
          );
        }
      }

      const client = await pool.connect();
      try {
        await client.query('BEGIN');
        for (const vector of vectors) {
          await client.query(
            `
            INSERT INTO ${tableName} (id, embedding, content, metadata, document_id, chunk_index, namespace)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
              embedding = EXCLUDED.embedding,
              content = EXCLUDED.content,
              metadata = EXCLUDED.metadata,
              document_id = EXCLUDED.document_id,
              chunk_index = EXCLUDED.chunk_index,
              namespace = EXCLUDED.namespace
            `,
            [
              vector.id,
              `[${vector.values.join(',')}]`,
              vector.content,
              JSON.stringify(vector.metadata),
              vector.documentId,
              vector.chunkIndex,
              namespace,
            ],
          );
        }
        await client.query('COMMIT');
      } catch (err) {
        await client.query('ROLLBACK');
        throw new StoreError(
          'STORE_ERROR',
          `PgVector upsert failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      } finally {
        client.release();
      }
    },

    async query(
      embedding: number[],
      options: QueryOptions,
    ): Promise<SearchResult<M>[]> {
      await ensureTable();
      const namespace = options.namespace ?? 'default';

      if (embedding.length !== dimensions) {
        throw new DimensionMismatchError(
          'DIMENSION_MISMATCH',
          `Query embedding dimension ${embedding.length} does not match expected ${dimensions}`,
        );
      }

      const client = await pool.connect();
      try {
        const filterJson = options.filter
          ? JSON.stringify(options.filter)
          : null;
        const result = await client.query(
          `
          SELECT
            id,
            content,
            metadata,
            document_id as "documentId",
            chunk_index as "chunkIndex",
            1 - (embedding <=> $1) as score
          FROM ${tableName}
          WHERE namespace = $2
            AND ($3::jsonb IS NULL OR metadata @> $3::jsonb)
          ORDER BY embedding <=> $1
          LIMIT $4
          `,
          [
            `[${embedding.join(',')}]`,
            namespace,
            filterJson,
            options.topK ?? 5,
          ],
        );

        return (result.rows as PgVectorRow[]).map((row) => ({
          id: row.id,
          score: Number(row.score),
          content: row.content,
          metadata: (row.metadata ?? {}) as M,
          documentId: row.documentId,
          chunkIndex: row.chunkIndex,
          namespace,
        }));
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `PgVector query failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      } finally {
        client.release();
      }
    },

    async delete(
      ids: string[],
      options?: { namespace?: string },
    ): Promise<void> {
      await ensureTable();
      const namespace = options?.namespace ?? 'default';

      const client = await pool.connect();
      try {
        await client.query(
          `DELETE FROM ${tableName} WHERE id = ANY($1) AND namespace = $2`,
          [ids, namespace],
        );
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `PgVector delete failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      } finally {
        client.release();
      }
    },
  };
}
