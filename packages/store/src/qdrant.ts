import type {
  Metadata,
  Vector,
  SearchResult,
  VectorStore,
  QueryOptions,
} from '@rag-sdk/core';
import { DimensionMismatchError, StoreError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface QdrantConfig {
  url: string;
  apiKey?: string;
  collectionName: string;
  dimensions?: number;
  id?: string;
}

interface QdrantHit {
  id: string | number;
  score?: number;
  payload?: {
    content?: unknown;
    metadata?: unknown;
    documentId?: unknown;
    chunkIndex?: unknown;
  };
}

function buildQdrantFilter(filter?: Record<string, unknown>): unknown {
  if (!filter) return undefined;
  const conditions = Object.entries(filter).map(([key, value]) => ({
    key,
    match: { value },
  }));
  return { must: conditions };
}

export async function createQdrantStore<
  M extends Metadata = Metadata,
>(config: QdrantConfig): Promise<VectorStore<M>> {
  await requirePeer(
    '@qdrant/js-client-rest',
    'Qdrant adapter requires @qdrant/js-client-rest',
  );

  const { QdrantClient } = await import('@qdrant/js-client-rest');
  const client = new QdrantClient({
    url: config.url,
    apiKey: config.apiKey,
  });

  const collectionName = config.collectionName;
  const expectedDimensions = config.dimensions;

  return {
    id: config.id,
    dimensions: expectedDimensions,

    async upsert(
      vectors: Vector<M>[],
      options?: { namespace?: string },
    ): Promise<void> {
      if (expectedDimensions !== undefined) {
        for (const vector of vectors) {
          if (vector.values.length !== expectedDimensions) {
            throw new DimensionMismatchError(
              'DIMENSION_MISMATCH',
              `Vector dimension ${vector.values.length} does not match expected ${expectedDimensions}`,
            );
          }
        }
      }

      const points = vectors.map((vector) => ({
        id: vector.id,
        vector: vector.values,
        payload: {
          content: vector.content,
          metadata: vector.metadata,
          documentId: vector.documentId,
          chunkIndex: vector.chunkIndex,
          namespace: options?.namespace ?? vector.namespace ?? 'default',
        },
      }));

      try {
        await client.upsert(collectionName, { points });
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `Qdrant upsert failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      }
    },

    async query(
      embedding: number[],
      options: QueryOptions,
    ): Promise<SearchResult<M>[]> {
      if (
        expectedDimensions !== undefined &&
        embedding.length !== expectedDimensions
      ) {
        throw new DimensionMismatchError(
          'DIMENSION_MISMATCH',
          `Query embedding dimension ${embedding.length} does not match expected ${expectedDimensions}`,
        );
      }

      try {
        const response = await client.search(collectionName, {
          vector: embedding,
          limit: options.topK ?? 5,
          filter: buildQdrantFilter(options.filter) as Record<string, unknown>,
          with_payload: true,
        });

        const hits = Array.isArray(response) ? (response as QdrantHit[]) : [];
        return hits.map((hit) => ({
          id: String(hit.id),
          score: hit.score ?? 0,
          content: String(hit.payload?.content ?? ''),
          metadata: (hit.payload?.metadata ?? {}) as M,
          documentId: hit.payload?.documentId as string | undefined,
          chunkIndex: hit.payload?.chunkIndex as number | undefined,
          namespace: options.namespace,
        }));
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `Qdrant query failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      }
    },

    async delete(
      ids: string[],
      _options?: { namespace?: string },
    ): Promise<void> {
      try {
        await client.delete(collectionName, {
          points: ids,
        });
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `Qdrant delete failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      }
    },
  };
}
