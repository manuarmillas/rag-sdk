import type {
  Metadata,
  Vector,
  SearchResult,
  VectorStore,
  QueryOptions,
} from '@rag-sdk/core';
import { DimensionMismatchError, StoreError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface PineconeConfig {
  apiKey: string;
  indexName: string;
  dimensions?: number;
  id?: string;
}

interface PineconeMatch {
  id: string;
  score?: number;
  metadata?: Record<string, unknown>;
}

interface PineconeQueryResponse {
  matches?: PineconeMatch[];
}

export async function createPineconeStore<
  M extends Metadata = Metadata,
>(config: PineconeConfig): Promise<VectorStore<M>> {
  await requirePeer(
    '@pinecone-database/pinecone',
    'Pinecone adapter requires @pinecone-database/pinecone',
  );

  const { Pinecone } = await import('@pinecone-database/pinecone');
  const pc = new Pinecone({ apiKey: config.apiKey });
  const index = pc.index(config.indexName);
  const expectedDimensions = config.dimensions;

  function getNamespace(ns?: string) {
    return ns ? index.namespace(ns) : index;
  }

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

      const records = vectors.map((vector) => ({
        id: vector.id,
        values: vector.values,
        metadata: {
          content: vector.content,
          ...vector.metadata,
          documentId: vector.documentId,
          chunkIndex: vector.chunkIndex,
        },
      }));

      try {
        await getNamespace(options?.namespace).upsert(records as never[]);
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `Pinecone upsert failed: ${err instanceof Error ? err.message : String(err)}`,
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
        const response = await getNamespace(options.namespace).query({
          vector: embedding,
          topK: options.topK ?? 5,
          filter: options.filter,
          includeMetadata: true,
        });

        const matches = (response as PineconeQueryResponse).matches ?? [];
        return matches.map((match) => ({
          id: match.id,
          score: match.score ?? 0,
          content: String(match.metadata?.content ?? ''),
          metadata: Object.fromEntries(
            Object.entries(match.metadata ?? {}).filter(
              ([key]) =>
                key !== 'content' &&
                key !== 'documentId' &&
                key !== 'chunkIndex',
            ),
          ) as M,
          documentId: match.metadata?.documentId as string | undefined,
          chunkIndex: match.metadata?.chunkIndex as number | undefined,
          namespace: options.namespace,
        }));
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `Pinecone query failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      }
    },

    async delete(
      ids: string[],
      options?: { namespace?: string },
    ): Promise<void> {
      try {
        await getNamespace(options?.namespace).deleteMany(ids);
      } catch (err) {
        throw new StoreError(
          'STORE_ERROR',
          `Pinecone delete failed: ${err instanceof Error ? err.message : String(err)}`,
          err,
        );
      }
    },
  };
}
