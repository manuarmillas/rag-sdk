import type {
  Metadata,
  Vector,
  SearchResult,
  VectorStore,
  QueryOptions,
} from '@rag-sdk/core';
import { DimensionMismatchError } from '@rag-sdk/core';

export interface MemoryStoreConfig {
  dimensions?: number;
  id?: string;
}

export function createMemoryStore<M extends Metadata = Metadata>(
  config?: MemoryStoreConfig,
): VectorStore<M> {
  const namespaces = new Map<string, Map<string, Vector<M>>>();
  const expectedDimensions = config?.dimensions;

  return {
    id: config?.id,
    dimensions: expectedDimensions,

    async upsert(
      vectors: Vector<M>[],
      options?: { namespace?: string },
    ): Promise<void> {
      await Promise.resolve();
      const namespace = options?.namespace ?? 'default';

      if (!namespaces.has(namespace)) {
        namespaces.set(namespace, new Map());
      }
      const store = namespaces.get(namespace)!;

      for (const vector of vectors) {
        if (
          expectedDimensions !== undefined &&
          vector.values.length !== expectedDimensions
        ) {
          throw new DimensionMismatchError(
            'DIMENSION_MISMATCH',
            `Vector dimension ${vector.values.length} does not match expected ${expectedDimensions}`,
          );
        }
        store.set(vector.id, vector);
      }
    },

    async query(
      embedding: number[],
      options: QueryOptions,
    ): Promise<SearchResult<M>[]> {
      await Promise.resolve();
      const namespace = options.namespace ?? 'default';
      const store = namespaces.get(namespace);
      if (!store) return [];

      if (
        expectedDimensions !== undefined &&
        embedding.length !== expectedDimensions
      ) {
        throw new DimensionMismatchError(
          'DIMENSION_MISMATCH',
          `Query embedding dimension ${embedding.length} does not match expected ${expectedDimensions}`,
        );
      }

      const topK = options.topK ?? 5;
      const filter = options.filter;
      const includeMetadata = options.includeMetadata ?? true;

      const results: SearchResult<M>[] = [];

      for (const vector of store.values()) {
        if (filter && !matchesFilter(vector.metadata, filter)) {
          continue;
        }

        const score = cosineSimilarity(embedding, vector.values);
        results.push({
          id: vector.id,
          score,
          content: vector.content,
          metadata: includeMetadata ? { ...vector.metadata } : ({} as M),
          documentId: vector.documentId,
          chunkIndex: vector.chunkIndex,
          namespace,
        });
      }

      results.sort((a, b) => b.score - a.score);
      return results.slice(0, topK);
    },

    async delete(
      ids: string[],
      options?: { namespace?: string },
    ): Promise<void> {
      await Promise.resolve();
      const namespace = options?.namespace ?? 'default';
      const store = namespaces.get(namespace);
      if (!store) return;

      for (const id of ids) {
        store.delete(id);
      }
    },
  };
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new DimensionMismatchError(
      'DIMENSION_MISMATCH',
      `Cosine similarity vectors have different lengths: ${a.length} vs ${b.length}`,
    );
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function matchesFilter(
  metadata: Record<string, unknown>,
  filter: Record<string, unknown>,
): boolean {
  for (const [key, value] of Object.entries(filter)) {
    if (metadata[key] !== value) return false;
  }
  return true;
}
