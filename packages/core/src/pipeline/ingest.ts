import type { Metadata, Document, Vector } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { VectorStore } from '../types/store.js';
import type { Chunker } from '../chunker/types.js';
import type { IngestOptions, ChunkOptions } from '../types/config.js';
import {
  ValidationError,
  ProviderError,
  DimensionMismatchError,
  BatchError,
  StoreError,
  ChunkingError,
} from '../errors.js';
import { validateChunkOptions } from '../validate.js';

export interface IngestDeps<M extends Metadata = Metadata> {
  provider: EmbeddingProvider;
  store: VectorStore<M>;
  chunker: Chunker<M>;
  defaultNamespace?: string;
  defaultChunkOpts?: ChunkOptions;
}

export async function ingestPipeline<M extends Metadata>(
  documents: Document<M>[],
  options: IngestOptions | undefined,
  deps: IngestDeps<M>,
): Promise<void> {
  if (!documents || documents.length === 0) {
    throw new ValidationError(
      'VALIDATION_ERROR',
      'No documents provided for ingestion',
    );
  }

  for (const doc of documents) {
    if (!doc.content || doc.content.trim().length === 0) {
      throw new ValidationError(
        'VALIDATION_ERROR',
        'Document content cannot be empty',
      );
    }
  }

  const namespace = options?.namespace ?? deps.defaultNamespace;
  const chunkOpts: ChunkOptions = {
    ...deps.defaultChunkOpts,
    ...options?.chunk,
  };
  validateChunkOptions(chunkOpts);

  let chunks;
  try {
    chunks = await deps.chunker.chunk(documents, chunkOpts);
  } catch (err) {
    const docId = documents[0]?.id ?? 'unknown';
    throw new ChunkingError(
      'CHUNKING_ERROR',
      `Chunking failed for document ${docId}`,
      err,
    );
  }

  if (chunks.length === 0) {
    return;
  }

  const texts = chunks.map((c) => c.content);
  const embeddings: number[][] = [];
  const maxBatchSize = deps.provider.maxBatchSize;

  if (maxBatchSize && maxBatchSize > 0) {
    for (let i = 0; i < texts.length; i += maxBatchSize) {
      const batch = texts.slice(i, i + maxBatchSize);
      const batchEmbeddings = await embedBatchSafe(deps.provider, batch);
      if (batchEmbeddings.length !== batch.length) {
        throw new BatchError(
          'BATCH_ERROR',
          `Embedding count mismatch: expected ${batch.length}, got ${batchEmbeddings.length}`,
        );
      }
      embeddings.push(...batchEmbeddings);
    }
  } else {
    const batchEmbeddings = await embedBatchSafe(deps.provider, texts);
    if (batchEmbeddings.length !== texts.length) {
      throw new BatchError(
        'BATCH_ERROR',
        `Embedding count mismatch: expected ${texts.length}, got ${batchEmbeddings.length}`,
      );
    }
    embeddings.push(...batchEmbeddings);
  }

  const expectedDimensions = deps.provider.dimensions;
  const storeDimensions = deps.store.dimensions;

  for (let i = 0; i < embeddings.length; i++) {
    const embedding = embeddings[i];
    if (embedding.length !== expectedDimensions) {
      throw new DimensionMismatchError(
        'DIMENSION_MISMATCH',
        `Embedding dimension mismatch at chunk ${i}: expected ${expectedDimensions}, got ${embedding.length}`,
      );
    }
    if (
      storeDimensions !== undefined &&
      embedding.length !== storeDimensions
    ) {
      throw new DimensionMismatchError(
        'DIMENSION_MISMATCH',
        `Embedding dimension mismatch with store: expected ${storeDimensions}, got ${embedding.length}`,
      );
    }
  }

  const vectors: Vector<M>[] = chunks.map((chunk, i) => ({
    id: chunk.id,
    values: embeddings[i],
    content: chunk.content,
    metadata: { ...(chunk.metadata ?? {}) } as M,
    documentId: chunk.documentId,
    chunkIndex: chunk.chunkIndex,
    namespace,
  }));

  try {
    await deps.store.upsert(vectors, { namespace });
  } catch (err) {
    if (err instanceof StoreError) {
      throw err;
    }
    throw new StoreError(
      'STORE_ERROR',
      err instanceof Error ? err.message : 'Store upsert failed',
      err,
    );
  }
}

async function embedBatchSafe(
  provider: EmbeddingProvider,
  texts: string[],
): Promise<number[][]> {
  try {
    return await provider.embedBatch(texts);
  } catch (err) {
    if (err instanceof ProviderError) {
      throw err;
    }
    throw new ProviderError(provider.id, 'embedBatch', err);
  }
}
