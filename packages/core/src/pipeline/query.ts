import type { Metadata, QueryResult, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { VectorStore, QueryOptions } from '../types/store.js';
import type { Reranker } from '../types/reranker.js';
import type { KeywordSearcher } from '../types/searcher.js';
import { rrfFusion } from './hybrid.js';
import {
  ValidationError,
  ProviderError,
  DimensionMismatchError,
  StoreError,
} from '../errors.js';

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

export interface QueryDeps<M extends Metadata = Metadata> {
  provider: EmbeddingProvider;
  store: VectorStore<M>;
  reranker?: Reranker<M>;
  keywordSearcher?: KeywordSearcher<M>;
  defaultNamespace?: string;
}

export async function queryPipeline<M extends Metadata>(
  text: string,
  options: QueryOptions | undefined,
  deps: QueryDeps<M>,
): Promise<QueryResult<M>> {
  if (!text || text.trim().length === 0) {
    throw new ValidationError(
      'VALIDATION_ERROR',
      'Query text cannot be empty',
    );
  }

  const namespace = options?.namespace ?? deps.defaultNamespace;

  const topK = options?.topK ?? 5;
  if (topK !== 5 && !isPositiveInteger(topK)) {
    throw new ValidationError(
      'VALIDATION_ERROR',
      'topK must be a positive integer',
    );
  }

  let embedding: number[];
  try {
    embedding = await deps.provider.embed(text);
  } catch (err) {
    if (err instanceof ProviderError) {
      throw err;
    }
    throw new ProviderError(deps.provider.id, 'embed', err);
  }

  if (embedding.length !== deps.provider.dimensions) {
    throw new DimensionMismatchError(
      'DIMENSION_MISMATCH',
      `Embedding dimension mismatch: expected ${deps.provider.dimensions}, got ${embedding.length}`,
    );
  }

  const storeDimensions = deps.store.dimensions;
  if (storeDimensions !== undefined && embedding.length !== storeDimensions) {
    throw new DimensionMismatchError(
      'DIMENSION_MISMATCH',
      `Embedding dimension mismatch with store: expected ${storeDimensions}, got ${embedding.length}`,
    );
  }

  let results;
  try {
    results = await deps.store.query(embedding, {
      topK,
      filter: options?.filter,
      namespace,
      includeMetadata: options?.includeMetadata,
    });
  } catch (err) {
    if (err instanceof StoreError) {
      throw err;
    }
    throw new StoreError(
      'STORE_ERROR',
      err instanceof Error ? err.message : 'Store query failed',
      err,
    );
  }

  if (
    deps.keywordSearcher &&
    options?.hybrid?.enabled !== false
  ) {
    let keywordResults: SearchResult<M>[];
    try {
      keywordResults = await deps.keywordSearcher.keywordSearch(text, {
        topK,
        filter: options?.filter,
        namespace,
        includeMetadata: options?.includeMetadata,
      });
    } catch (err) {
      throw new StoreError(
        'STORE_ERROR',
        'Keyword search failed',
        err,
      );
    }

    results = rrfFusion(results, keywordResults, options?.hybrid);
  }

  if (deps.reranker) {
    try {
      results = await deps.reranker.rerank(
        text,
        results,
        options?.rerank,
      );
    } catch (err) {
      if (err instanceof ProviderError) {
        throw err;
      }
      throw new ProviderError(deps.reranker.id, 'rerank', err);
    }

    const topN = options?.rerank?.topN;
    if (topN !== undefined && topN >= 0 && results.length > topN) {
      results = results.slice(0, topN);
    }
  }

  return {
    query: text,
    results,
    namespace,
  };
}
