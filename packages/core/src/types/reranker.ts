import type { Metadata, SearchResult } from './document.js';

export interface Reranker<M extends Metadata = Metadata> {
  readonly id: string;
  readonly modelId?: string;

  /**
   * Reorders search results by relevance to the query.
   *
   * @remarks The `score` field in returned results is replaced with the
   * reranker's relevance score, not the original cosine similarity from the
   * vector store.
   */
  rerank(
    query: string,
    results: SearchResult<M>[],
    options?: RerankOptions,
  ): Promise<SearchResult<M>[]>;
}

export interface RerankOptions {
  topN?: number;
}
