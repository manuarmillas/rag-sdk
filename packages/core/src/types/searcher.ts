import type { Metadata, SearchResult } from './document.js';
import type { QueryOptions } from './store.js';

export interface KeywordSearcher<M extends Metadata = Metadata> {
  readonly id?: string;
  keywordSearch(text: string, options: QueryOptions): Promise<SearchResult<M>[]>;
}

export interface HybridOptions {
  enabled?: boolean;
  vectorWeight?: number;
  keywordWeight?: number;
  rrfK?: number;
}
