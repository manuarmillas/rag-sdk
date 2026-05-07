import type { Metadata, MetadataFilter, Vector, SearchResult } from './document.js';

export interface VectorStore<M extends Metadata = Metadata> {
  readonly id?: string;
  readonly dimensions?: number;
  upsert(vectors: Vector<M>[], options?: { namespace?: string }): Promise<void>;
  query(embedding: number[], options: QueryOptions): Promise<SearchResult<M>[]>;
  delete(ids: string[], options?: { namespace?: string }): Promise<void>;
}

export interface QueryOptions {
  topK?: number;
  filter?: MetadataFilter;
  namespace?: string;
  includeMetadata?: boolean;
}
