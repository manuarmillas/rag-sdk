import type { Metadata, Document, QueryResult } from './document.js';
import type { EmbeddingProvider } from './provider.js';
import type { VectorStore, QueryOptions } from './store.js';
import type { Chunker } from '../chunker/types.js';

export interface RagConfig<
  M extends Metadata = Metadata,
  P extends EmbeddingProvider = EmbeddingProvider,
  S extends VectorStore<M> = VectorStore<M>,
> {
  provider: P;
  store: S;
  chunker?: Chunker<M>;
  namespace?: string;
  chunk?: ChunkOptions;
}

export interface RagSDK<M extends Metadata = Metadata> {
  ingest(documents: Document<M>[], options?: IngestOptions): Promise<void>;
  query(text: string, options?: QueryOptions): Promise<QueryResult<M>>;
}

export interface ChunkOptions {
  chunkSize?: number;
  overlap?: number;
  separators?: string[];
}

export interface IngestOptions {
  namespace?: string;
  chunk?: ChunkOptions;
}
