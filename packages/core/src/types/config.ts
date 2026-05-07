import type { Metadata, Document, QueryResult } from './document.js';
import type { EmbeddingProvider } from './provider.js';
import type { VectorStore, QueryOptions } from './store.js';
import type { Chunker } from '../chunker/types.js';
import type { Generator, GenerateOptions, GenerationResult } from './generator.js';

export interface RagConfig<
  M extends Metadata = Metadata,
  P extends EmbeddingProvider = EmbeddingProvider,
  S extends VectorStore<M> = VectorStore<M>,
> {
  provider: P;
  store: S;
  chunker?: Chunker<M>;
  generator?: Generator<M>;
  namespace?: string;
  chunk?: ChunkOptions;
}

export interface RagSDK<M extends Metadata = Metadata> {
  ingest(documents: Document<M>[], options?: IngestOptions): Promise<void>;
  query(text: string, options?: QueryOptions): Promise<QueryResult<M>>;
  generate(text: string, options?: GeneratePipelineOptions): Promise<GenerationResult<M>>;
  generateStream(text: string, options?: GeneratePipelineOptions): AsyncGenerator<string, void, undefined>;
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

export interface GeneratePipelineOptions extends QueryOptions {
  generate?: GenerateOptions;
}
