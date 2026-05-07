// Factory
export { rag } from './rag.js';

// Types
export type {
  Metadata,
  MetadataFilter,
  VectorValues,
  Document,
  Chunk,
  Vector,
  SearchResult,
  QueryResult,
} from './types/document.js';

export type { EmbeddingProvider } from './types/provider.js';

export type { VectorStore, QueryOptions } from './types/store.js';

export type {
  RagConfig,
  RagSDK,
  ChunkOptions,
  IngestOptions,
} from './types/config.js';

export type { Chunker } from './chunker/types.js';

// Chunking
export { RecursiveCharacterTextSplitter } from './chunker/recursive-splitter.js';

// Validation
export { validateChunkOptions } from './validate.js';

// Errors
export {
  RagSdkError,
  ConfigurationError,
  ValidationError,
  ProviderError,
  StoreError,
  ChunkingError,
  DimensionMismatchError,
  BatchError,
} from './errors.js';
