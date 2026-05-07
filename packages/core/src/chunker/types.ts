import type { Metadata, Document, Chunk } from '../types/document.js';
import type { ChunkOptions } from '../types/config.js';

export interface Chunker<M extends Metadata = Metadata> {
  chunk(
    documents: Document<M>[],
    options?: ChunkOptions,
  ): Promise<Chunk<M>[]> | Chunk<M>[];
}
