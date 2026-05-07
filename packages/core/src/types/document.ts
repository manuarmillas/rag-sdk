export type Metadata = Record<string, unknown>;
export type MetadataFilter = Record<string, unknown>;
export type VectorValues = readonly number[];

export interface Document<M extends Metadata = Metadata> {
  id?: string;
  content: string;
  metadata?: M;
}

export interface Chunk<M extends Metadata = Metadata> extends Document<M> {
  id: string;
  documentId: string;
  chunkIndex: number;
  startChar: number;
  endChar: number;
}

export interface Vector<M extends Metadata = Metadata> {
  id: string;
  values: number[];
  content: string;
  metadata: M;
  documentId: string;
  chunkIndex: number;
  namespace?: string;
}

export interface SearchResult<M extends Metadata = Metadata> {
  id: string;
  score: number;
  content: string;
  metadata: M;
  documentId?: string;
  chunkIndex?: number;
  namespace?: string;
}

export interface QueryResult<M extends Metadata = Metadata> {
  query: string;
  results: SearchResult<M>[];
  namespace?: string;
}
