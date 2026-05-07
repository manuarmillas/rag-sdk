import type { Metadata, SearchResult } from './document.js';

export interface Generator<M extends Metadata = Metadata> {
  readonly id: string;
  readonly modelId: string;
  generate(request: GenerateRequest<M>, options?: GenerateOptions): Promise<GenerationResult<M>>;
  generateStream?(request: GenerateRequest<M>, options?: GenerateOptions): AsyncGenerator<string, void, undefined>;
}

export interface GenerateRequest<M extends Metadata = Metadata> {
  query: string;
  context: SearchResult<M>[];
  systemPrompt?: string;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
}

export interface GenerationUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

export interface GenerationResult<M extends Metadata = Metadata> {
  query: string;
  answer: string;
  context: SearchResult<M>[];
  modelId: string;
  usage?: GenerationUsage;
}
