export interface EmbeddingProvider {
  readonly id: string;
  readonly modelId: string;
  readonly dimensions: number;
  readonly maxBatchSize?: number;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}
