import OpenAI from 'openai';
import type { EmbeddingProvider } from '@rag-sdk/core';
import { ProviderError, ConfigurationError } from '@rag-sdk/core';

export interface CreateOpenAIConfig {
  apiKey?: string;
  model?: string;
  dimensions?: number;
}

export function createOpenAI(config?: CreateOpenAIConfig): EmbeddingProvider {
  const client = new OpenAI({ apiKey: config?.apiKey });
  const modelId = config?.model ?? 'text-embedding-3-small';
  const dimensions = config?.dimensions;

  if (modelId !== 'text-embedding-3-small' && dimensions === undefined) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      `OpenAI model "${modelId}" requires an explicit dimensions value`,
    );
  }

  return {
    id: 'openai',
    modelId,
    dimensions: dimensions ?? 1536,
    maxBatchSize: 2048,

    async embed(text: string): Promise<number[]> {
      try {
        const response = await client.embeddings.create({
          model: modelId,
          input: text,
          ...(dimensions !== undefined ? { dimensions } : {}),
        });
        return response.data[0].embedding;
      } catch (err) {
        throw new ProviderError('openai', 'embed', err);
      }
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      try {
        const response = await client.embeddings.create({
          model: modelId,
          input: texts,
          ...(dimensions !== undefined ? { dimensions } : {}),
        });
        return response.data.map((d) => d.embedding);
      } catch (err) {
        throw new ProviderError('openai', 'embedBatch', err);
      }
    },
  };
}
