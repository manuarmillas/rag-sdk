import type { EmbeddingProvider } from '@rag-sdk/core';
import { ProviderError, ConfigurationError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface CreateOpenAIConfig {
  apiKey?: string;
  model?: string;
  dimensions?: number;
}

export function createOpenAI(config?: CreateOpenAIConfig): EmbeddingProvider {
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
        await requirePeer('openai', 'Package "openai" is required for OpenAI embeddings');
        const { default: OpenAI } = await import('openai');
        const client = new OpenAI({ apiKey: config?.apiKey });

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
        await requirePeer('openai', 'Package "openai" is required for OpenAI embeddings');
        const { default: OpenAI } = await import('openai');
        const client = new OpenAI({ apiKey: config?.apiKey });

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
