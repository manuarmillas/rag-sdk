import type { EmbeddingProvider } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface CohereEmbeddingConfig {
  apiKey?: string;
  model?: string;
  dimensions?: number;
  inputType?: 'search_document' | 'search_query' | 'classification' | 'clustering';
}

export function createCohereEmbedding(
  config?: CohereEmbeddingConfig,
): EmbeddingProvider {
  const modelId = config?.model ?? 'embed-english-v3.0';
  const dimensions = config?.dimensions ?? 1024;
  const inputType = config?.inputType;

  return {
    id: 'cohere',
    modelId,
    dimensions,

    async embed(text: string): Promise<number[]> {
      try {
        await requirePeer('cohere-ai', 'Package "cohere-ai" is required for Cohere embeddings');
        const { CohereClient } = await import('cohere-ai');
        const client = new CohereClient({ token: config?.apiKey ?? '' });

        const response = await client.embed({
          texts: [text],
          model: modelId,
          ...(inputType ? { inputType } : {}),
          embeddingTypes: ['float'],
        });

        return ((response.embeddings as { float: number[][] }).float)[0];
      } catch (err) {
        throw new ProviderError('cohere', 'embed', err);
      }
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      try {
        await requirePeer('cohere-ai', 'Package "cohere-ai" is required for Cohere embeddings');
        const { CohereClient } = await import('cohere-ai');
        const client = new CohereClient({ token: config?.apiKey ?? '' });

        const response = await client.embed({
          texts,
          model: modelId,
          ...(inputType ? { inputType } : {}),
          embeddingTypes: ['float'],
        });

        return (response.embeddings as { float: number[][] }).float;
      } catch (err) {
        throw new ProviderError('cohere', 'embedBatch', err);
      }
    },
  };
}
