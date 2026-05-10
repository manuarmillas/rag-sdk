import type { EmbeddingProvider } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface VoyageEmbeddingConfig {
  apiKey?: string;
  model?: string;
  dimensions?: number;
  inputType?: 'query' | 'document';
}

export function createVoyageEmbedding(
  config?: VoyageEmbeddingConfig,
): EmbeddingProvider {
  const modelId = config?.model ?? 'voyage-3-lite';
  const dimensions = config?.dimensions ?? 512;
  const inputType = config?.inputType;

  return {
    id: 'voyage',
    modelId,
    dimensions,

    async embed(text: string): Promise<number[]> {
      try {
        await requirePeer('voyageai', 'Package "voyageai" is required for VoyageAI embeddings');
        const { VoyageAIClient } = await import('voyageai');
        const client = new VoyageAIClient({ apiKey: config?.apiKey ?? '' });

        const response = await client.embed({
          input: text,
          model: modelId,
          ...(inputType ? { inputType } : {}),
        });

        const data = response.data;
        if (!data || data.length === 0 || !data[0]?.embedding) {
          throw new Error('VoyageAI embed returned empty response');
        }

        return data[0].embedding;
      } catch (err) {
        throw new ProviderError('voyage', 'embed', err);
      }
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      try {
        await requirePeer('voyageai', 'Package "voyageai" is required for VoyageAI embeddings');
        const { VoyageAIClient } = await import('voyageai');
        const client = new VoyageAIClient({ apiKey: config?.apiKey ?? '' });

        const response = await client.embed({
          input: texts,
          model: modelId,
          ...(inputType ? { inputType } : {}),
        });

        const data = response.data;
        if (!data || data.length === 0) {
          throw new Error('VoyageAI embed returned empty response');
        }

        return data.map((d) => d.embedding ?? []);
      } catch (err) {
        throw new ProviderError('voyage', 'embedBatch', err);
      }
    },
  };
}
