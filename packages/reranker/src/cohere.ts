import { CohereClient } from 'cohere-ai';
import type { Reranker, RerankOptions } from '@rag-sdk/core';
import type { Metadata, SearchResult } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';

export interface CreateCohereRerankerConfig {
  apiKey?: string;
  model?: string;
}

export function createCohereReranker(
  config?: CreateCohereRerankerConfig,
): Reranker {
  const client = new CohereClient({
    token: config?.apiKey ?? '',
  });
  const modelId = config?.model ?? 'rerank-english-v3.0';

  return {
    id: 'cohere',
    modelId,

    async rerank<M extends Metadata>(
      query: string,
      results: SearchResult<M>[],
      options?: RerankOptions,
    ): Promise<SearchResult<M>[]> {
      if (results.length === 0) {
        return [];
      }

      const documents = results.map((r) => r.content);

      try {
        const response = await client.rerank({
          query,
          documents,
          model: modelId,
          ...(options?.topN !== undefined ? { topN: options.topN } : {}),
        });

        const reranked: SearchResult<M>[] = [];
        for (const item of response.results) {
          const original = results[item.index];
          if (original) {
            reranked.push({
              ...original,
              score: item.relevanceScore,
            });
          }
        }

        return reranked;
      } catch (err) {
        throw new ProviderError('cohere', 'rerank', err);
      }
    },
  };
}
