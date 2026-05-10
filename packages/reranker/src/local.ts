import type { Reranker, RerankOptions, SearchResult, Metadata } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface LocalRerankerConfig {
  model?: string;
  batchSize?: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyPipeline = (inputs: string[]) => Promise<unknown>;

const pipelines = new Map<string, Promise<AnyPipeline>>();

export function __resetPipelines(): void {
  pipelines.clear();
}

async function getPipeline(task: string, model: string): Promise<AnyPipeline> {
  const key = `${task}:${model}`;
  if (!pipelines.has(key)) {
    pipelines.set(
      key,
      (async () => {
        await requirePeer(
          '@huggingface/transformers',
          'Package "@huggingface/transformers" is required for local reranking',
        );
        const { pipeline } = await import('@huggingface/transformers');
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return pipeline(task as any, model) as unknown as AnyPipeline;
      })(),
    );
  }
  return pipelines.get(key)!;
}

export function createLocalReranker(
  config?: LocalRerankerConfig,
): Reranker {
  const modelId = config?.model ?? 'Xenova/ms-marco-MiniLM-L-6-v2';
  const batchSize = config?.batchSize ?? 32;

  return {
    id: 'local',
    modelId,

    async rerank<M extends Metadata>(
      query: string,
      results: SearchResult<M>[],
      options?: RerankOptions,
    ): Promise<SearchResult<M>[]> {
      if (results.length === 0) {
        return [];
      }

      try {
        const classifier = await getPipeline('text-classification', modelId);

        const inputs = results.map((r) => `${query} [SEP] ${r.content}`);
        const scores: number[] = [];

        for (let i = 0; i < inputs.length; i += batchSize) {
          const batch = inputs.slice(i, i + batchSize);
          const batchOutputs = await classifier(batch);
          const batchResults = Array.isArray(batchOutputs) ? batchOutputs : [batchOutputs];
          for (const output of batchResults) {
            scores.push((output as { score: number }).score);
          }
        }

        const reranked = results.map((result, index) => ({
          ...result,
          score: scores[index] ?? 0,
        }));

        reranked.sort((a, b) => b.score - a.score);

        if (options?.topN !== undefined) {
          return reranked.slice(0, options.topN);
        }

        return reranked;
      } catch (err) {
        throw new ProviderError('local', 'rerank', err);
      }
    },
  };
}
