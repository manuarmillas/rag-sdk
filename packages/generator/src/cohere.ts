import type { Generator, GenerationResult, GenerateOptions, GenerateRequest, Metadata } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';
import { buildPrompt } from './prompt.js';

export interface CohereGeneratorConfig {
  apiKey?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export function createCohereGenerator(
  config?: CohereGeneratorConfig,
): Generator {
  const modelId = config?.model ?? 'command-r-plus';

  return {
    id: 'cohere',
    modelId,

    async generate<M extends Metadata>(
      request: GenerateRequest<M>,
      options?: GenerateOptions,
    ): Promise<GenerationResult<M>> {
      const prompt = buildPrompt(request);

      try {
        await requirePeer('cohere-ai', 'Package "cohere-ai" is required for Cohere generation');
        const { CohereClient } = await import('cohere-ai');
        const client = new CohereClient({ token: config?.apiKey ?? '' });

        const response = await client.chat({
          message: prompt,
          model: modelId,
          preamble: options?.systemPrompt,
          temperature: options?.temperature ?? config?.temperature,
          maxTokens: options?.maxTokens ?? config?.maxTokens,
        });

        const answer = response.text ?? '';

        return {
          query: request.query,
          answer,
          context: request.context,
          modelId,
          usage: response.meta?.tokens
            ? {
                promptTokens: response.meta.tokens.inputTokens,
                completionTokens: response.meta.tokens.outputTokens,
                totalTokens:
                  (response.meta.tokens.inputTokens ?? 0) +
                  (response.meta.tokens.outputTokens ?? 0),
              }
            : undefined,
        };
      } catch (err) {
        throw new ProviderError('cohere', 'generate', err);
      }
    },

    async *generateStream<M extends Metadata>(
      request: GenerateRequest<M>,
      options?: GenerateOptions,
    ): AsyncGenerator<string, void, undefined> {
      const prompt = buildPrompt(request);

      try {
        await requirePeer('cohere-ai', 'Package "cohere-ai" is required for Cohere generation');
        const { CohereClient } = await import('cohere-ai');
        const client = new CohereClient({ token: config?.apiKey ?? '' });

        const stream = await client.chatStream({
          message: prompt,
          model: modelId,
          preamble: options?.systemPrompt,
          temperature: options?.temperature ?? config?.temperature,
          maxTokens: options?.maxTokens ?? config?.maxTokens,
        });

        for await (const event of stream) {
          if (event.eventType === 'text-generation') {
            const text = (event as { text?: string }).text;
            if (text) {
              yield text;
            }
          }
        }
      } catch (err) {
        throw new ProviderError('cohere', 'generateStream', err);
      }
    },
  };
}
