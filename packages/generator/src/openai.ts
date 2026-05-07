import OpenAI from 'openai';
import type { Generator, GenerationResult, GenerateOptions, GenerateRequest } from '@rag-sdk/core';
import type { Metadata } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { buildPrompt } from './prompt.js';

export interface CreateOpenAIGeneratorConfig {
  apiKey?: string;
  model?: string;
  baseURL?: string;
}

export function createOpenAIGenerator(
  config?: CreateOpenAIGeneratorConfig,
): Generator {
  const client = new OpenAI({
    apiKey: config?.apiKey,
    baseURL: config?.baseURL,
  });
  const modelId = config?.model ?? 'gpt-4o-mini';

  return {
    id: 'openai',
    modelId,

    async generate<M extends Metadata>(
      request: GenerateRequest<M>,
      options?: GenerateOptions,
    ): Promise<GenerationResult<M>> {
      const prompt = buildPrompt(request);

      try {
        const response = await client.chat.completions.create({
          model: modelId,
          messages: [
            ...(options?.systemPrompt
              ? [{ role: 'system' as const, content: options.systemPrompt }]
              : []),
            { role: 'user' as const, content: prompt },
          ],
          max_tokens: options?.maxTokens,
          temperature: options?.temperature,
        });

        const choice = response.choices[0];
        const answer = choice?.message?.content ?? '';

        return {
          query: request.query,
          answer,
          context: request.context,
          modelId,
          usage: response.usage
            ? {
                promptTokens: response.usage.prompt_tokens,
                completionTokens: response.usage.completion_tokens,
                totalTokens: response.usage.total_tokens,
              }
            : undefined,
        };
      } catch (err) {
        throw new ProviderError('openai', 'generate', err);
      }
    },

    async *generateStream<M extends Metadata>(
      request: GenerateRequest<M>,
      options?: GenerateOptions,
    ): AsyncGenerator<string, void, undefined> {
      const prompt = buildPrompt(request);

      try {
        const stream = await client.chat.completions.create({
          model: modelId,
          messages: [
            ...(options?.systemPrompt
              ? [{ role: 'system' as const, content: options.systemPrompt }]
              : []),
            { role: 'user' as const, content: prompt },
          ],
          max_tokens: options?.maxTokens,
          temperature: options?.temperature,
          stream: true,
        });

        for await (const chunk of stream) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            yield content;
          }
        }
      } catch (err) {
        throw new ProviderError('openai', 'generateStream', err);
      }
    },
  };
}
