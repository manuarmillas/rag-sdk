import type { Generator, GenerationResult, GenerateOptions, GenerateRequest, Metadata } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { buildPrompt } from './prompt.js';

export interface OllamaGeneratorConfig {
  baseURL?: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
}

async function postJson(baseURL: string, path: string, body: unknown): Promise<Response> {
  const response = await fetch(`${baseURL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`HTTP ${response.status} ${response.statusText}: ${text}`);
  }

  return response;
}

export function createOllamaGenerator(
  config: OllamaGeneratorConfig,
): Generator {
  const baseURL = config.baseURL ?? 'http://localhost:11434';
  const modelId = config.model;

  return {
    id: 'ollama',
    modelId,

    async generate<M extends Metadata>(
      request: GenerateRequest<M>,
      options?: GenerateOptions,
    ): Promise<GenerationResult<M>> {
      const prompt = buildPrompt(request);

      try {
        const response = await postJson(baseURL, '/api/chat', {
          model: modelId,
          messages: [
            ...(options?.systemPrompt
              ? [{ role: 'system', content: options.systemPrompt }]
              : []),
            { role: 'user', content: prompt },
          ],
          stream: false,
          options: {
            temperature: options?.temperature ?? config.temperature,
            num_predict: options?.maxTokens ?? config.maxTokens,
          },
        });

        const data = (await response.json()) as {
          message?: { content?: string };
        };
        const answer = data.message?.content ?? '';

        return {
          query: request.query,
          answer,
          context: request.context,
          modelId,
        };
      } catch (err) {
        throw new ProviderError('ollama', 'generate', err);
      }
    },

    async *generateStream<M extends Metadata>(
      request: GenerateRequest<M>,
      options?: GenerateOptions,
    ): AsyncGenerator<string, void, undefined> {
      const prompt = buildPrompt(request);

      try {
        const response = await postJson(baseURL, '/api/chat', {
          model: modelId,
          messages: [
            ...(options?.systemPrompt
              ? [{ role: 'system', content: options.systemPrompt }]
              : []),
            { role: 'user', content: prompt },
          ],
          stream: true,
          options: {
            temperature: options?.temperature ?? config.temperature,
            num_predict: options?.maxTokens ?? config.maxTokens,
          },
        });

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-argument
        while (true) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          const { done, value } = await reader.read();
          if (done || !value) break;

          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            const jsonStr = trimmed.startsWith('data: ')
              ? trimmed.slice(6)
              : trimmed;

            if (!jsonStr) continue;

            try {
              const chunk = JSON.parse(jsonStr) as {
                message?: { content?: string };
                response?: string;
                done?: boolean;
              };

              if (chunk.done) continue;

              const content = chunk.message?.content ?? chunk.response ?? '';
              if (content) {
                yield content;
              }
            } catch {
              // Skip malformed lines
            }
          }
        }

        // Process any remaining data in buffer
        if (buffer.trim()) {
          const jsonStr = buffer.trim().startsWith('data: ')
            ? buffer.trim().slice(6)
            : buffer.trim();
          try {
            const chunk = JSON.parse(jsonStr) as {
              message?: { content?: string };
              response?: string;
            };
            const content = chunk.message?.content ?? chunk.response ?? '';
            if (content) {
              yield content;
            }
          } catch {
            // Skip malformed final line
          }
        }
      } catch (err) {
        throw new ProviderError('ollama', 'generateStream', err);
      }
    },
  };
}
