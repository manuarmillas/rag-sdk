import type { EmbeddingProvider } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';

export interface OllamaEmbeddingConfig {
  baseURL?: string;
  model: string;
  dimensions?: number;
}

async function postJson(baseURL: string, path: string, body: unknown): Promise<unknown> {
  const response = await fetch(`${baseURL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`HTTP ${response.status} ${response.statusText}: ${text}`);
  }

  return response.json();
}

export function createOllamaEmbedding(
  config: OllamaEmbeddingConfig,
): EmbeddingProvider {
  const baseURL = config.baseURL ?? 'http://localhost:11434';
  const modelId = config.model;
  const dimensions = config.dimensions ?? 4096;

  async function doEmbed(text: string): Promise<number[]> {
    const data = await postJson(baseURL, '/api/embeddings', { model: modelId, prompt: text });
    return (data as { embedding: number[] }).embedding;
  }

  return {
    id: 'ollama',
    modelId,
    dimensions,

    async embed(text: string): Promise<number[]> {
      try {
        return await doEmbed(text);
      } catch (err) {
        throw new ProviderError('ollama', 'embed', err);
      }
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      try {
        return await Promise.all(texts.map((text) => doEmbed(text)));
      } catch (err) {
        throw new ProviderError('ollama', 'embedBatch', err);
      }
    },
  };
}
