import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createOllamaEmbedding } from '../ollama.js';
import { ProviderError } from '@rag-sdk/core';

describe('createOllamaEmbedding', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function mockFetchResponse(data: unknown, ok = true) {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok,
      status: ok ? 200 : 500,
      statusText: ok ? 'OK' : 'Internal Server Error',
      text: async () => JSON.stringify(data),
      json: async () => data,
    });
  }

  it('returns an EmbeddingProvider with correct id and modelId', () => {
    const provider = createOllamaEmbedding({ model: 'llama3' });
    expect(provider.id).toBe('ollama');
    expect(provider.modelId).toBe('llama3');
  });

  it('uses default baseURL and dimensions', () => {
    const provider = createOllamaEmbedding({ model: 'llama3' });
    expect(provider.dimensions).toBe(4096);
  });

  it('allows custom baseURL and dimensions', () => {
    const provider = createOllamaEmbedding({
      model: 'llama3',
      baseURL: 'http://custom:11434',
      dimensions: 1024,
    });
    expect(provider.dimensions).toBe(1024);
  });

  it('embeds a single text', async () => {
    mockFetchResponse({ embedding: [0.1, 0.2, 0.3] });
    const provider = createOllamaEmbedding({ model: 'llama3' });

    const result = await provider.embed('hello world');

    expect(result).toEqual([0.1, 0.2, 0.3]);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:11434/api/embeddings',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'llama3', prompt: 'hello world' }),
      }),
    );
  });

  it('embeds a batch of texts', async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(async (_url: string, init: { body: string }) => {
      const body = JSON.parse(init.body) as { prompt: string };
      const embedding = body.prompt === 'hello' ? [0.1, 0.2] : [0.2, 0.4];
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => JSON.stringify({ embedding }),
        json: async () => ({ embedding }),
      };
    });

    const provider = createOllamaEmbedding({ model: 'llama3' });
    const result = await provider.embedBatch(['hello', 'world']);

    expect(result).toEqual([
      [0.1, 0.2],
      [0.2, 0.4],
    ]);
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
  });

  it('wraps HTTP errors in ProviderError', async () => {
    mockFetchResponse({ error: 'model not found' }, false);
    const provider = createOllamaEmbedding({ model: 'unknown' });

    await expect(provider.embed('test')).rejects.toThrow(ProviderError);
  });

  it('wraps network errors in ProviderError', async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Connection refused'),
    );
    const provider = createOllamaEmbedding({ model: 'llama3' });

    await expect(provider.embed('test')).rejects.toThrow(ProviderError);
  });
});
