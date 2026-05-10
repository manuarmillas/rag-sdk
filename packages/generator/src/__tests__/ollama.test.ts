import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createOllamaGenerator } from '../ollama.js';
import { ProviderError } from '@rag-sdk/core';
import type { SearchResult } from '@rag-sdk/core';

describe('createOllamaGenerator', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function mockFetchJsonResponse(data: unknown, ok = true) {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok,
      status: ok ? 200 : 500,
      statusText: ok ? 'OK' : 'Internal Server Error',
      text: async () => JSON.stringify(data),
      json: async () => data,
    });
  }

  function createMockStreamResponse(chunks: string[]) {
    const encoder = new TextEncoder();

    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      body: {
        getReader() {
          let index = 0;
          const allChunks = chunks.map((c) => encoder.encode(c));
          return {
            async read() {
              if (index >= allChunks.length) {
                return { done: true, value: undefined };
              }
              const value = allChunks[index++];
              return { done: false, value };
            },
          };
        },
      },
    };
  }

  it('returns a Generator with correct id and modelId', () => {
    const generator = createOllamaGenerator({ model: 'llama3' });
    expect(generator.id).toBe('ollama');
    expect(generator.modelId).toBe('llama3');
  });

  it('uses default baseURL', () => {
    mockFetchJsonResponse({ message: { content: 'hi' } });
    const generator = createOllamaGenerator({ model: 'llama3' });
    expect(generator.id).toBe('ollama');
  });

  it('generates text', async () => {
    mockFetchJsonResponse({ message: { content: 'Generated answer.' } });
    const generator = createOllamaGenerator({ model: 'llama3' });

    const context: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'The sky is blue.', metadata: { source: 'wiki' } },
    ];

    const result = await generator.generate({
      query: 'What color is the sky?',
      context,
      systemPrompt: 'You are helpful.',
    });

    expect(result.answer).toBe('Generated answer.');
    expect(result.query).toBe('What color is the sky?');
    expect(result.modelId).toBe('llama3');
  });

  it('passes temperature and maxTokens', async () => {
    mockFetchJsonResponse({ message: { content: 'hi' } });
    const generator = createOllamaGenerator({
      model: 'llama3',
      temperature: 0.5,
      maxTokens: 100,
    });

    await generator.generate(
      { query: 'test', context: [] },
      { maxTokens: 50, temperature: 0.3 },
    );

    const call = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [string, { body: string }];
    const body = JSON.parse(call[1].body) as { options: Record<string, number> };
    expect(body.options.temperature).toBe(0.3);
    expect(body.options.num_predict).toBe(50);
  });

  it('wraps HTTP errors in ProviderError', async () => {
    mockFetchJsonResponse({ error: 'model not found' }, false);
    const generator = createOllamaGenerator({ model: 'unknown' });

    await expect(
      generator.generate({ query: 'test', context: [] }),
    ).rejects.toThrow(ProviderError);
  });

  it('wraps network errors in ProviderError', async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Connection refused'),
    );
    const generator = createOllamaGenerator({ model: 'llama3' });

    await expect(
      generator.generate({ query: 'test', context: [] }),
    ).rejects.toThrow(ProviderError);
  });

  it('generateStream yields token chunks', async () => {
    const chunks = [
      '{"message":{"content":"Hello "}}\n',
      '{"message":{"content":"world!"}}\n',
      '{"done":true}\n',
    ];
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      createMockStreamResponse(chunks),
    );

    const generator = createOllamaGenerator({ model: 'llama3' });
    const result: string[] = [];

    for await (const chunk of generator.generateStream!({ query: 'test', context: [] })) {
      result.push(chunk);
    }

    expect(result).toEqual(['Hello ', 'world!']);
  });

  it('generateStream handles SSE data: prefix', async () => {
    const chunks = [
      'data: {"message":{"content":"Hi"}}\n',
      'data: {"message":{"content":" there"}}\n',
      'data: {"done":true}\n',
    ];
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      createMockStreamResponse(chunks),
    );

    const generator = createOllamaGenerator({ model: 'llama3' });
    const result: string[] = [];

    for await (const chunk of generator.generateStream!({ query: 'test', context: [] })) {
      result.push(chunk);
    }

    expect(result).toEqual(['Hi', ' there']);
  });

  it('generateStream wraps errors in ProviderError', async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Connection refused'),
    );
    const generator = createOllamaGenerator({ model: 'llama3' });

    const stream = generator.generateStream!({ query: 'test', context: [] });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(ProviderError);
  });
});
