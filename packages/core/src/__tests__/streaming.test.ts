import { describe, it, expect, vi } from 'vitest';
import { generateStreamPipeline } from '../pipeline/generate-stream.js';
import { ConfigurationError, ProviderError } from '../errors.js';
import type { Metadata, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { QueryOptions } from '../types/store.js';
import type { Generator } from '../types/generator.js';

function createMockProvider(opts?: { dimensions?: number }): EmbeddingProvider {
  const dims = opts?.dimensions ?? 3;
  return {
    id: 'mock-provider',
    modelId: 'mock-model',
    dimensions: dims,
    async embed(text: string): Promise<number[]> {
      return Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map((text) =>
        Array.from({ length: dims }, (_, i) => text.charCodeAt(0) + i),
      );
    },
  };
}

function createMockStore<M extends Metadata = Metadata>(opts?: { dimensions?: number }) {
  return {
    dimensions: opts?.dimensions,
    upsert: vi.fn(async () => {}),
    query: vi.fn(
      async (_embedding: number[], _options: QueryOptions) =>
        [] as SearchResult<M>[],
    ),
    delete: vi.fn(async () => {}),
  };
}

function createMockStreamingGenerator<M extends Metadata = Metadata>(
  opts?: {
    tokens?: string[];
    failStream?: boolean;
    failMidStream?: boolean;
  },
): Generator<M> {
  return {
    id: 'mock-generator',
    modelId: 'gpt-mock',
    generate: vi.fn(async () => ({
      query: 'test',
      answer: 'mock',
      context: [],
      modelId: 'gpt-mock',
    })),
    generateStream: vi.fn(async function* () {
      if (opts?.failStream) throw new Error('stream init error');
      const tokens = opts?.tokens ?? ['Hello ', 'world!'];
      for (let i = 0; i < tokens.length; i++) {
        if (opts?.failMidStream && i === 1) throw new Error('mid stream error');
        yield tokens[i];
      }
    }),
  };
}

describe('generateStreamPipeline', () => {
  it('yields tokens in order from generator.generateStream', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'hello', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const generator = createMockStreamingGenerator({ tokens: ['First ', 'second ', 'third'] });

    const chunks: string[] = [];
    for await (const chunk of generateStreamPipeline('test query', { topK: 3 }, {
      provider,
      store,
      generator,
    })) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual(['First ', 'second ', 'third']);
    expect(store.query).toHaveBeenCalledTimes(1);
  });

  it('calls queryPipeline before streaming and forwards context', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'ctx1', metadata: { src: 'a' } },
      { id: '2', score: 0.8, content: 'ctx2', metadata: { src: 'b' } },
    ];
    store.query.mockResolvedValue(mockResults);

    const generator = createMockStreamingGenerator();

    const chunks: string[] = [];
    for await (const chunk of generateStreamPipeline('hello world', { topK: 2, generate: { systemPrompt: 'sys' } }, {
      provider,
      store,
      generator,
    })) {
      chunks.push(chunk);
    }

    expect(store.query).toHaveBeenCalledTimes(1);
    expect(generator.generateStream).toHaveBeenCalledWith(
      {
        query: 'hello world',
        context: mockResults,
        systemPrompt: 'sys',
      },
      { systemPrompt: 'sys' },
    );
  });

  it('throws ConfigurationError when generator is missing', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    const stream = generateStreamPipeline('test', {}, { provider, store });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(ConfigurationError);
  });

  it('throws ConfigurationError when generator lacks generateStream', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    const generator = createMockStreamingGenerator();
    delete (generator as { generateStream?: unknown }).generateStream;

    const stream = generateStreamPipeline('test', {}, { provider, store, generator });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(ConfigurationError);
  });

  it('propagates mid-stream errors', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([]);

    const generator = createMockStreamingGenerator({ failMidStream: true, tokens: ['ok ', 'fail ', 'nope'] });

    const stream = generateStreamPipeline('test', {}, { provider, store, generator });

    const chunks: string[] = [];
    await expect(async () => {
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
    }).rejects.toThrow(ProviderError);

    expect(chunks).toEqual(['ok ']);
  });

  it('wraps non-ProviderError stream errors in ProviderError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([]);

    const generator = createMockStreamingGenerator({ failStream: true });

    const stream = generateStreamPipeline('test', {}, { provider, store, generator });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(ProviderError);
  });

  it('re-throws existing ProviderError without wrapping', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([]);

    const existingError = new ProviderError('mock-gen', 'generateStream', new Error('network'));

    const generator: Generator = {
      id: 'mock-generator',
      modelId: 'gpt-mock',
      generate: vi.fn(async () => ({ query: 'test', answer: 'mock', context: [], modelId: 'gpt-mock' })),
      // eslint-disable-next-line require-yield
      generateStream: async function* () {
        throw existingError;
      },
    };

    const stream = generateStreamPipeline('test', {}, { provider, store, generator });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(existingError);
  });

  it('forwards generate options to generator.generateStream', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    store.query.mockResolvedValue([]);

    const generator = createMockStreamingGenerator();

    const chunks: string[] = [];
    for await (const chunk of generateStreamPipeline('test', {
      topK: 5,
      generate: { maxTokens: 100, temperature: 0.7, systemPrompt: 'be brief' },
    }, {
      provider,
      store,
      generator,
    })) {
      chunks.push(chunk);
    }

    expect(generator.generateStream).toHaveBeenCalledWith(
      expect.objectContaining({ query: 'test', systemPrompt: 'be brief' }),
      { maxTokens: 100, temperature: 0.7, systemPrompt: 'be brief' },
    );
  });
});
