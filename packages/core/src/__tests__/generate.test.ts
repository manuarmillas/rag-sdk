import { describe, it, expect, vi } from 'vitest';
import { generatePipeline } from '../pipeline/generate.js';
import { ConfigurationError, ProviderError } from '../errors.js';
import type { Metadata, SearchResult } from '../types/document.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { QueryOptions } from '../types/store.js';
import type { Generator, GenerationResult } from '../types/generator.js';

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

function createMockGenerator<M extends Metadata = Metadata>(
  opts?: {
    failGenerate?: boolean;
    result?: GenerationResult<M>;
  },
): Generator<M> {
  return {
    id: 'mock-generator',
    modelId: 'gpt-mock',
    generate: vi.fn(async (_req, _opts): Promise<GenerationResult<M>> => {
      if (opts?.failGenerate) throw new Error('generate failed');
      return (
        opts?.result ?? {
          query: 'test query',
          answer: 'mock answer',
          context: [],
          modelId: 'gpt-mock',
        }
      );
    }),
  };
}

describe('generatePipeline', () => {
  it('calls queryPipeline then generator.generate and returns result', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'hello', metadata: {} },
    ];
    store.query.mockResolvedValue(mockResults);

    const generator = createMockGenerator({
      result: {
        query: 'hello world',
        answer: 'generated answer',
        context: mockResults,
        modelId: 'gpt-mock',
        usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
      },
    });

    const result = await generatePipeline('hello world', { topK: 3 }, {
      provider,
      store,
      generator,
    });

    expect(store.query).toHaveBeenCalledTimes(1);
    expect(generator.generate).toHaveBeenCalledTimes(1);
    expect(result.answer).toBe('generated answer');
    expect(result.query).toBe('hello world');
    expect(result.context).toBe(mockResults);
    expect(result.modelId).toBe('gpt-mock');
    expect(result.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 });
  });

  it('throws ConfigurationError when generator is missing', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });

    await expect(
      generatePipeline('hello', {}, { provider, store }),
    ).rejects.toThrow(ConfigurationError);
  });

  it('forwards query text and context to generator.generate', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const mockResults: SearchResult[] = [
      { id: '1', score: 0.9, content: 'context one', metadata: { src: 'a' } },
      { id: '2', score: 0.8, content: 'context two', metadata: { src: 'b' } },
    ];
    store.query.mockResolvedValue(mockResults);

    const generator = createMockGenerator();

    await generatePipeline(
      'what is this?',
      { topK: 2, generate: { maxTokens: 100, temperature: 0.5, systemPrompt: 'sys' } },
      { provider, store, generator },
    );

    expect(generator.generate).toHaveBeenCalledWith(
      {
        query: 'what is this?',
        context: mockResults,
        systemPrompt: 'sys',
      },
      { maxTokens: 100, temperature: 0.5, systemPrompt: 'sys' },
    );
  });

  it('wraps generator errors in ProviderError', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const generator = createMockGenerator({ failGenerate: true });

    await expect(
      generatePipeline('hello', {}, { provider, store, generator }),
    ).rejects.toThrow(ProviderError);
  });

  it('re-throws existing ProviderError without wrapping', async () => {
    const provider = createMockProvider({ dimensions: 3 });
    const store = createMockStore({ dimensions: 3 });
    const generator = createMockGenerator();
    const existingError = new ProviderError('mock-gen', 'generate', new Error('network'));
    (generator.generate as ReturnType<typeof vi.fn>).mockRejectedValue(existingError);

    await expect(
      generatePipeline('hello', {}, { provider, store, generator }),
    ).rejects.toThrow(existingError);
  });
});
