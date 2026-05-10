import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createCohereEmbedding } from '../cohere.js';
import { ProviderError } from '@rag-sdk/core';

// Mock the cohere-ai module
vi.mock('cohere-ai', () => {
  return {
    CohereClient: vi.fn(),
  };
});

import { CohereClient } from 'cohere-ai';

interface MockEmbedArgs {
  texts: string[];
  model: string;
  inputType?: string;
  embeddingTypes: string[];
}

let lastEmbedCall: MockEmbedArgs | undefined;

function createMockClient(opts?: { failEmbed?: boolean }) {
  const mockEmbed = vi.fn(async () => ({
    embeddings: {
      float: [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
      ],
    },
  }));

  (CohereClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
    embed: vi.fn(async (...args: unknown[]) => {
      lastEmbedCall = args[0] as MockEmbedArgs;
      if (opts?.failEmbed) throw new Error('embed error');
      return mockEmbed();
    }),
  }));

  return { mockEmbed };
}

describe('createCohereEmbedding', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    lastEmbedCall = undefined;
  });

  it('returns an EmbeddingProvider with correct id and modelId', () => {
    createMockClient();
    const provider = createCohereEmbedding({ model: 'embed-multilingual-v3.0' });
    expect(provider.id).toBe('cohere');
    expect(provider.modelId).toBe('embed-multilingual-v3.0');
  });

  it('uses default model and dimensions', () => {
    createMockClient();
    const provider = createCohereEmbedding();
    expect(provider.modelId).toBe('embed-english-v3.0');
    expect(provider.dimensions).toBe(1024);
  });

  it('allows custom dimensions', () => {
    createMockClient();
    const provider = createCohereEmbedding({ dimensions: 512 });
    expect(provider.dimensions).toBe(512);
  });

  it('embeds a single text', async () => {
    createMockClient();
    const provider = createCohereEmbedding();

    const result = await provider.embed('hello world');

    expect(result).toEqual([0.1, 0.2, 0.3]);
    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.texts).toEqual(['hello world']);
    expect(lastEmbedCall!.model).toBe('embed-english-v3.0');
    expect(lastEmbedCall!.embeddingTypes).toEqual(['float']);
  });

  it('embeds a batch of texts', async () => {
    createMockClient();
    const provider = createCohereEmbedding();

    const result = await provider.embedBatch(['hello', 'world']);

    expect(result).toEqual([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.texts).toEqual(['hello', 'world']);
  });

  it('passes inputType when specified', async () => {
    createMockClient();
    const provider = createCohereEmbedding({ inputType: 'search_query' });

    await provider.embed('test');

    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.inputType).toBe('search_query');
  });

  it('does not pass inputType when undefined', async () => {
    createMockClient();
    const provider = createCohereEmbedding();

    await provider.embed('test');

    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.inputType).toBeUndefined();
  });

  it('wraps SDK errors in ProviderError', async () => {
    createMockClient({ failEmbed: true });
    const provider = createCohereEmbedding();

    await expect(provider.embed('test')).rejects.toThrow(ProviderError);
  });

  it('wraps SDK errors in ProviderError for embedBatch', async () => {
    createMockClient({ failEmbed: true });
    const provider = createCohereEmbedding();

    await expect(provider.embedBatch(['test'])).rejects.toThrow(ProviderError);
  });
});
