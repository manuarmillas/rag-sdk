import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createVoyageEmbedding } from '../voyage.js';
import { ProviderError } from '@rag-sdk/core';

// Mock the voyageai module
vi.mock('voyageai', () => {
  return {
    VoyageAIClient: vi.fn(),
  };
});

import { VoyageAIClient } from 'voyageai';

interface MockEmbedArgs {
  input: string | string[];
  model: string;
  inputType?: string;
}

let lastEmbedCall: MockEmbedArgs | undefined;

function createMockClient(opts?: { failEmbed?: boolean }) {
  const mockEmbed = vi.fn(async () => ({
    data: [
      { embedding: [0.1, 0.2, 0.3], index: 0 },
      { embedding: [0.4, 0.5, 0.6], index: 1 },
    ],
  }));

  (VoyageAIClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
    embed: vi.fn(async (...args: unknown[]) => {
      lastEmbedCall = args[0] as MockEmbedArgs;
      if (opts?.failEmbed) throw new Error('embed error');
      return mockEmbed();
    }),
  }));

  return { mockEmbed };
}

describe('createVoyageEmbedding', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    lastEmbedCall = undefined;
  });

  it('returns an EmbeddingProvider with correct id and modelId', () => {
    createMockClient();
    const provider = createVoyageEmbedding({ model: 'voyage-3' });
    expect(provider.id).toBe('voyage');
    expect(provider.modelId).toBe('voyage-3');
  });

  it('uses default model and dimensions', () => {
    createMockClient();
    const provider = createVoyageEmbedding();
    expect(provider.modelId).toBe('voyage-3-lite');
    expect(provider.dimensions).toBe(512);
  });

  it('allows custom dimensions', () => {
    createMockClient();
    const provider = createVoyageEmbedding({ dimensions: 1024 });
    expect(provider.dimensions).toBe(1024);
  });

  it('embeds a single text', async () => {
    createMockClient();
    const provider = createVoyageEmbedding();

    const result = await provider.embed('hello world');

    expect(result).toEqual([0.1, 0.2, 0.3]);
    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.input).toBe('hello world');
    expect(lastEmbedCall!.model).toBe('voyage-3-lite');
  });

  it('embeds a batch of texts', async () => {
    createMockClient();
    const provider = createVoyageEmbedding();

    const result = await provider.embedBatch(['hello', 'world']);

    expect(result).toEqual([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.input).toEqual(['hello', 'world']);
  });

  it('passes inputType when specified', async () => {
    createMockClient();
    const provider = createVoyageEmbedding({ inputType: 'query' });

    await provider.embed('test');

    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.inputType).toBe('query');
  });

  it('does not pass inputType when undefined', async () => {
    createMockClient();
    const provider = createVoyageEmbedding();

    await provider.embed('test');

    expect(lastEmbedCall).toBeDefined();
    expect(lastEmbedCall!.inputType).toBeUndefined();
  });

  it('wraps SDK errors in ProviderError', async () => {
    createMockClient({ failEmbed: true });
    const provider = createVoyageEmbedding();

    await expect(provider.embed('test')).rejects.toThrow(ProviderError);
  });

  it('wraps SDK errors in ProviderError for embedBatch', async () => {
    createMockClient({ failEmbed: true });
    const provider = createVoyageEmbedding();

    await expect(provider.embedBatch(['test'])).rejects.toThrow(ProviderError);
  });
});
