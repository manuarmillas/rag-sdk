import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLocalEmbedding, __resetPipelines } from '../local.js';
import { ProviderError } from '@rag-sdk/core';

// Mock the @huggingface/transformers module
vi.mock('@huggingface/transformers', () => {
  return {
    pipeline: vi.fn(),
  };
});

import { pipeline } from '@huggingface/transformers';

interface MockPipeline {
  (inputs: string[]): Promise<unknown>;
}

function createMockPipeline(embeddings: number[][]): MockPipeline {
  return vi.fn(async (inputs: string[]) => {
    return inputs.map((_, i) => embeddings[i] ?? [0.1, 0.2, 0.3]);
  });
}

function setupPipeline(embeddings: number[][]) {
  (pipeline as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    async () => createMockPipeline(embeddings),
  );
}

describe('createLocalEmbedding', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetPipelines();
  });

  it('returns an EmbeddingProvider with correct id and default modelId', () => {
    setupPipeline([[0.1, 0.2, 0.3]]);
    const provider = createLocalEmbedding();
    expect(provider.id).toBe('local');
    expect(provider.modelId).toBe('Xenova/all-MiniLM-L6-v2');
  });

  it('uses custom model when specified', () => {
    setupPipeline([[0.1, 0.2, 0.3]]);
    const provider = createLocalEmbedding({ model: 'custom-model' });
    expect(provider.modelId).toBe('custom-model');
  });

  it('uses default dimensions', () => {
    setupPipeline([[0.1, 0.2, 0.3]]);
    const provider = createLocalEmbedding();
    expect(provider.dimensions).toBe(384);
  });

  it('embeds a single text', async () => {
    setupPipeline([[0.1, 0.2, 0.3]]);
    const provider = createLocalEmbedding();

    const result = await provider.embed('hello world');

    expect(result).toEqual([0.1, 0.2, 0.3]);
  });

  it('embeds a batch of texts', async () => {
    setupPipeline([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    const provider = createLocalEmbedding();

    const result = await provider.embedBatch(['hello', 'world']);

    expect(result).toEqual([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
  });

  it('does not call pipeline at factory creation time', () => {
    setupPipeline([[0.1, 0.2, 0.3]]);
    createLocalEmbedding();

    expect(pipeline).not.toHaveBeenCalled();
  });

  it('wraps pipeline errors in ProviderError', async () => {
    (pipeline as unknown as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      throw new Error('pipeline error');
    });
    const provider = createLocalEmbedding();

    await expect(provider.embed('test')).rejects.toThrow(ProviderError);
  });

  it('wraps pipeline errors in ProviderError for embedBatch', async () => {
    (pipeline as unknown as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      throw new Error('pipeline error');
    });
    const provider = createLocalEmbedding();

    await expect(provider.embedBatch(['test'])).rejects.toThrow(ProviderError);
  });
});
