import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLocalReranker, __resetPipelines } from '../local.js';
import { ProviderError } from '@rag-sdk/core';
import type { SearchResult } from '@rag-sdk/core';

// Mock the @huggingface/transformers module
vi.mock('@huggingface/transformers', () => {
  return {
    pipeline: vi.fn(),
  };
});

import { pipeline } from '@huggingface/transformers';

interface MockPipeline {
  (inputs: string[]): Promise<Array<{ label: string; score: number }>>;
}

function createMockPipeline(scores: number[]): MockPipeline {
  return vi.fn(async (inputs: string[]) => {
    return inputs.map((_, i) => ({
      label: 'LABEL_0',
      score: scores[i] ?? 0.5,
    }));
  });
}

function setupPipeline(scores: number[]) {
  (pipeline as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    async () => createMockPipeline(scores),
  );
}

describe('createLocalReranker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetPipelines();
  });

  it('returns a Reranker with correct id and default modelId', () => {
    setupPipeline([0.9, 0.8]);
    const reranker = createLocalReranker();
    expect(reranker.id).toBe('local');
    expect(reranker.modelId).toBe('Xenova/ms-marco-MiniLM-L-6-v2');
  });

  it('uses custom model when specified', () => {
    setupPipeline([0.9]);
    const reranker = createLocalReranker({ model: 'custom-model' });
    expect(reranker.modelId).toBe('custom-model');
  });

  it('reorders results based on scores', async () => {
    setupPipeline([0.75, 0.98]);
    const reranker = createLocalReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first document', metadata: {} },
      { id: 'doc-2', score: 0.8, content: 'second document', metadata: {} },
    ];

    const reordered = await reranker.rerank('test query', results);

    expect(reordered.length).toBe(2);
    expect(reordered[0].id).toBe('doc-2');
    expect(reordered[1].id).toBe('doc-1');
  });

  it('replaces scores with pipeline scores', async () => {
    setupPipeline([0.75, 0.98]);
    const reranker = createLocalReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
      { id: 'doc-2', score: 0.8, content: 'second', metadata: {} },
    ];

    const reordered = await reranker.rerank('test query', results);

    expect(reordered[0].score).toBe(0.98);
    expect(reordered[1].score).toBe(0.75);
  });

  it('preserves all result fields through reranking', async () => {
    setupPipeline([0.95]);
    const reranker = createLocalReranker();

    const results: SearchResult[] = [
      {
        id: 'chunk-1',
        score: 0.9,
        content: 'hello world',
        metadata: { source: 'test' },
        documentId: 'doc-1',
        chunkIndex: 0,
        namespace: 'ns-a',
      },
    ];

    const reordered = await reranker.rerank('test query', results);

    expect(reordered.length).toBe(1);
    const r = reordered[0];
    expect(r.id).toBe('chunk-1');
    expect(r.content).toBe('hello world');
    expect(r.metadata).toEqual({ source: 'test' });
    expect(r.documentId).toBe('doc-1');
    expect(r.chunkIndex).toBe(0);
    expect(r.namespace).toBe('ns-a');
  });

  it('applies topN after sorting', async () => {
    setupPipeline([0.5, 0.9, 0.7]);
    const reranker = createLocalReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
      { id: 'doc-2', score: 0.8, content: 'second', metadata: {} },
      { id: 'doc-3', score: 0.7, content: 'third', metadata: {} },
    ];

    const reordered = await reranker.rerank('test query', results, { topN: 2 });

    expect(reordered.length).toBe(2);
    expect(reordered[0].id).toBe('doc-2');
    expect(reordered[1].id).toBe('doc-3');
  });

  it('returns empty array when input results are empty', async () => {
    setupPipeline([]);
    const reranker = createLocalReranker();

    const reordered = await reranker.rerank('test query', []);

    expect(reordered).toEqual([]);
  });

  it('does not call pipeline at factory creation time', () => {
    setupPipeline([0.9]);
    createLocalReranker();

    expect(pipeline).not.toHaveBeenCalled();
  });

  it('wraps errors in ProviderError', async () => {
    (pipeline as unknown as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      throw new Error('pipeline error');
    });
    const reranker = createLocalReranker();

    const results: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'first', metadata: {} },
    ];

    await expect(
      reranker.rerank('test query', results),
    ).rejects.toThrow(ProviderError);
  });
});
