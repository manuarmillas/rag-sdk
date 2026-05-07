import { describe, it, expect } from 'vitest';
import type { EmbeddingProvider } from '../types/provider.js';

function hashText(text: string, dimensions: number): number[] {
  const vec: number[] = [];
  for (let i = 0; i < dimensions; i++) {
    let hash = 0;
    for (let j = 0; j < text.length; j++) {
      hash = (hash * 31 + text.charCodeAt(j) + i * 7) % 1000;
    }
    vec.push(hash / 1000);
  }
  return vec;
}

function createContractProvider(opts?: {
  dimensions?: number;
  maxBatchSize?: number;
}): EmbeddingProvider {
  const dimensions = opts?.dimensions ?? 3;
  return {
    id: 'contract-test',
    modelId: 'contract-model',
    dimensions,
    maxBatchSize: opts?.maxBatchSize,
    async embed(text: string): Promise<number[]> {
      return hashText(text, dimensions);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map((text) => hashText(text, dimensions));
    },
  };
}

describe('EmbeddingProvider contract', () => {
  it('has readonly id, modelId, and dimensions', () => {
    const provider = createContractProvider();

    expect(provider.id).toBe('contract-test');
    expect(provider.modelId).toBe('contract-model');
    expect(provider.dimensions).toBe(3);
    expect(typeof provider.id).toBe('string');
    expect(typeof provider.modelId).toBe('string');
    expect(typeof provider.dimensions).toBe('number');
  });

  it('id and modelId are non-empty strings', () => {
    const provider = createContractProvider();

    expect(provider.id.length).toBeGreaterThan(0);
    expect(provider.modelId.length).toBeGreaterThan(0);
  });

  it('dimensions is a positive integer', () => {
    const provider = createContractProvider({ dimensions: 1536 });

    expect(provider.dimensions).toBeGreaterThan(0);
    expect(Number.isInteger(provider.dimensions)).toBe(true);
  });

  it('embed returns Promise<number[]> with correct length', async () => {
    const provider = createContractProvider({ dimensions: 4 });

    const result = await provider.embed('hello');

    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(4);
    result.forEach((val) => expect(typeof val).toBe('number'));
  });

  it('embed returns consistent-length vectors', async () => {
    const provider = createContractProvider({ dimensions: 5 });

    const r1 = await provider.embed('a');
    const r2 = await provider.embed('hello world');

    expect(r1.length).toBe(5);
    expect(r2.length).toBe(5);
  });

  it('embedBatch returns Promise<number[][]> with correct count', async () => {
    const provider = createContractProvider({ dimensions: 3 });

    const results = await provider.embedBatch(['a', 'b', 'c']);

    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBe(3);
    results.forEach((vec) => {
      expect(Array.isArray(vec)).toBe(true);
      expect(vec.length).toBe(3);
    });
  });

  it('embedBatch returns empty array for empty input', async () => {
    const provider = createContractProvider();

    const results = await provider.embedBatch([]);

    expect(results).toEqual([]);
  });

  it('maxBatchSize is optional and a positive integer when present', () => {
    const without = createContractProvider();
    expect(without.maxBatchSize).toBeUndefined();

    const withSize = createContractProvider({ maxBatchSize: 2048 });
    expect(withSize.maxBatchSize).toBe(2048);
    expect(Number.isInteger(withSize.maxBatchSize!)).toBe(true);
  });

  it('embed embeds a single text (not array)', async () => {
    const provider = createContractProvider({ dimensions: 2 });

    const result = await provider.embed('single');

    // Should be a flat array, not nested
    expect(result.length).toBe(2);
    expect(typeof result[0]).toBe('number');
    expect(Array.isArray(result[0])).toBe(false);
  });

  it('embedBatch each result matches dimensions', async () => {
    const provider = createContractProvider({ dimensions: 8 });

    const results = await provider.embedBatch(['x', 'yy', 'zzz']);

    results.forEach((vec) => {
      expect(vec.length).toBe(8);
    });
  });

  it('different inputs produce different embeddings', async () => {
    const provider = createContractProvider({ dimensions: 10 });

    const r1 = await provider.embed('cat');
    const r2 = await provider.embed('dog');

    const areEqual = r1.every((val, i) => val === r2[i]);
    expect(areEqual).toBe(false);
  });
});
