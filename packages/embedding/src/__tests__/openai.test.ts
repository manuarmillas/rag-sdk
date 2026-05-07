import { describe, it, expect } from 'vitest';
import { createOpenAI } from '../openai.js';
import { ConfigurationError } from '@rag-sdk/core';

describe('createOpenAI', () => {
  it('throws ConfigurationError for custom model without dimensions', () => {
    expect(() =>
      createOpenAI({ model: 'text-embedding-3-large' }),
    ).toThrow(ConfigurationError);
  });

  it('works with custom model and explicit dimensions', () => {
    const provider = createOpenAI({
      model: 'text-embedding-3-large',
      dimensions: 3072,
    });
    expect(provider.dimensions).toBe(3072);
    expect(provider.modelId).toBe('text-embedding-3-large');
  });

  it('works with default model without explicit dimensions', () => {
    const provider = createOpenAI();
    expect(provider.dimensions).toBe(1536);
    expect(provider.modelId).toBe('text-embedding-3-small');
  });
});
