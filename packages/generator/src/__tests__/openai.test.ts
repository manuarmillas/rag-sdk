import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createOpenAIGenerator } from '../openai.js';
import { ProviderError } from '@rag-sdk/core';
import type { SearchResult } from '@rag-sdk/core';

// Mock the openai module
vi.mock('openai', () => {
  return {
    default: vi.fn(),
  };
});

import OpenAI from 'openai';

interface MockCreateArgs {
  model: string;
  messages: Array<{ role: string; content: string }>;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}

let lastCreateCall: MockCreateArgs | undefined;

function createMockClient(opts?: { failGenerate?: boolean; failStream?: boolean }) {
  const mockCreate = vi.fn(async () => ({
    choices: [
      {
        message: {
          content: 'This is the generated answer.',
        },
      },
    ],
    usage: {
      prompt_tokens: 42,
      completion_tokens: 8,
      total_tokens: 50,
    },
  }));

  const mockStream = {
    async *[Symbol.asyncIterator]() {
      if (opts?.failStream) throw new Error('stream error');
      yield { choices: [{ delta: { content: 'Hello ' } }] };
      yield { choices: [{ delta: { content: 'world!' } }] };
      yield { choices: [{ delta: { content: undefined } }] };
    },
  };

  (OpenAI as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
    chat: {
      completions: {
        create: vi.fn(async (...args: unknown[]) => {
          lastCreateCall = args[0] as MockCreateArgs;
          const params = lastCreateCall;
          if (params?.stream) {
            if (opts?.failStream) throw new Error('stream error');
            return mockStream;
          }
          if (opts?.failGenerate) throw new Error('generate error');
          return mockCreate();
        }),
      },
    },
  }));

  return { mockCreate };
}

describe('createOpenAIGenerator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns a Generator with correct id and modelId', () => {
    createMockClient();
    const generator = createOpenAIGenerator({ model: 'gpt-4' });
    expect(generator.id).toBe('openai');
    expect(generator.modelId).toBe('gpt-4');
  });

  it('uses default model when not specified', () => {
    createMockClient();
    const generator = createOpenAIGenerator();
    expect(generator.modelId).toBe('gpt-4o-mini');
  });

  it('builds prompt with context and calls chat.completions.create', async () => {
    createMockClient();
    const generator = createOpenAIGenerator();

    const context: SearchResult[] = [
      { id: 'doc-1', score: 0.9, content: 'The sky is blue.', metadata: { source: 'wiki' } },
    ];

    const result = await generator.generate({
      query: 'What color is the sky?',
      context,
      systemPrompt: 'You are a helpful assistant.',
    });

    expect(result.answer).toBe('This is the generated answer.');
    expect(result.query).toBe('What color is the sky?');
    expect(result.context).toBe(context);
    expect(result.modelId).toBe('gpt-4o-mini');
    expect(result.usage).toEqual({
      promptTokens: 42,
      completionTokens: 8,
      totalTokens: 50,
    });
  });

  it('passes maxTokens and temperature to the API', async () => {
    createMockClient();
    const generator = createOpenAIGenerator();
    lastCreateCall = undefined;

    await generator.generate(
      { query: 'test', context: [] },
      { maxTokens: 100, temperature: 0.5 },
    );

    expect(lastCreateCall).toBeDefined();
    expect(lastCreateCall!.max_tokens).toBe(100);
    expect(lastCreateCall!.temperature).toBe(0.5);
  });

  it('wraps errors in ProviderError', async () => {
    createMockClient({ failGenerate: true });
    const generator = createOpenAIGenerator();

    await expect(
      generator.generate({ query: 'test', context: [] }),
    ).rejects.toThrow(ProviderError);
  });

  it('generateStream yields token chunks', async () => {
    createMockClient();
    const generator = createOpenAIGenerator();

    const chunks: string[] = [];
    for await (const chunk of generator.generateStream!({ query: 'test', context: [] })) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual(['Hello ', 'world!']);
  });

  it('generateStream wraps errors in ProviderError', async () => {
    createMockClient({ failStream: true });
    const generator = createOpenAIGenerator();

    const stream = generator.generateStream!({ query: 'test', context: [] });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(ProviderError);
  });
});
