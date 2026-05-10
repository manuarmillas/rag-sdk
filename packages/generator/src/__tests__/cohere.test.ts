import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createCohereGenerator } from '../cohere.js';
import { ProviderError } from '@rag-sdk/core';
import type { SearchResult } from '@rag-sdk/core';

// Mock the cohere-ai module
vi.mock('cohere-ai', () => {
  return {
    CohereClient: vi.fn(),
  };
});

import { CohereClient } from 'cohere-ai';

interface MockChatArgs {
  message: string;
  model?: string;
  preamble?: string;
  temperature?: number;
  maxTokens?: number;
}

let lastChatCall: MockChatArgs | undefined;

function createMockClient(opts?: { failGenerate?: boolean; failStream?: boolean }) {
  const mockChat = vi.fn(async () => ({
    text: 'This is the generated answer.',
    meta: {
      tokens: {
        inputTokens: 42,
        outputTokens: 8,
      },
    },
  }));

  const mockStream = {
    async *[Symbol.asyncIterator]() {
      if (opts?.failStream) throw new Error('stream error');
      yield { eventType: 'text-generation', text: 'Hello ' };
      yield { eventType: 'text-generation', text: 'world!' };
      yield { eventType: 'stream-end' };
    },
  };

  (CohereClient as unknown as ReturnType<typeof vi.fn>).mockImplementation(() => ({
    chat: vi.fn(async (...args: unknown[]) => {
      lastChatCall = args[0] as MockChatArgs;
      if (opts?.failGenerate) throw new Error('generate error');
      return mockChat();
    }),
    chatStream: vi.fn(async () => {
      if (opts?.failStream) throw new Error('stream error');
      return mockStream;
    }),
  }));

  return { mockChat };
}

describe('createCohereGenerator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    lastChatCall = undefined;
  });

  it('returns a Generator with correct id and modelId', () => {
    createMockClient();
    const generator = createCohereGenerator({ model: 'command-r' });
    expect(generator.id).toBe('cohere');
    expect(generator.modelId).toBe('command-r');
  });

  it('uses default model when not specified', () => {
    createMockClient();
    const generator = createCohereGenerator();
    expect(generator.modelId).toBe('command-r-plus');
  });

  it('generates text with token usage', async () => {
    createMockClient();
    const generator = createCohereGenerator();

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
    expect(result.modelId).toBe('command-r-plus');
    expect(result.usage).toEqual({
      promptTokens: 42,
      completionTokens: 8,
      totalTokens: 50,
    });
  });

  it('passes maxTokens and temperature to the API', async () => {
    createMockClient();
    const generator = createCohereGenerator({ temperature: 0.5, maxTokens: 100 });
    lastChatCall = undefined;

    await generator.generate(
      { query: 'test', context: [] },
      { maxTokens: 50, temperature: 0.3 },
    );

    expect(lastChatCall).toBeDefined();
    expect(lastChatCall!.maxTokens).toBe(50);
    expect(lastChatCall!.temperature).toBe(0.3);
  });

  it('passes systemPrompt as preamble', async () => {
    createMockClient();
    const generator = createCohereGenerator();

    await generator.generate(
      { query: 'test', context: [] },
      { systemPrompt: 'Be concise.' },
    );

    expect(lastChatCall).toBeDefined();
    expect(lastChatCall!.preamble).toBe('Be concise.');
  });

  it('wraps errors in ProviderError', async () => {
    createMockClient({ failGenerate: true });
    const generator = createCohereGenerator();

    await expect(
      generator.generate({ query: 'test', context: [] }),
    ).rejects.toThrow(ProviderError);
  });

  it('generateStream yields token chunks', async () => {
    createMockClient();
    const generator = createCohereGenerator();

    const chunks: string[] = [];
    for await (const chunk of generator.generateStream!({ query: 'test', context: [] })) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual(['Hello ', 'world!']);
  });

  it('generateStream wraps errors in ProviderError', async () => {
    createMockClient({ failStream: true });
    const generator = createCohereGenerator();

    const stream = generator.generateStream!({ query: 'test', context: [] });

    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of stream) {
        // consume
      }
    }).rejects.toThrow(ProviderError);
  });
});
