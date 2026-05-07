import { describe, it, expect } from 'vitest';
import { SemanticChunker } from '../semantic-chunker.js';
import type { EmbeddingProvider } from '@rag-sdk/core';

function fakeProvider(
  strategy: 'sequential' | 'alternating' | 'random',
): EmbeddingProvider {
  return {
    id: 'fake',
    modelId: 'fake-model',
    dimensions: 4,
    async embed(text: string): Promise<number[]> {
      return this.embedBatch([text]).then((r) => r[0] ?? []);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map((t, i) => {
        if (strategy === 'sequential') {
          // Each text gets a slightly different vector
          const base = i * 0.1;
          return [1 - base, base, 0.5, 0.5];
        }
        if (strategy === 'alternating') {
          // Two distinct clusters
          return i % 2 === 0 ? [1, 0, 0, 0] : [0, 1, 0, 0];
        }
        // Random but deterministic per text
        let hash = 0;
        for (let j = 0; j < t.length; j++) {
          hash = (hash << 5) - hash + t.charCodeAt(j);
          hash |= 0;
        }
        const norm = Math.abs(hash) % 1000;
        return [
          (norm % 10) / 10,
          ((norm * 7) % 10) / 10,
          ((norm * 13) % 10) / 10,
          ((norm * 31) % 10) / 10,
        ];
      });
    },
  };
}

describe('SemanticChunker', () => {
  it('returns empty array for empty documents', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'));
    const chunks = await chunker.chunk([]);
    expect(chunks).toEqual([]);
  });

  it('skips documents with empty content', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'));
    const chunks = await chunker.chunk([{ content: '' }]);
    expect(chunks).toEqual([]);
  });

  it('chunks single document without semantic breaks when similar', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'), {
      chunkSize: 1000,
      threshold: 0.9, // high threshold = few breaks
    });
    const docs = [
      {
        id: 'doc-1',
        content: 'First paragraph. Second paragraph. Third paragraph.',
      },
    ];
    const chunks = await chunker.chunk(docs);
    expect(chunks.length).toBe(1);
    expect(chunks[0].documentId).toBe('doc-1');
    expect(chunks[0].chunkIndex).toBe(0);
  });

  it('splits at semantic boundaries when threshold is low', async () => {
    const chunker = new SemanticChunker(fakeProvider('alternating'), {
      chunkSize: 1000,
      threshold: 0.5,
    });
    const docs = [
      {
        id: 'doc-2',
        content:
          'Alpha text here. Alpha more. Beta text here. Beta more. Alpha again.',
      },
    ];
    const chunks = await chunker.chunk(docs);
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('respects chunkSize by splitting large semantic groups', async () => {
    const longText = 'word '.repeat(500);
    const chunker = new SemanticChunker(fakeProvider('random'), {
      chunkSize: 100,
      threshold: 0.0, // never break on semantics
    });
    const docs = [{ id: 'doc-3', content: longText }];
    const chunks = await chunker.chunk(docs);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.content.length).toBeLessThanOrEqual(100 + 20); // allow small overshoot from token joining
    }
  });

  it('preserves document metadata', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'));
    const docs = [
      {
        id: 'doc-4',
        content: 'Hello world.',
        metadata: { source: 'test' },
      },
    ];
    const chunks = await chunker.chunk(docs);
    expect(chunks[0].metadata).toMatchObject({ source: 'test' });
  });

  it('assigns deterministic chunk ids for same input', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'));
    const docs = [{ id: 'doc-5', content: 'Hello world.' }];
    const chunks1 = await chunker.chunk(docs);
    const chunks2 = await chunker.chunk(docs);
    expect(chunks1[0].id).toBe(chunks2[0].id);
  });

  it('assigns sequential chunkIndex per document', async () => {
    const chunker = new SemanticChunker(fakeProvider('alternating'), {
      chunkSize: 50,
      threshold: 0.5,
    });
    const docs = [
      {
        id: 'doc-6',
        content:
          'A1 A2 A3. B1 B2 B3. A4 A5 A6. B4 B5 B6. A7 A8 A9. B7 B8 B9.',
      },
    ];
    const chunks = await chunker.chunk(docs);
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].chunkIndex).toBe(i);
    }
  });

  it('tracks correct character offsets', async () => {
    const text = 'Hello world. This is a test.';
    const chunker = new SemanticChunker(fakeProvider('sequential'), {
      chunkSize: 1000,
      threshold: 0.9,
    });
    const docs = [{ id: 'doc-7', content: text }];
    const chunks = await chunker.chunk(docs);
    for (const chunk of chunks) {
      expect(text.slice(chunk.startChar, chunk.endChar)).toBe(chunk.content);
    }
  });

  it('works with async chunk() signature', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'));
    const result = chunker.chunk([{ content: 'Test' }]);
    expect(result).toBeInstanceOf(Promise);
    const chunks = await result;
    expect(chunks.length).toBe(1);
  });

  it('generates fallback document id when missing', async () => {
    const chunker = new SemanticChunker(fakeProvider('sequential'));
    const docs = [{ content: 'No id here.' }];
    const chunks = await chunker.chunk(docs);
    expect(chunks[0].documentId).toBeDefined();
    expect(typeof chunks[0].documentId).toBe('string');
    expect(chunks[0].documentId.length).toBeGreaterThan(0);
  });
});
