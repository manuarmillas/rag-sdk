import { describe, it, expect } from 'vitest';
import { RecursiveCharacterTextSplitter } from '../chunker/recursive-splitter.js';

describe('RecursiveCharacterTextSplitter', () => {
  it('splits text by separators', () => {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 20,
      overlap: 0,
    });
    const docs = [
      { content: 'Hello world.\n\nThis is a test.\nAnother line.' },
    ];
    const chunks = splitter.chunk(docs);
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].content).toBeDefined();
  });

  it('preserves overlap between chunks', () => {
    const text = 'a'.repeat(100);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 30,
      overlap: 10,
    });
    const docs = [{ content: text }];
    const chunks = splitter.chunk(docs);
    expect(chunks.length).toBeGreaterThan(1);
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1].content;
      const curr = chunks[i].content;
      expect(curr.startsWith(prev.slice(-10))).toBe(true);
    }
  });

  it('tracks correct character offsets', () => {
    const text = 'Hello world.\n\nThis is a test.';
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 20,
      overlap: 0,
    });
    const docs = [{ content: text }];
    const chunks = splitter.chunk(docs);
    for (const chunk of chunks) {
      expect(text.slice(chunk.startChar, chunk.endChar)).toBe(chunk.content);
    }
  });

  it('assigns sequential chunkIndex per document', () => {
    const text = 'a'.repeat(200);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 50,
      overlap: 0,
    });
    const docs = [{ content: text }];
    const chunks = splitter.chunk(docs);
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].chunkIndex).toBe(i);
    }
  });

  it('skips empty documents', () => {
    const splitter = new RecursiveCharacterTextSplitter();
    const docs = [{ content: '' }, { content: 'Hello' }];
    const chunks = splitter.chunk(docs);
    expect(chunks.length).toBe(1);
    expect(chunks[0].content).toBe('Hello');
  });

  it('assigns random UUID when document has no id', () => {
    const splitter = new RecursiveCharacterTextSplitter();
    const docs = [{ content: 'Hello' }];
    const chunks = splitter.chunk(docs);
    expect(chunks[0].documentId).toBeDefined();
    expect(typeof chunks[0].documentId).toBe('string');
    expect(chunks[0].documentId.length).toBeGreaterThan(0);
  });

  it('preserves provided document id', () => {
    const splitter = new RecursiveCharacterTextSplitter();
    const docs = [{ id: 'doc-1', content: 'Hello' }];
    const chunks = splitter.chunk(docs);
    expect(chunks[0].documentId).toBe('doc-1');
  });

  it('splits large document into multiple chunks', () => {
    const text = 'a'.repeat(2500);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      overlap: 200,
    });
    const docs = [{ content: text }];
    const chunks = splitter.chunk(docs);
    expect(chunks.length).toBeGreaterThan(2);
  });
});
