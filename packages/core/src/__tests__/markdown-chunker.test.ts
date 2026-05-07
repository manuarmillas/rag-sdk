import { describe, it, expect } from 'vitest';
import { MarkdownChunker } from '../chunker/markdown-chunker.js';

describe('MarkdownChunker', () => {
  it('splits by top-level headings', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [
      {
        id: 'doc-1',
        content: '# Intro\nIntro text.\n# Body\nBody text.',
      },
    ];
    const chunks = chunker.chunk(docs);
    expect(chunks.length).toBe(2);
    expect(chunks[0].content).toBe('Intro text.');
    expect(chunks[0].metadata).toMatchObject({
      heading: 'Intro',
      headingLevel: 1,
      headingPath: ['Intro'],
    });
    expect(chunks[1].content).toBe('Body text.');
    expect(chunks[1].metadata).toMatchObject({
      heading: 'Body',
      headingLevel: 1,
      headingPath: ['Body'],
    });
  });

  it('maintains heading hierarchy in headingPath', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [
      {
        id: 'doc-2',
        content: '# Title\nTitle text.\n## Section A\nA text.\n### Sub\nSub text.\n## Section B\nB text.',
      },
    ];
    const chunks = chunker.chunk(docs);
    expect(chunks.length).toBe(4);

    expect(chunks[0].metadata).toMatchObject({
      heading: 'Title',
      headingLevel: 1,
      headingPath: ['Title'],
    });
    expect(chunks[1].metadata).toMatchObject({
      heading: 'Section A',
      headingLevel: 2,
      headingPath: ['Title', 'Section A'],
    });
    expect(chunks[2].metadata).toMatchObject({
      heading: 'Sub',
      headingLevel: 3,
      headingPath: ['Title', 'Section A', 'Sub'],
    });
    expect(chunks[3].metadata).toMatchObject({
      heading: 'Section B',
      headingLevel: 2,
      headingPath: ['Title', 'Section B'],
    });
  });

  it('handles headings up to level 6', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [
      {
        id: 'doc-3',
        content:
          '# h1\n# h2\n## h2\n### h3\n#### h4\n##### h5\n###### h6\nBody.',
      },
    ];
    const chunks = chunker.chunk(docs);
    const last = chunks[chunks.length - 1];
    expect(last.metadata).toMatchObject({
      heading: 'h6',
      headingLevel: 6,
    });
  });

  it('groups content by section', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [
      {
        id: 'doc-4',
        content: '# A\nLine 1\nLine 2\n# B\nLine 3\n',
      },
    ];
    const chunks = chunker.chunk(docs);
    expect(chunks[0].content).toBe('Line 1\nLine 2');
    expect(chunks[1].content).toBe('Line 3\n');
  });

  it('splits oversized sections with overlap', () => {
    const chunker = new MarkdownChunker({ chunkSize: 20, overlap: 5 });
    const docs = [
      {
        id: 'doc-5',
        content: '# Section\n' + 'a'.repeat(50),
      },
    ];
    const chunks = chunker.chunk(docs);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.metadata).toMatchObject({ heading: 'Section' });
    }
  });

  it('tracks correct character offsets', () => {
    const text = '# Hello\nWorld\n# Foo\nBar';
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [{ id: 'doc-6', content: text }];
    const chunks = chunker.chunk(docs);
    for (const chunk of chunks) {
      expect(text.slice(chunk.startChar, chunk.endChar)).toBe(chunk.content);
    }
  });

  it('skips empty documents', () => {
    const chunker = new MarkdownChunker();
    const docs = [{ content: '' }, { content: '# Hi\nThere' }];
    const chunks = chunker.chunk(docs);
    expect(chunks.length).toBe(1);
    expect(chunks[0].content).toBe('There');
  });

  it('treats no-heading document as single section', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [{ id: 'doc-7', content: 'Just some text.' }];
    const chunks = chunker.chunk(docs);
    expect(chunks.length).toBe(1);
    expect(chunks[0].content).toBe('Just some text.');
    expect(chunks[0].metadata).not.toHaveProperty('heading');
    expect(chunks[0].metadata).not.toHaveProperty('headingLevel');
  });

  it('assigns deterministic chunk ids', () => {
    const chunker = new MarkdownChunker();
    const docs = [{ id: 'doc-8', content: '# A\nHello' }];
    const chunks1 = chunker.chunk(docs);
    const chunks2 = chunker.chunk(docs);
    expect(chunks1[0].id).toBe(chunks2[0].id);
  });

  it('preserves user metadata and gives reserved keys precedence', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [
      {
        id: 'doc-9',
        content: '# Title\nBody',
        metadata: { source: 'web', heading: 'UserHeading' },
      },
    ];
    const chunks = chunker.chunk(docs);
    expect(chunks[0].metadata).toMatchObject({
      source: 'web',
      heading: 'Title',
      headingLevel: 1,
    });
  });

  it('assigns sequential chunkIndex per document', () => {
    const chunker = new MarkdownChunker({ chunkSize: 10, overlap: 0 });
    const docs = [
      {
        id: 'doc-10',
        content: '# A\n' + 'x'.repeat(30) + '\n# B\n' + 'y'.repeat(30),
      },
    ];
    const chunks = chunker.chunk(docs);
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].chunkIndex).toBe(i);
    }
  });

  it('emits heading-only chunk when section has no body', () => {
    const chunker = new MarkdownChunker({ chunkSize: 1000, overlap: 0 });
    const docs = [{ id: 'doc-11', content: '# Empty\n# Next\nBody' }];
    const chunks = chunker.chunk(docs);
    const emptyChunk = chunks.find((c) => c.metadata?.heading === 'Empty');
    expect(emptyChunk).toBeDefined();
    expect(emptyChunk!.content).toBe('');
  });
});
