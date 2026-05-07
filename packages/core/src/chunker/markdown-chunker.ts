import { createHash } from 'crypto';
import type { Metadata, Document, Chunk } from '../types/document.js';
import type { Chunker } from './types.js';
import type { ChunkOptions } from '../types/config.js';

function chunkId(documentId: string, chunkIndex: number, content: string): string {
  return createHash('sha256')
    .update(`${documentId}:${chunkIndex}:${content}`)
    .digest('hex')
    .slice(0, 16);
}

interface Section {
  heading: string;
  level: number;
  headingPath: string[];
  content: string;
  startChar: number;
}

export class MarkdownChunker<M extends Metadata = Metadata> implements Chunker<M> {
  readonly chunkSize: number;
  readonly overlap: number;

  constructor(options?: ChunkOptions) {
    this.chunkSize = options?.chunkSize ?? 1000;
    this.overlap = options?.overlap ?? 200;
  }

  chunk(documents: Document<M>[], options?: ChunkOptions): Chunk<M>[] {
    const chunkSize = options?.chunkSize ?? this.chunkSize;
    const overlap = options?.overlap ?? this.overlap;

    const chunks: Chunk<M>[] = [];

    for (const doc of documents) {
      const documentId = doc.id ?? chunkId('fallback', 0, doc.content ?? '');

      if (!doc.content) {
        continue;
      }

      const sections = this.parseSections(doc.content);
      let chunkIndex = 0;

      for (const section of sections) {
        const sectionChunks = this.chunkSection(
          section,
          documentId,
          chunkIndex,
          chunkSize,
          overlap,
          doc.metadata,
        );
        chunks.push(...sectionChunks);
        chunkIndex += sectionChunks.length;
      }
    }

    return chunks;
  }

  private parseSections(content: string): Section[] {
    const lines = content.split('\n');
    const sections: Section[] = [];

    const headingStack: { heading: string; level: number }[] = [];
    let currentHeading = '';
    let currentLevel = 0;
    let currentContent: string[] = [];
    let sectionStart = 0;
    let lineStartOffset = 0;

    const flushSection = (_endOffset: number) => {
      const sectionContent = currentContent.join('\n');
      if (currentHeading || sectionContent.length > 0) {
        sections.push({
          heading: currentHeading,
          level: currentLevel,
          headingPath: headingStack.map((h) => h.heading),
          content: sectionContent,
          startChar: sectionStart,
        });
      }
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const match = line.match(/^(#{1,6})\s+(.*)$/);

      if (match) {
        const level = match[1].length;
        const heading = match[2].trim();
        const headingStart = lineStartOffset;

        flushSection(headingStart);

        while (
          headingStack.length > 0 &&
          headingStack[headingStack.length - 1].level >= level
        ) {
          headingStack.pop();
        }
        headingStack.push({ heading, level });

        currentHeading = heading;
        currentLevel = level;
        currentContent = [];
        sectionStart = headingStart + line.length + 1;
      } else {
        currentContent.push(line);
      }

      lineStartOffset += line.length + 1;
    }

    flushSection(content.length);

    if (sections.length === 0) {
      sections.push({
        heading: '',
        level: 0,
        headingPath: [],
        content,
        startChar: 0,
      });
    }

    return sections;
  }

  private chunkSection(
    section: Section,
    documentId: string,
    startIndex: number,
    chunkSize: number,
    overlap: number,
    userMetadata?: M,
  ): Chunk<M>[] {
    const { content, heading, level, headingPath, startChar } = section;

    if (!content || content.length === 0) {
      if (!heading) {
        return [];
      }
      // Heading with no content — emit a single empty-content chunk
      return [
        {
          id: chunkId(documentId, startIndex, ''),
          documentId,
          chunkIndex: startIndex,
          startChar,
          endChar: startChar,
          content: '',
          metadata: {
            ...(userMetadata ?? {}),
            heading,
            headingLevel: level,
            headingPath,
          } as unknown as M,
        },
      ];
    }

    if (content.length <= chunkSize) {
      return [
        {
          id: chunkId(documentId, startIndex, content),
          documentId,
          chunkIndex: startIndex,
          startChar,
          endChar: startChar + content.length,
          content,
          metadata: {
            ...(userMetadata ?? {}),
            ...(heading ? { heading, headingLevel: level, headingPath } : {}),
          } as unknown as M,
        },
      ];
    }

    const texts = this.splitWithOverlap(content, chunkSize, overlap);
    const sectionChunks: Chunk<M>[] = [];
    let cursor = 0;

    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const idx = content.indexOf(text, cursor);
      const absStart = idx >= 0 ? startChar + idx : startChar + cursor;
      const absEnd = absStart + text.length;
      cursor = Math.max(0, idx + text.length - overlap);

      sectionChunks.push({
        id: chunkId(documentId, startIndex + i, text),
        documentId,
        chunkIndex: startIndex + i,
        startChar: absStart,
        endChar: absEnd,
        content: text,
        metadata: {
          ...(userMetadata ?? {}),
          ...(heading ? { heading, headingLevel: level, headingPath } : {}),
        } as unknown as M,
      });
    }

    return sectionChunks;
  }

  private splitWithOverlap(text: string, chunkSize: number, overlap: number): string[] {
    const separators = ['\n\n', '\n', ' ', ''];
    const pieces = this.splitText(text, chunkSize, separators);
    return this.mergePieces(pieces, chunkSize, overlap);
  }

  private splitText(text: string, chunkSize: number, separators: string[]): string[] {
    const separator = separators[0] ?? '';

    if (separator === '') {
      return text.split('');
    }

    const parts = text.split(separator);
    const result: string[] = [];

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];

      if (part.length <= chunkSize) {
        if (part) result.push(part);
      } else {
        const subParts = this.splitText(part, chunkSize, separators.slice(1));
        result.push(...subParts);
      }

      if (i < parts.length - 1 && separator) {
        result.push(separator);
      }
    }

    return result;
  }

  private mergePieces(pieces: string[], chunkSize: number, overlap: number): string[] {
    if (pieces.length === 0) return [];
    if (pieces.length === 1) return pieces;

    const chunks: string[] = [];
    let current = pieces[0];

    for (let i = 1; i < pieces.length; i++) {
      const piece = pieces[i];
      const candidate = current + piece;

      if (candidate.length > chunkSize && current.length > 0) {
        chunks.push(current);
        const overlapStart = Math.max(0, current.length - overlap);
        current = current.slice(overlapStart) + piece;
      } else {
        current = candidate;
      }
    }

    if (current.length > 0) {
      chunks.push(current);
    }

    return chunks;
  }
}
