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

export class RecursiveCharacterTextSplitter<M extends Metadata = Metadata>
  implements Chunker<M>
{
  readonly chunkSize: number;
  readonly overlap: number;
  readonly separators: string[];

  constructor(options?: ChunkOptions) {
    this.chunkSize = options?.chunkSize ?? 1000;
    this.overlap = options?.overlap ?? 200;
    this.separators = options?.separators ?? ['\n\n', '\n', ' ', ''];
  }

  chunk(documents: Document<M>[], options?: ChunkOptions): Chunk<M>[] {
    const chunkSize = options?.chunkSize ?? this.chunkSize;
    const overlap = options?.overlap ?? this.overlap;
    const separators = options?.separators ?? this.separators;

    const chunks: Chunk<M>[] = [];

    for (const doc of documents) {
      const documentId = doc.id ?? chunkId('fallback', 0, doc.content ?? '');

      if (!doc.content) {
        continue;
      }

      const pieces = this.splitText(doc.content, chunkSize, separators);
      const chunkTexts = this.mergePieces(pieces, chunkSize, overlap);

      let cursor = 0;
      for (let i = 0; i < chunkTexts.length; i++) {
        const text = chunkTexts[i];
        const startChar = doc.content.indexOf(text, cursor);
        const endChar = startChar + text.length;
        cursor = Math.max(0, endChar - overlap);

        chunks.push({
          id: chunkId(documentId, i, text),
          documentId,
          chunkIndex: i,
          startChar,
          endChar,
          content: text,
          metadata: { ...(doc.metadata ?? {}) } as M,
        });
      }
    }

    return chunks;
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
