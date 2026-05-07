import { createHash } from 'crypto';
import type { Metadata, Document, Chunk } from '@rag-sdk/core';
import type { Chunker } from '@rag-sdk/core';
import type { ChunkOptions } from '@rag-sdk/core';
import type { EmbeddingProvider } from '@rag-sdk/core';
import { cosineSimilarity } from './similarity.js';

function chunkId(documentId: string, chunkIndex: number, content: string): string {
  return createHash('sha256')
    .update(`${documentId}:${chunkIndex}:${content}`)
    .digest('hex')
    .slice(0, 16);
}

interface Token {
  text: string;
  start: number;
  end: number;
}

export class SemanticChunker<M extends Metadata = Metadata> implements Chunker<M> {
  readonly chunkSize: number;
  readonly overlap: number;
  readonly threshold: number;

  constructor(
    private readonly provider: EmbeddingProvider,
    options?: ChunkOptions & { threshold?: number },
  ) {
    this.chunkSize = options?.chunkSize ?? 1000;
    this.overlap = options?.overlap ?? 200;
    this.threshold = options?.threshold ?? 0.5;
  }

  async chunk(
    documents: Document<M>[],
    options?: ChunkOptions,
  ): Promise<Chunk<M>[]> {
    const chunkSize = options?.chunkSize ?? this.chunkSize;
    const overlap = options?.overlap ?? this.overlap;

    const chunks: Chunk<M>[] = [];

    for (const doc of documents) {
      const documentId = doc.id ?? chunkId('fallback', 0, doc.content ?? '');

      if (!doc.content) {
        continue;
      }

      const tokens = this.tokenize(doc.content, chunkSize);
      if (tokens.length === 0) {
        continue;
      }

      const embeddings = await this.provider.embedBatch(
        tokens.map((t) => t.text),
      );

      const breakpoints = this.findBreakpoints(embeddings, this.threshold);
      const groups = this.buildGroups(tokens, breakpoints, chunkSize);

      for (let i = 0; i < groups.length; i++) {
        const group = groups[i];
        let content = group.map((t) => t.text).join(' ');
        let startChar = group[0].start;
        const endChar = group[group.length - 1].end;

        // Apply overlap from previous chunk
        if (overlap > 0 && i > 0) {
          const prevGroup = groups[i - 1];
          const prevText = prevGroup.map((t) => t.text).join(' ');
          const overlapText = prevText.slice(-overlap);
          const overlapStart = doc.content.lastIndexOf(
            overlapText,
            startChar + overlapText.length,
          );
          if (overlapStart >= 0 && overlapStart < startChar) {
            startChar = overlapStart;
            content = doc.content.slice(startChar, endChar);
          }
        }

        chunks.push({
          id: chunkId(documentId, i, content),
          documentId,
          chunkIndex: i,
          startChar,
          endChar,
          content,
          metadata: { ...(doc.metadata ?? {}) } as M,
        });
      }
    }

    return chunks;
  }

  private tokenize(text: string, chunkSize: number): Token[] {
    const paragraphs = splitByDelimiter(text, '\n\n');
    const tokens: Token[] = [];

    for (const para of paragraphs) {
      const sentences = splitSentences(text, para.start, para.end);
      for (const s of sentences) {
        if (s.text.length <= chunkSize) {
          tokens.push(s);
        } else {
          tokens.push(...splitByWords(text, s.start, s.end, chunkSize));
        }
      }
    }

    return mergeTinyTokens(tokens, 10);
  }

  private findBreakpoints(embeddings: number[][], threshold: number): number[] {
    const breakpoints: number[] = [];
    for (let i = 1; i < embeddings.length; i++) {
      const sim = cosineSimilarity(embeddings[i - 1], embeddings[i]);
      if (sim < threshold) {
        breakpoints.push(i);
      }
    }
    return breakpoints;
  }

  private buildGroups(
    tokens: Token[],
    breakpoints: number[],
    chunkSize: number,
  ): Token[][] {
    const groups: Token[][] = [];
    let current: Token[] = [];
    let currentLen = 0;

    const flush = () => {
      if (current.length > 0) {
        groups.push(current);
        current = [];
        currentLen = 0;
      }
    };

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const atBreakpoint = breakpoints.includes(i);

      if (atBreakpoint && current.length > 0) {
        flush();
      }

      if (currentLen + token.text.length > chunkSize && current.length > 0) {
        flush();
      }

      current.push(token);
      currentLen += token.text.length + 1;
    }

    flush();

    return groups;
  }
}

function splitByDelimiter(text: string, delimiter: string): Token[] {
  const result: Token[] = [];
  let cursor = 0;

  while (cursor < text.length) {
    const idx = text.indexOf(delimiter, cursor);
    const end = idx === -1 ? text.length : idx;
    const part = text.slice(cursor, end);
    if (part) {
      result.push({ text: part, start: cursor, end });
    }
    cursor = idx === -1 ? text.length : end + delimiter.length;
  }

  return result;
}

function splitSentences(
  fullText: string,
  paraStart: number,
  paraEnd: number,
): Token[] {
  const text = fullText.slice(paraStart, paraEnd);
  const sentences: Token[] = [];
  const regex = /([.!?]+\s+|\n+)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const sentenceText = text.slice(lastIndex, match.index + match[0].length);
    if (sentenceText.trim()) {
      sentences.push({
        text: sentenceText.trimEnd(),
        start: paraStart + lastIndex,
        end: paraStart + lastIndex + sentenceText.length,
      });
    }
    lastIndex = match.index + match[0].length;
  }

  const remainder = text.slice(lastIndex);
  if (remainder.trim()) {
    sentences.push({
      text: remainder.trimEnd(),
      start: paraStart + lastIndex,
      end: paraStart + paraEnd,
    });
  }

  if (sentences.length === 0 && text.trim()) {
    sentences.push({
      text: text.trimEnd(),
      start: paraStart,
      end: paraEnd,
    });
  }

  return sentences;
}

function splitByWords(
  fullText: string,
  start: number,
  end: number,
  chunkSize: number,
): Token[] {
  const text = fullText.slice(start, end);
  const tokens: Token[] = [];
  let cursor = 0;

  while (cursor < text.length) {
    let chunkEnd = Math.min(cursor + chunkSize, text.length);
    if (chunkEnd < text.length) {
      const spaceIdx = text.lastIndexOf(' ', chunkEnd);
      if (spaceIdx > cursor) {
        chunkEnd = spaceIdx;
      }
    }
    const part = text.slice(cursor, chunkEnd).trimEnd();
    if (part) {
      tokens.push({ text: part, start: start + cursor, end: start + chunkEnd });
    }
    cursor = chunkEnd + 1;
  }

  return tokens;
}

function mergeTinyTokens(tokens: Token[], minLength: number): Token[] {
  if (tokens.length === 0) return [];
  const merged: Token[] = [{ ...tokens[0] }];
  let canMerge = true;

  for (let i = 1; i < tokens.length; i++) {
    const last = merged[merged.length - 1];
    const curr = tokens[i];

    if (canMerge && (last.text.length < minLength || curr.text.length < minLength)) {
      last.text += ' ' + curr.text;
      last.end = curr.end;
      canMerge = false;
    } else {
      merged.push({ ...curr });
      canMerge = true;
    }
  }

  return merged;
}
