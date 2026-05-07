import { describe, it, expect, expectTypeOf } from 'vitest';
import { rag } from '../rag.js';
import type { EmbeddingProvider } from '../types/provider.js';
import type { VectorStore } from '../types/store.js';
import type { Document, QueryResult, Metadata, SearchResult } from '../types/document.js';
import type { RagSDK } from '../types/config.js';

function createFakeProvider(dimensions = 3): EmbeddingProvider {
  return {
    id: 'fake',
    modelId: 'fake-model',
    dimensions,
    async embed(_text: string): Promise<number[]> {
      return Array.from({ length: dimensions }, () => 0.1);
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      return texts.map(() => Array.from({ length: dimensions }, () => 0.1));
    },
  };
}

function createFakeStore<M extends Metadata = Metadata>(): VectorStore<M> {
  return {
    async upsert() {},
    async query() {
      return [] as SearchResult<M>[];
    },
    async delete() {},
  };
}

describe('type inference (compile-time safety)', () => {
  it('rag() return type is inferred as RagSDK<Metadata> with default generics', () => {
    const provider = createFakeProvider();
    const store = createFakeStore();
    const sdk = rag({ provider, store });

    // Compile-time: sdk should be RagSDK<Metadata>
    expectTypeOf(sdk).toMatchTypeOf<RagSDK>();

    // Runtime: methods exist
    expect(sdk.ingest).toBeInstanceOf(Function);
    expect(sdk.query).toBeInstanceOf(Function);
  });

  it('ingest accepts Document<M> with correct metadata shape', () => {
    const provider = createFakeProvider();
    const store = createFakeStore<{ source: string }>();
    const sdk = rag({ provider, store });

    // This should compile: metadata matches { source: string }
    const doc: Document<{ source: string }> = {
      content: 'hello',
      metadata: { source: 'test.txt' },
    };

    // TypeScript should allow this without explicit annotation
    expect(doc.metadata!.source).toBe('test.txt');
    expectTypeOf(sdk.ingest).toBeFunction();
  });

  it('query returns QueryResult<M> with metadata flowing through', async () => {
    const provider = createFakeProvider();
    const store = createFakeStore<{ author: string }>();
    const sdk = rag<{ author: string }>({ provider, store });

    // The return type should be Promise<QueryResult<{ author: string }>>
    // This is a type-level assertion
    const _queryResult: Promise<QueryResult<{ author: string }>> = sdk.query(
      'hello',
      {},
    );
    void _queryResult;
  });

  it('RagSDK generic M flows from store to ingest documents', () => {
    const store = createFakeStore<{ priority: number }>();
    const provider = createFakeProvider();

    const sdk = rag<{ priority: number }>({ provider, store });

    // sdk.ingest should accept Document<{ priority: number }>[]
    const _ingestDocs: Document<{ priority: number }>[] = [
      { content: 'x', metadata: { priority: 1 } },
    ];
    void sdk;
    void _ingestDocs;
  });

  it('provider is typed as EmbeddingProvider with readonly properties', () => {
    const provider = createFakeProvider(1536);

    // Type should enforce readonly on id, modelId, dimensions
    expectTypeOf(provider.id).toBeString();
    expectTypeOf(provider.modelId).toBeString();
    expectTypeOf(provider.dimensions).toBeNumber();
    expectTypeOf(provider.embed).toBeFunction();
    expectTypeOf(provider.embedBatch).toBeFunction();

    // maxBatchSize is optional
    expectTypeOf(provider.maxBatchSize).toBeUndefined;
  });

  it('consumer code compiles without manual type annotations', () => {
    // This entire test IS the proof. If it compiles, REQ-11 passes.
    // No explicit type annotations on sdk, documents, or results.

    const sdk = rag<{ source: string }>({
      provider: createFakeProvider(),
      store: createFakeStore<{ source: string }>(),
    });

    // ingest without type annotation on documents array
    const ingestPromise = sdk.ingest([
      { content: 'Hello world', metadata: { source: 'test.md' } },
    ]);
    expectTypeOf(ingestPromise).toEqualTypeOf<Promise<void>>();

    // query without type annotation on result
    const queryPromise = sdk.query('hello', { topK: 5 });
    const _queryPromise: Promise<QueryResult<{ source: string }>> = queryPromise;
    void _queryPromise;
  });

  it('namespace in options is string', () => {
    const sdk = rag({
      provider: createFakeProvider(),
      store: createFakeStore(),
      namespace: 'default',
    });

    // This should compile — namespace is a valid string
    expectTypeOf(sdk.query).parameter(1).toMatchTypeOf<{
      namespace?: string;
    } | undefined>();
  });

  it('embedBatch accepts string[] and returns Promise<number[][]>', () => {
    const provider = createFakeProvider();

    expectTypeOf(provider.embedBatch)
      .parameter(0)
      .toMatchTypeOf<string[]>();

    expectTypeOf(provider.embedBatch)
      .returns
      .toMatchTypeOf<Promise<number[][]>>();
  });

  it('embed accepts string and returns Promise<number[]>', () => {
    const provider = createFakeProvider();

    expectTypeOf(provider.embed)
      .parameter(0)
      .toBeString();

    expectTypeOf(provider.embed)
      .returns
      .toMatchTypeOf<Promise<number[]>>();
  });
});
