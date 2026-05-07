# RAG SDK

A modular, type-safe SDK for building Retrieval-Augmented Generation (RAG) pipelines in TypeScript.

## Packages

| Package | Description |
|---------|-------------|
| `@rag-sdk/core` | Core types, ports, and pipeline orchestration |
| `@rag-sdk/store-memory` | In-memory vector store reference implementation |
| `@rag-sdk/providers` | Embedding provider adapters (OpenAI, etc.) |

## Installation

```bash
pnpm add @rag-sdk/core @rag-sdk/store-memory @rag-sdk/providers
```

## Quickstart

```ts
import { rag } from '@rag-sdk/core';
import { createMemoryStore } from '@rag-sdk/store-memory';
import { createOpenAI } from '@rag-sdk/providers';

const sdk = rag({
  provider: createOpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  store: createMemoryStore({ dimensions: 1536 }),
});

await sdk.ingest([{ content: 'Hello world', metadata: { source: 'test' } }]);
const result = await sdk.query('hello');
console.log(result.results);
```
