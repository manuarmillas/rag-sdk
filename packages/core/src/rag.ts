import type { Metadata } from './types/document.js';
import type { EmbeddingProvider } from './types/provider.js';
import type { VectorStore } from './types/store.js';
import type { RagConfig, RagSDK, ChunkOptions, GeneratePipelineOptions } from './types/config.js';
import { RecursiveCharacterTextSplitter } from './chunker/recursive-splitter.js';
import { ConfigurationError } from './errors.js';
import { validateChunkOptions } from './validate.js';
import { ingestPipeline } from './pipeline/ingest.js';
import { queryPipeline } from './pipeline/query.js';
import { generatePipeline } from './pipeline/generate.js';

export function rag<
  M extends Metadata = Metadata,
  P extends EmbeddingProvider = EmbeddingProvider,
  S extends VectorStore<M> = VectorStore<M>,
>(config: RagConfig<M, P, S>): RagSDK<M> {
  if (!config.provider) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider is required',
    );
  }
  if (!config.store) {
    throw new ConfigurationError('CONFIGURATION_ERROR', 'Store is required');
  }

  const provider = config.provider;
  const store = config.store;

  if (!provider.id || typeof provider.id !== 'string') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider must have a valid id',
    );
  }
  if (!provider.modelId || typeof provider.modelId !== 'string') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider must have a valid modelId',
    );
  }
  if (
    typeof provider.dimensions !== 'number' ||
    !Number.isInteger(provider.dimensions) ||
    provider.dimensions <= 0
  ) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider dimensions must be a positive integer',
    );
  }
  if (
    provider.maxBatchSize !== undefined &&
    (typeof provider.maxBatchSize !== 'number' ||
      provider.maxBatchSize <= 0 ||
      !Number.isInteger(provider.maxBatchSize))
  ) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider maxBatchSize must be a positive integer',
    );
  }

  if (typeof provider.embed !== 'function') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider must implement embed()',
    );
  }
  if (typeof provider.embedBatch !== 'function') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Provider must implement embedBatch()',
    );
  }

  if (typeof store.upsert !== 'function') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Store must implement upsert()',
    );
  }
  if (typeof store.query !== 'function') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Store must implement query()',
    );
  }
  if (typeof store.delete !== 'function') {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Store must implement delete()',
    );
  }

  const generator = config.generator;
  if (generator !== undefined) {
    if (!generator.id || typeof generator.id !== 'string') {
      throw new ConfigurationError(
        'CONFIGURATION_ERROR',
        'Generator must have a valid id',
      );
    }
    if (!generator.modelId || typeof generator.modelId !== 'string') {
      throw new ConfigurationError(
        'CONFIGURATION_ERROR',
        'Generator must have a valid modelId',
      );
    }
    if (typeof generator.generate !== 'function') {
      throw new ConfigurationError(
        'CONFIGURATION_ERROR',
        'Generator must implement generate()',
      );
    }
  }

  const reranker = config.reranker;
  if (reranker !== undefined) {
    if (!reranker.id || typeof reranker.id !== 'string') {
      throw new ConfigurationError(
        'CONFIGURATION_ERROR',
        'Reranker must have a valid id',
      );
    }
    if (typeof reranker.rerank !== 'function') {
      throw new ConfigurationError(
        'CONFIGURATION_ERROR',
        'Reranker must implement rerank()',
      );
    }
  }

  const chunkOpts: ChunkOptions | undefined = config.chunk;
  validateChunkOptions(chunkOpts);

  const chunker =
    config.chunker ?? new RecursiveCharacterTextSplitter(chunkOpts);

  return {
    async ingest(documents, options) {
      return ingestPipeline(documents, options, {
        provider,
        store,
        chunker,
        defaultNamespace: config.namespace,
        defaultChunkOpts: chunkOpts,
      });
    },

    async query(text, options) {
      return queryPipeline(text, options, {
        provider,
        store,
        reranker,
        defaultNamespace: config.namespace,
      });
    },

    async generate(text, options?: GeneratePipelineOptions) {
      return generatePipeline(text, options, {
        provider,
        store,
        reranker,
        defaultNamespace: config.namespace,
        generator,
      });
    },

    async *generateStream(text, options?: GeneratePipelineOptions) {
      if (!generator) {
        throw new ConfigurationError(
          'CONFIGURATION_ERROR',
          'Generator is required for generateStream()',
        );
      }
      if (!generator.generateStream) {
        throw new ConfigurationError(
          'CONFIGURATION_ERROR',
          'Generator does not support streaming',
        );
      }

      const queryResult = await queryPipeline(text, options, {
        provider,
        store,
        reranker,
        defaultNamespace: config.namespace,
      });

      const generateOptions = options?.generate;

      yield* generator.generateStream(
        {
          query: text,
          context: queryResult.results,
          systemPrompt: generateOptions?.systemPrompt,
        },
        generateOptions,
      );
    },
  };
}
