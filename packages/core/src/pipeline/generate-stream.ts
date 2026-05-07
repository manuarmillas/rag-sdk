import type { Metadata } from '../types/document.js';
import type { Generator } from '../types/generator.js';
import type { GeneratePipelineOptions } from '../types/config.js';
import { ConfigurationError, ProviderError } from '../errors.js';
import { queryPipeline, type QueryDeps } from './query.js';

export interface GenerateDeps<M extends Metadata = Metadata> extends QueryDeps<M> {
  generator?: Generator<M>;
}

export async function* generateStreamPipeline<M extends Metadata>(
  text: string,
  options: GeneratePipelineOptions | undefined,
  deps: GenerateDeps<M>,
): AsyncGenerator<string, void, undefined> {
  if (!deps.generator) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Generator is required for generateStream()',
    );
  }

  if (!deps.generator.generateStream) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Generator does not support streaming',
    );
  }

  const queryResult = await queryPipeline(text, options, deps);

  const generateOptions = options?.generate;

  try {
    yield* deps.generator.generateStream(
      {
        query: text,
        context: queryResult.results,
        systemPrompt: generateOptions?.systemPrompt,
      },
      generateOptions,
    );
  } catch (err) {
    if (err instanceof ProviderError) {
      throw err;
    }
    throw new ProviderError(deps.generator.id, 'generateStream', err);
  }
}
