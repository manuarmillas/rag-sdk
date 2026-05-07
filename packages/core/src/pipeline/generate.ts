import type { Metadata } from '../types/document.js';
import type { Generator, GenerateOptions, GenerationResult } from '../types/generator.js';
import type { GeneratePipelineOptions } from '../types/config.js';
import { ConfigurationError, ProviderError } from '../errors.js';
import { queryPipeline, type QueryDeps } from './query.js';

export interface GenerateDeps<M extends Metadata = Metadata> extends QueryDeps<M> {
  generator?: Generator<M>;
}

export async function generatePipeline<M extends Metadata>(
  text: string,
  options: GeneratePipelineOptions | undefined,
  deps: GenerateDeps<M>,
): Promise<GenerationResult<M>> {
  if (!deps.generator) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Generator is required for generate()',
    );
  }

  const queryResult = await queryPipeline(text, options, deps);

  const generateOptions: GenerateOptions | undefined = options?.generate;

  try {
    return await deps.generator.generate(
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
    throw new ProviderError(deps.generator.id, 'generate', err);
  }
}
