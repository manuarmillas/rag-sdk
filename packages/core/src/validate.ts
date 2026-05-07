import type { ChunkOptions } from './types/config.js';
import { ConfigurationError } from './errors.js';

export function validateChunkOptions(opts?: ChunkOptions): void {
  if (opts === undefined) {
    return;
  }

  const chunkSize = opts.chunkSize ?? 1000;
  const overlap = opts.overlap ?? 200;

  if (
    typeof chunkSize !== 'number' ||
    !Number.isInteger(chunkSize) ||
    chunkSize <= 0
  ) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'chunkSize must be a positive integer',
    );
  }

  if (typeof overlap !== 'number' || !Number.isInteger(overlap) || overlap < 0) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'overlap must be a non-negative integer',
    );
  }

  if (overlap >= chunkSize) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'Chunk overlap must be smaller than chunk size',
    );
  }

  if (opts.separators !== undefined && opts.separators.length === 0) {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      'separators must not be an empty array',
    );
  }
}
