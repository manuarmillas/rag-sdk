import { describe, it, expect } from 'vitest';
import {
  RagSdkError,
  ConfigurationError,
  ValidationError,
  ProviderError,
  StoreError,
  ChunkingError,
  DimensionMismatchError,
  BatchError,
} from '../errors.js';

describe('Error classes', () => {
  it('RagSdkError has code, message, and cause', () => {
    const cause = new Error('root');
    const err = new RagSdkError('TEST', 'message', cause);
    expect(err.code).toBe('TEST');
    expect(err.message).toBe('message');
    expect(err.cause).toBe(cause);
    expect(err.name).toBe('RagSdkError');
  });

  it('ConfigurationError extends RagSdkError', () => {
    const err = new ConfigurationError('CONFIG', 'bad config');
    expect(err).toBeInstanceOf(RagSdkError);
    expect(err.name).toBe('ConfigurationError');
  });

  it('ValidationError extends RagSdkError', () => {
    const err = new ValidationError('VALIDATION', 'bad input');
    expect(err).toBeInstanceOf(RagSdkError);
    expect(err.name).toBe('ValidationError');
  });

  it('ProviderError wraps provider id and operation', () => {
    const cause = new Error('network');
    const err = new ProviderError('openai', 'embed', cause);
    expect(err.code).toBe('PROVIDER_ERROR');
    expect(err.message).toContain('openai');
    expect(err.message).toContain('embed');
    expect(err.cause).toBe(cause);
    expect(err.name).toBe('ProviderError');
  });

  it('StoreError extends RagSdkError', () => {
    const err = new StoreError('STORE', 'store failed');
    expect(err).toBeInstanceOf(RagSdkError);
    expect(err.name).toBe('StoreError');
  });

  it('ChunkingError extends RagSdkError', () => {
    const err = new ChunkingError('CHUNK', 'chunk failed');
    expect(err).toBeInstanceOf(RagSdkError);
    expect(err.name).toBe('ChunkingError');
  });

  it('DimensionMismatchError extends RagSdkError', () => {
    const err = new DimensionMismatchError('DIM', 'dims wrong');
    expect(err).toBeInstanceOf(RagSdkError);
    expect(err.name).toBe('DimensionMismatchError');
  });

  it('BatchError extends RagSdkError', () => {
    const err = new BatchError('BATCH', 'count wrong');
    expect(err).toBeInstanceOf(RagSdkError);
    expect(err.name).toBe('BatchError');
  });
});
