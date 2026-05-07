export class RagSdkError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = new.target.name;
  }
}

export class ConfigurationError extends RagSdkError {}
export class ValidationError extends RagSdkError {}

export class ProviderError extends RagSdkError {
  constructor(
    providerId: string,
    operation: 'embed' | 'embedBatch',
    cause: unknown,
  ) {
    super('PROVIDER_ERROR', `Provider ${providerId} failed during ${operation}`, cause);
  }
}

export class StoreError extends RagSdkError {}
export class ChunkingError extends RagSdkError {}
export class DimensionMismatchError extends RagSdkError {}
export class BatchError extends RagSdkError {}
