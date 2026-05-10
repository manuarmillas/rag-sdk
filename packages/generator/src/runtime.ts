import { ConfigurationError } from '@rag-sdk/core';

export async function requirePeer(
  pkg: string,
  installHint: string,
): Promise<void> {
  try {
    await import(pkg);
  } catch {
    throw new ConfigurationError(
      'CONFIGURATION_ERROR',
      `@rag-sdk/generator: ${installHint}. Run: npm install ${pkg}`,
    );
  }
}
