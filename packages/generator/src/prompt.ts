import type { Metadata } from '@rag-sdk/core';
import type { GenerateRequest } from '@rag-sdk/core';

export function buildPrompt<M extends Metadata = Metadata>(
  request: GenerateRequest<M>,
): string {
  const { query, context, systemPrompt } = request;

  const parts: string[] = [];

  if (systemPrompt) {
    parts.push(systemPrompt);
    parts.push('');
  }

  if (context.length > 0) {
    parts.push('Context:');
    for (const result of context) {
      const metaEntries = Object.entries(result.metadata ?? {})
        .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
        .join(', ');
      const metaLine = metaEntries ? ` [${metaEntries}]` : '';
      parts.push(`  [${result.id}]${metaLine} ${result.content}`);
    }
    parts.push('');
  }

  parts.push(`Query: ${query}`);
  parts.push('');
  parts.push('Answer:');

  return parts.join('\n');
}
