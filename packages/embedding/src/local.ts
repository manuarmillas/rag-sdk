import type { EmbeddingProvider } from '@rag-sdk/core';
import { ProviderError } from '@rag-sdk/core';
import { requirePeer } from './runtime.js';

export interface LocalEmbeddingConfig {
  model?: string;
  dimensions?: number;
  batchSize?: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyPipeline = (inputs: string[]) => Promise<unknown>;

const pipelines = new Map<string, Promise<AnyPipeline>>();

export function __resetPipelines(): void {
  pipelines.clear();
}

async function getPipeline(task: string, model: string): Promise<AnyPipeline> {
  const key = `${task}:${model}`;
  if (!pipelines.has(key)) {
    pipelines.set(
      key,
      (async () => {
        await requirePeer(
          '@huggingface/transformers',
          'Package "@huggingface/transformers" is required for local embeddings',
        );
        const { pipeline } = await import('@huggingface/transformers');
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return pipeline(task as any, model) as unknown as AnyPipeline;
      })(),
    );
  }
  return pipelines.get(key)!;
}

function tensorToArray(data: unknown): number[] {
  // Handle different output formats from transformers.js
  if (Array.isArray(data)) {
    return data as number[];
  }
  // If it's a tensor-like object with data or tolist method
  if (data && typeof data === 'object') {
    const obj = data as Record<string, unknown>;
    if (Array.isArray(obj.data)) {
      return obj.data as number[];
    }
    if (typeof obj.tolist === 'function') {
      const list = (obj as { tolist: () => unknown }).tolist();
      return list as number[];
    }
    // Float32Array or similar typed array
    if (obj.ort_tensor || typeof (obj as { cpu?: unknown }).cpu === 'function') {
      // ONNX tensor — try to extract data
      const tensor = obj as { cpu?: () => unknown; data?: unknown };
      const cpuData = typeof tensor.cpu === 'function' ? tensor.cpu() : obj;
      const raw = (cpuData as { data?: unknown }).data ?? (cpuData as { ort_tensor?: { cpu?: () => unknown } }).ort_tensor;
      if (raw && typeof (raw as { cpu?: () => unknown }).cpu === 'function') {
        const inner = (raw as { cpu: () => unknown }).cpu();
        const innerData = (inner as { data?: unknown }).data;
        if (Array.isArray(innerData)) return innerData as number[];
        if (innerData && typeof innerData === 'object' && 'data' in innerData) {
          const dataObj = innerData as { data: ArrayLike<number> };
          return Array.from(dataObj.data);
        }
      }
    }
  }
  throw new Error('Unexpected pipeline output format');
}

/**
 * Mean pools a 2D array [seq_len, hidden_dim] → [hidden_dim].
 * For MiniLM and similar models, this produces the sentence embedding.
 */
function poolEmbedding(tokens: number[][]): number[] {
  if (tokens.length === 0) return [];
  const dim = tokens[0].length;
  const pooled = new Array<number>(dim).fill(0);
  for (const token of tokens) {
    for (let i = 0; i < dim; i++) {
      pooled[i] += token[i];
    }
  }
  for (let i = 0; i < dim; i++) {
    pooled[i] /= tokens.length;
  }
  // L2 normalize
  const norm = Math.sqrt(pooled.reduce((sum, v) => sum + v * v, 0));
  if (norm > 0) {
    for (let i = 0; i < dim; i++) {
      pooled[i] /= norm;
    }
  }
  return pooled;
}

export function createLocalEmbedding(
  config?: LocalEmbeddingConfig,
): EmbeddingProvider {
  const modelId = config?.model ?? 'Xenova/all-MiniLM-L6-v2';
  const dimensions = config?.dimensions ?? 384;
  const batchSize = config?.batchSize ?? 32;

  return {
    id: 'local',
    modelId,
    dimensions,

    async embed(text: string): Promise<number[]> {
      try {
        const pipe = await getPipeline('feature-extraction', modelId);
        const output = await pipe([text]);
        return extractEmbedding(output, 0);
      } catch (err) {
        throw new ProviderError('local-embedding', 'embed', err);
      }
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      try {
        const pipe = await getPipeline('feature-extraction', modelId);
        const results: number[][] = [];

        for (let i = 0; i < texts.length; i += batchSize) {
          const batch = texts.slice(i, i + batchSize);
          const output = await pipe(batch);
          for (let j = 0; j < batch.length; j++) {
            results.push(extractEmbedding(output, j));
          }
        }

        return results;
      } catch (err) {
        throw new ProviderError('local-embedding', 'embedBatch', err);
      }
    },
  };
}

function extractEmbedding(output: unknown, index: number): number[] {
  // For feature-extraction with batch input, the pipeline may return:
  // - A tensor-like object with .tolist() → 2D/3D array (index selects the item)
  // - An array of tensor-like objects (one per input)
  if (Array.isArray(output)) {
    const item = output[index] as Record<string, unknown>;
    return tensorToArray(item);
  }
  // Single tensor output → try tolist() to get the full array
  const tensor = tensorToArray(output);
  // The top-level could be 1D (single), 2D [batch, hidden], or 3D [batch, seq, hidden]
  const first = tensor[0];
  if (typeof first === 'number') {
    // 1D — already a single embedding (flat)
    return tensor;
  }
  // Get the item for our batch index
  const item = tensor[index];
  if (!Array.isArray(item)) {
    return (item !== undefined ? [item as unknown as number] : []) as number[];
  }
  // Check if we have 3D [batch, seq, hidden] or 2D [batch, hidden]
  if (item.length > 0 && Array.isArray(item[0])) {
    // 3D — mean pool across sequence dimension
    return poolEmbedding(item as number[][]);
  }
  // 2D — already a single vector per item
  return item as number[];
}
