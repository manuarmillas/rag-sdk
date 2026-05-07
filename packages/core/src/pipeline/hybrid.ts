import type { Metadata, SearchResult } from '../types/document.js';
import type { HybridOptions } from '../types/searcher.js';

/**
 * Reciprocal Rank Fusion (RRF) combines ranked lists from vector and keyword
 * search into a single fused ranking.
 *
 * Formula:
 *   score = vectorWeight/(rrfK + vectorRank) + keywordWeight/(rrfK + keywordRank)
 *
 * Ranks are 1-indexed. When a result appears in only one list, the other term
 * is omitted. Deduplication preserves content/metadata from the vector result
 * when both lists contain the same document; otherwise the keyword result is
 * used.
 */
export function rrfFusion<M extends Metadata>(
  vectorResults: SearchResult<M>[],
  keywordResults: SearchResult<M>[],
  options?: HybridOptions,
): SearchResult<M>[] {
  const vectorWeight = options?.vectorWeight ?? 1;
  const keywordWeight = options?.keywordWeight ?? 1;
  const rrfK = options?.rrfK ?? 60;

  const fusedScores = new Map<string, number>();
  const resultMap = new Map<string, SearchResult<M>>();

  // Vector results: rank starts at 1
  for (let i = 0; i < vectorResults.length; i++) {
    const result = vectorResults[i];
    const rank = i + 1;
    const score = vectorWeight / (rrfK + rank);

    fusedScores.set(result.id, (fusedScores.get(result.id) ?? 0) + score);
    resultMap.set(result.id, result); // vector result takes priority
  }

  // Keyword results: rank starts at 1
  for (let i = 0; i < keywordResults.length; i++) {
    const result = keywordResults[i];
    const rank = i + 1;
    const score = keywordWeight / (rrfK + rank);

    fusedScores.set(result.id, (fusedScores.get(result.id) ?? 0) + score);
    if (!resultMap.has(result.id)) {
      resultMap.set(result.id, result);
    }
  }

  // Sort by fused score descending
  const ids = Array.from(fusedScores.keys()).sort(
    (a, b) => fusedScores.get(b)! - fusedScores.get(a)!,
  );

  return ids.map((id) => ({
    ...resultMap.get(id)!,
    score: fusedScores.get(id)!,
  }));
}
