/**
 * RAG SDK V3 — Full RAG Pipeline Showcase
 * ========================================
 *
 * Construye un RAG completo usando TODOS los componentes de V3:
 *   - Embeddings: Local (CPU), Ollama, Cohere, VoyageAI, OpenAI
 *   - Vector Store: Memoria (sin dep), Qdrant (producción)
 *   - Generación: Local (Ollama), Cohere, OpenAI
 *   - Reranking: Local cross-encoder (CPU), Cohere cloud
 *   - Chunking: Markdown, Semántico
 *   - Búsqueda Híbrida: vector + keyword (RRF fusion)
 *   - Streaming: generación token por token
 *
 * Cada sección es autocontenida — podés correr la que tengas los
 * servicios disponibles. Las que son 100% locales no necesitan nada.
 *
 * Requisitos por proveedor:
 *   Local (reranker + embeddings):  npm install @huggingface/transformers
 *   Ollama (embedding + generator): ollama serve corriendo en localhost:11434
 *   Cohere (embedding + generator): COHERE_API_KEY en .env
 *   VoyageAI (embedding):           VOYAGE_API_KEY en .env
 *   OpenAI (embedding + generator): OPENAI_API_KEY en .env
 *   Qdrant (store):                 qdrant corriendo + npm install @qdrant/js-client-rest
 */

// ─── Imports ────────────────────────────────────────────────────────────────
import { rag, MarkdownChunker, ConfigurationError, DimensionMismatchError } from '@rag-sdk/core';
import {
  createOllamaEmbedding,
  createCohereEmbedding,
  createVoyageEmbedding,
  createLocalEmbedding,
} from '@rag-sdk/embedding';
import { createMemoryStore } from '@rag-sdk/store';
import { createOpenAIGenerator, createOllamaGenerator, createCohereGenerator } from '@rag-sdk/generator';
import { createCohereReranker, createLocalReranker } from '@rag-sdk/reranker';
import { SemanticChunker } from '@rag-sdk/chunker';
import type { Metadata } from '@rag-sdk/core';

// ─── Helpers ────────────────────────────────────────────────────────────────
const divider = () => console.log('\n' + '─'.repeat(60) + '\n');
const section = (title: string) => console.log(`\n📌 ${title}`);
const step = (msg: string) => console.log(`  ⚡ ${msg}`);

/** Simula docs de una empresa — reemplazá con tu propio dataset. */
const DOCUMENTS = [
  {
    id: 'doc-1',
    content: `# Arquitectura de Microservicios
Los microservicios son un estilo arquitectónico que estructura una aplicación
como una colección de servicios independientes. Cada servicio corre en su propio
proceso y se comunica mediante APIs ligeras, típicamente HTTP/REST o gRPC.
Las ventajas incluyen despliegue independiente, escalado granular y diversidad
tecnológica. Las desventajas son la complejidad operativa, latencia de red y
consistencia eventual de datos.`,
  },
  {
    id: 'doc-2',
    content: `# Docker y Contenedores
Docker empaqueta aplicaciones con todas sus dependencias en contenedores aislados.
A diferencia de las máquinas virtuales, los contenedores comparten el kernel del
host, lo que los hace mucho más ligeros. Un Dockerfile define la imagen, y
docker-compose orquesta múltiples contenedores. Kubernetes extiende esto a nivel
de clúster con auto-scaling, service discovery y rolling updates.`,
  },
  {
    id: 'doc-3',
    content: `# TypeScript Avanzado
TypeScript agrega tipado estático a JavaScript. Los genéricos permiten funciones
que trabajan con múltiples tipos manteniendo la seguridad. Los tipos condicionales
(infer, extends) habilitan lógica de tipos en tiempo de compilación. Template
literal types (prefix-\${string}) permiten modelar strings con estructura.
Los mapped types transforman propiedades de objetos automáticamente.`,
  },
  {
    id: 'doc-4',
    content: `# Retrieval-Augmented Generation (RAG)
RAG combina recuperación de documentos con generación de lenguaje. Primero,
los documentos se dividen en chunks, se convierten en embeddings y se almacenan
en una base de datos vectorial. Al recibir una query, se recuperan los chunks
más similares por similitud coseno y se inyectan en el prompt del LLM. Esto
fundamenta las respuestas en datos reales, reduciendo alucinaciones.`,
  },
  {
    id: 'doc-5',
    content: `# Reranking en RAG
El reranking mejora la calidad de los resultados de búsqueda vectorial. Después
de recuperar top-K documentos por similitud, un reranker (cross-encoder) evalúa
cada par (query, documento) y reordena los resultados por relevancia semántica
real. Los modelos cross-encoder como ms-marco-MiniLM analizan query y documento
juntos, capturando relaciones que los embeddings individuales no pueden.`,
  },
];

// ─── 1. RAG 100% Local (sin APIs, sin servidores) ────────────────────────────
async function ragFullyLocal() {
  section('1. RAG 100% Local — CPU only, sin API keys');

  step('Creando SDK con embeddings locales + reranker local + store en memoria...');
  const sdk = rag({
    provider: createLocalEmbedding(), // 384d, 80MB model, CPU
    store: createMemoryStore({ dimensions: 384 }),
    reranker: createLocalReranker(), // cross-encoder, CPU
  });

  step('Ingestando documentos...');
  await sdk.ingest(DOCUMENTS);
  console.log(`     ✅ ${DOCUMENTS.length} documentos indexados`);

  step('Query: "¿Cómo funciona el reranking?"');
  const result = await sdk.query('¿Cómo funciona el reranking?', {
    topK: 5,
    rerank: { topN: 3 },
  });

  for (let i = 0; i < result.results.length; i++) {
    const r = result.results[i];
    console.log(`  ${i + 1}. [${r.score.toFixed(4)}] ${r.content.slice(0, 100)}...`);
  }

  return result;
}

// ─── 2. RAG con Ollama (embeddings + generación local) ─────────────────────
async function ragOllama() {
  section('2. Ollama — embeddings locales + generación con Llama');
  console.log('   Requiere: ollama pull nomic-embed-text && ollama pull llama3.2');

  try {
    const sdk = rag({
      provider: createOllamaEmbedding({ model: 'nomic-embed-text' }),
      store: createMemoryStore({ dimensions: 768 }),
      generator: createOllamaGenerator({ model: 'llama3.2', temperature: 0.1 }),
      reranker: createLocalReranker(),
    });

    step('Ingestando documentos...');
    await sdk.ingest(DOCUMENTS);

    step('Pregunta al LLM: "¿Qué diferencia a los contenedores de las VMs?"');
    const genResult = await sdk.generate('¿Qué diferencia a los contenedores de las VMs?');

    console.log('   📝 Respuesta:');
    console.log(`   ${genResult.answer.slice(0, 300)}...`);
    return genResult;
  } catch (err) {
    console.log('   ⚠️  Ollama no disponible — salteando');
  }
}

// ─── 3. RAG con Cohere (embeddings + generación cloud) ────────────────────
async function ragCohere() {
  section('3. Cohere — embeddings + generación cloud');
  console.log('   Requiere: COHERE_API_KEY en .env');

  const apiKey = process.env.COHERE_API_KEY;
  if (!apiKey) {
    console.log('   ⚠️  COHERE_API_KEY no configurada — salteando');
    return;
  }

  const sdk = rag({
    provider: createCohereEmbedding({ apiKey }),
    store: createMemoryStore({ dimensions: 1024 }),
    generator: createCohereGenerator({ apiKey, temperature: 0.1 }),
  });

  step('Ingestando documentos...');
  await sdk.ingest(DOCUMENTS);

  step('Pregunta: "¿Cuáles son las ventajas de TypeScript?"');
  const result = await sdk.generate('¿Cuáles son las ventajas de TypeScript?');
  console.log(`   📝 ${result.answer.slice(0, 300)}...`);
}

// ─── 4. RAG con OpenAI + VoyageAI (embeddings) + Cohere (reranker) ─────────
async function ragMultiCloud() {
  section('4. Multi-cloud: VoyageAI embeddings + OpenAI generator + Cohere reranker');
  console.log('   Requiere: VOYAGE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY');

  const voyageKey = process.env.VOYAGE_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  const cohereKey = process.env.COHERE_API_KEY;

  if (!voyageKey || !openaiKey || !cohereKey) {
    console.log('   ⚠️  Faltan API keys — salteando');
    return;
  }

  const sdk = rag({
    provider: createVoyageEmbedding({ apiKey: voyageKey }),
    store: createMemoryStore({ dimensions: 512 }),
    generator: createOpenAIGenerator({ apiKey: openaiKey, model: 'gpt-4o-mini' }),
    reranker: createCohereReranker({ apiKey: cohereKey }),
  });

  step('Ingestando...');
  await sdk.ingest(DOCUMENTS);

  step('Query rerankeada: "¿Qué es RAG y cómo se relaciona con los microservicios?"');
  const result = await sdk.query('¿Qué es RAG y cómo se relaciona con los microservicios?', {
    topK: 8,
    rerank: { topN: 3 },
  });

  for (let i = 0; i < result.results.length; i++) {
    console.log(`  ${i + 1}. [${result.results[i].score.toFixed(4)}] ${result.results[i].content.slice(0, 80)}...`);
  }
}
// ─── 5. RAG Completo: Streaming + Chunking + Híbrido ────────────────────────
async function ragFullPipeline() {
  section('5. Pipeline completo — Markdown chunking + hybrid search + streaming');

  step('Creando chunker markdown + keyword searcher de juguete...');
  const chunker = new MarkdownChunker({ chunkSize: 400, overlap: 100 });

  const toyKeywordSearcher = {
    id: 'toy-keyword',
    // eslint-disable-next-line @typescript-eslint/require-await
    keywordSearch: async (text: string, _options?: Record<string, unknown>) => {
      const keywords = text.toLowerCase().split(/\s+/);
      return [
        {
          id: 'k1',
          content: 'Los contenedores Docker comparten el kernel del host, haciéndolos más ligeros que las VMs.',
          score: keywords.some((k) => k.includes('docker') || k.includes('contenedor')) ? 0.9 : 0.1,
          metadata: {} as Metadata,
          documentId: 'doc-2',
          chunkIndex: 0,
        },
      ];
    },
  };

  step('Creando SDK con componentes locales...');
  const sdk = rag({
    provider: createLocalEmbedding(),
    store: createMemoryStore({ dimensions: 384 }),
    chunker,
    keywordSearcher: toyKeywordSearcher,
    reranker: createLocalReranker(),
  });

  step('Ingestando con chunking markdown...');
  await sdk.ingest(DOCUMENTS);

  // ── Query híbrida ──
  step('Búsqueda híbrida (vector + keyword, RRF fusion)...');
  const hybridResult = await sdk.query('Docker y TypeScript', {
    topK: 5,
    hybrid: { enabled: true, vectorWeight: 1, keywordWeight: 0.5 },
  });

  console.log('   🔀 Resultados híbridos (pre-rerank):');
  for (const r of hybridResult.results) {
    console.log(`     [${r.score.toFixed(4)}] ${r.content.slice(0, 70)}...`);
  }

  // ── Generación con streaming ──
  if (process.env.OPENAI_API_KEY) {
    section('5b. Streaming generation (requiere OpenAI)');
    const streamSdk = rag({
      provider: createLocalEmbedding(),
      store: createMemoryStore({ dimensions: 384 }),
      generator: createOpenAIGenerator({ apiKey: process.env.OPENAI_API_KEY, model: 'gpt-4o-mini' }),
      reranker: createLocalReranker(),
    });

    await streamSdk.ingest(DOCUMENTS);

    step('generateStream("Explica RAG en 2 frases")...');
    process.stdout.write('   📝 ');
    for await (const token of streamSdk.generateStream('Explica RAG en 2 frases')) {
      process.stdout.write(token);
    }
    console.log();
  }
}

// ─── 6. Chunking Semántico ─────────────────────────────────────────────────
async function ragSemanticChunking() {
  section('6. Chunking semántico con embeddings locales');

  step('Creando chunker semántico...');
  const provider = createLocalEmbedding();
  const chunker = new SemanticChunker(provider, {
    chunkSize: 300,
    threshold: 0.4,
  });

  const sdk = rag({
    provider,
    store: createMemoryStore({ dimensions: 384 }),
    chunker,
    reranker: createLocalReranker(),
  });

  step('Ingestando con chunking semántico...');
  await sdk.ingest([
    {
      id: 'semantic-1',
      content: `TypeScript es un superset de JavaScript. Agrega tipos estáticos, interfaces
y genéricos. Los genéricos permiten escribir funciones reutilizables que trabajan
con cualquier tipo. TypeScript se compila a JavaScript plano. El compilador tsc
realiza type-checking en tiempo de compilación, atrapando errores antes de
ejecutar el código.`,
    },
  ]);

  step('Query semántica...');
  const result = await sdk.query('¿Cómo funcionan los genéricos?', { topK: 3 });
  console.log(`   Top chunk: [${result.results[0]?.score.toFixed(4)}] ${result.results[0]?.content.slice(0, 100)}...`);
}

// ─── 7. Custom Metadata tipado ─────────────────────────────────────────────
async function ragTypedMetadata() {
  section('7. Metadata tipada — type-safe end-to-end');

  interface DocMetadata {
    [key: string]: unknown;
    author: string;
    category: 'architecture' | 'language' | 'ai';
    tags: string[];
  }

  step('Creando SDK con metadata tipada...');
  const sdk = rag<DocMetadata>({
    provider: createLocalEmbedding(),
    store: createMemoryStore<DocMetadata>({ dimensions: 384 }),
  });

  step('Ingestando documentos con metadata...');
  await sdk.ingest([
    {
      content: 'RAG combina búsqueda vectorial con generación de lenguaje.',
      metadata: { author: 'Equipo AI', category: 'ai', tags: ['rag', 'llm', 'embeddings'] },
    },
    {
      content: 'TypeScript ofrece genéricos, tipos condicionales, y mapped types.',
      metadata: { author: 'Dev Team', category: 'language', tags: ['typescript', 'types'] },
    },
    {
      content: 'Los microservicios permiten despliegue independiente por equipo.',
      metadata: { author: 'Dev Team', category: 'architecture', tags: ['microservices', 'deployment'] },
    },
  ]);

  step('Query con filtro por metadata (solo documentos de AI)...');
  const result = await sdk.query('¿Qué es RAG?', {
    topK: 3,
    filter: { category: 'ai' },
  });

  for (const r of result.results) {
    // TypeScript infiere el tipo de metadata automáticamente
    console.log(`   [${r.metadata.category}] ${r.metadata.author}: ${r.content.slice(0, 60)}...`);
  }
}

// ─── 8. Manejo de errores ──────────────────────────────────────────────────
async function ragErrorHandling() {
  section('8. Manejo de errores — provider offline, dimension mismatch');

  // ── Error: provider offline ──
  try {
    const sdk = rag({
      provider: createOllamaEmbedding({
        baseURL: 'http://localhost:9999', // puerto incorrecto
        model: 'nomic-embed-text',
      }),
      store: createMemoryStore({ dimensions: 768 }),
    });
    await sdk.query('test');
  } catch (err) {
    console.log('   ✅ Error esperado (Ollama offline):', (err as Error).message.slice(0, 80));
  }

  // ── Error: dimension mismatch ──
  try {
    const sdk = rag({
      provider: createLocalEmbedding(), // 384 dims
      store: createMemoryStore({ dimensions: 1536 }), // espera 1536
    });
    await sdk.ingest(DOCUMENTS);
  } catch (err) {
    if (err instanceof DimensionMismatchError) {
      console.log('   ✅ Error esperado (dimension mismatch):', err.message.slice(0, 80));
    } else if (err instanceof ConfigurationError) {
      console.log('   ✅ Error esperado (dimension mismatch):', err.message.slice(0, 80));
    } else {
      console.log('   ⚠️  Otro error:', err);
    }
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────
async function main() {
  console.log('═'.repeat(60));
  console.log('  RAG SDK V3 — Full Pipeline Showcase');
  console.log('═'.repeat(60));

  divider();

  // ✅ Secciones 100% locales (sin APIs, sin servidores)
  await ragFullyLocal();
  divider();
  await ragSemanticChunking();
  divider();
  await ragTypedMetadata();
  divider();
  await ragErrorHandling();
  divider();

  // ✅ Pipeline completo (usa OpenAI para streaming si está configurado)
  await ragFullPipeline();
  divider();

  // ⚠️ Requieren servicios externos
  await ragOllama();
  divider();
  await ragCohere();
  divider();
  await ragMultiCloud();
  divider();

  console.log('🎉 Demo completa.\n');
}

main().catch(console.error);
