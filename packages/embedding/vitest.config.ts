import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@rag-sdk/core': path.resolve(__dirname, '../../packages/core/src/index.ts'),
      '@rag-sdk/embedding': path.resolve(__dirname, './src/index.ts'),
    },
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
  }
});
