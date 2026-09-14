import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // FastAPI mounts the build under /static, so assets must be
  // requested from there rather than the site root.
  base: '/static/',
  // FastAPI serves the build straight out of app/static, so a clone can run
  // uvicorn without needing node installed.
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
