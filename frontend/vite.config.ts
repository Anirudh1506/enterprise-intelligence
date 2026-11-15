import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// API calls go through the dev server proxy, so no CORS setup is needed locally
const backend = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': backend,
      '/health': backend,
    },
  },
})
