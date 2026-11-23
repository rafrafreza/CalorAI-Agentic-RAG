import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const api = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': api,
      '/health': api,
      '/metrics': api,
    },
  },
})
