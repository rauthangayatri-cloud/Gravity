import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** Keep the dashboard's charting dependency out of the application entry chunk. */
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return "react";
          if (id.includes("victory-vendor") || /[\\/]node_modules[\\/]d3-/.test(id)) return "charts-core";
          if (/[\\/]node_modules[\\/]recharts[\\/]/.test(id)) return "recharts";
          return undefined;
        },
      },
    },
  },
});
