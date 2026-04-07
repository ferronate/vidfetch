import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Scope Turbopack to this frontend directory to avoid watching the repo root.
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
