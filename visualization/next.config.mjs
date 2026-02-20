/** @type {import('next').NextConfig} */
const nextConfig = {
  // Transpile three.js ESM sub-packages (OrbitControls, etc.)
  transpilePackages: ['three'],
}

export default nextConfig
