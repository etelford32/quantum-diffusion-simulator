import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Quantum Diffusion Simulator',
  description:
    'Interactive 3D visualization of quantum wavefunctions — Landau levels, electron dynamics, emergent complexity.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
