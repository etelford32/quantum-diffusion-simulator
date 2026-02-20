'use client'

import { useMemo, useState } from 'react'
import dynamic from 'next/dynamic'
import {
  computeLandauState,
  cyclotronFrequency,
  landauEnergy,
  magneticLength,
} from '@/utils/landau'
import type { SimulationFrame, SimulationMetadata } from '@/components/QuantumScene'

// Skip SSR — WebGL requires a browser environment
const QuantumScene = dynamic(() => import('@/components/QuantumScene'), {
  ssr: false,
  loading: () => (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: 'rgba(0,255,200,0.5)', fontFamily: 'monospace', fontSize: 13,
      letterSpacing: 2,
    }}>
      INITIALIZING SIMULATION…
    </div>
  ),
})

const E_CHARGE = 1.602176634e-19

export default function Home() {
  const [B, setB] = useState(1.0)
  const [n, setN] = useState(0)
  const [colormap, setColormap] = useState<'plasma' | 'viridis' | 'phase'>('plasma')
  const [threshold, setThreshold] = useState(0.02)
  const [pointSize, setPointSize] = useState(2.0)

  // Recompute wavefunction when physics params change
  const gridData = useMemo(
    () => computeLandauState(n, B, 32, 32, 32, 80),
    [n, B],
  )

  const frames: SimulationFrame[] = useMemo(
    () => [
      {
        t: 0,
        probability: gridData.probability,
        psi_real: gridData.psi_real,
        psi_imag: gridData.psi_imag,
        energy: gridData.energy_ev,
      },
    ],
    [gridData],
  )

  const metadata: SimulationMetadata = useMemo(
    () => ({
      Nx: gridData.Nx,
      Ny: gridData.Ny,
      Nz: gridData.Nz,
      x_nm: gridData.x_nm,
      y_nm: gridData.y_nm,
      z_nm: gridData.z_nm,
      B_tesla: B,
      T_kelvin: 0,
      n_landau: n,
      landau_energies_ev: new Float32Array(
        [0, 1, 2, 3, 4, 5].map((i) => landauEnergy(i, B) / E_CHARGE),
      ),
    }),
    [gridData, B, n],
  )

  const wc = cyclotronFrequency(B)
  const lB_nm = magneticLength(B) * 1e9
  const E0_meV = landauEnergy(0, B) / E_CHARGE * 1000
  const En_meV = landauEnergy(n, B) / E_CHARGE * 1000

  const panelStyle: React.CSSProperties = {
    position: 'absolute',
    top: 16,
    right: 16,
    background: 'rgba(0,5,16,0.88)',
    border: '1px solid rgba(0,255,200,0.25)',
    borderRadius: 8,
    padding: '16px 18px',
    color: '#00ffc8',
    fontFamily: "'Courier New', monospace",
    fontSize: 12,
    width: 230,
    backdropFilter: 'blur(6px)',
    userSelect: 'none',
  }

  const labelStyle: React.CSSProperties = {
    display: 'block',
    marginBottom: 2,
    opacity: 0.8,
  }

  const sliderRowStyle: React.CSSProperties = {
    marginBottom: 10,
  }

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      {/* 3-D scene — takes full viewport */}
      <QuantumScene
        frames={frames}
        metadata={metadata}
        colormap={colormap}
        threshold={threshold}
        pointSize={pointSize}
        autoPlay={false}
        showAxes
        showLandauRings
      />

      {/* ── Controls panel ─────────────────────────────────────────── */}
      <div style={panelStyle}>
        <div style={{ fontWeight: 'bold', marginBottom: 14, fontSize: 13, letterSpacing: 1 }}>
          CONTROLS
        </div>

        <div style={sliderRowStyle}>
          <label style={labelStyle}>B field: {B.toFixed(1)} T</label>
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.1}
            value={B}
            onChange={(e) => setB(parseFloat(e.target.value))}
          />
        </div>

        <div style={sliderRowStyle}>
          <label style={labelStyle}>Landau level n = {n}</label>
          <input
            type="range"
            min={0}
            max={5}
            step={1}
            value={n}
            onChange={(e) => setN(parseInt(e.target.value))}
          />
        </div>

        <div style={sliderRowStyle}>
          <label style={labelStyle}>Threshold: {threshold.toFixed(3)}</label>
          <input
            type="range"
            min={0.001}
            max={0.25}
            step={0.001}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
        </div>

        <div style={{ ...sliderRowStyle, marginBottom: 14 }}>
          <label style={labelStyle}>Point size: {pointSize.toFixed(1)}</label>
          <input
            type="range"
            min={0.5}
            max={6}
            step={0.1}
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
          />
        </div>

        {/* Colormap selector */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ ...labelStyle, marginBottom: 6 }}>Colormap</label>
          <div style={{ display: 'flex', gap: 4 }}>
            {(['plasma', 'viridis', 'phase'] as const).map((c) => (
              <button
                key={c}
                onClick={() => setColormap(c)}
                style={{
                  flex: 1,
                  padding: '4px 0',
                  fontSize: 10,
                  background: colormap === c ? 'rgba(0,255,200,0.18)' : 'transparent',
                  border: `1px solid ${colormap === c ? '#00ffc8' : 'rgba(0,255,200,0.25)'}`,
                  color: '#00ffc8',
                  cursor: 'pointer',
                  borderRadius: 4,
                  fontFamily: 'monospace',
                }}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        {/* Physics readout */}
        <div
          style={{
            borderTop: '1px solid rgba(0,255,200,0.18)',
            paddingTop: 12,
            lineHeight: 1.9,
          }}
        >
          <div style={{ color: 'rgba(0,255,200,0.45)', fontSize: 10, marginBottom: 4, letterSpacing: 1 }}>
            PHYSICS  (B = {B.toFixed(1)} T)
          </div>
          <div>ωc = {wc.toExponential(3)} rad/s</div>
          <div>lB = {lB_nm.toFixed(2)} nm</div>
          <div>E₀ = {E0_meV.toFixed(4)} meV</div>
          <div style={{ color: '#7fffd4' }}>
            E{n} = {En_meV.toFixed(4)} meV
          </div>
        </div>
      </div>

      {/* ── Footer hint ────────────────────────────────────────────── */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          color: 'rgba(0,255,200,0.3)',
          fontFamily: 'monospace',
          fontSize: 11,
          textAlign: 'center',
          pointerEvents: 'none',
          letterSpacing: 1,
        }}
      >
        QUANTUM DIFFUSION SIMULATOR — drag to orbit · scroll to zoom
      </div>
    </div>
  )
}
