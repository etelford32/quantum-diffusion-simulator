'use client'

// visualization/src/components/QuantumScene.tsx
//
// Main 3D scene for quantum wavefunction visualization.
// Renders |ψ|² probability density as a volumetric point cloud.
// Landau levels appear as Gaussian blobs (n=0) or ring distributions (n≥1).

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { useRef, useEffect } from 'react'

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SimulationFrame {
  t: number
  probability: Float32Array   // flattened [Nx*Ny*Nz]
  psi_real?: Float32Array
  psi_imag?: Float32Array
  energy?: number             // ⟨E⟩ in eV
}

export interface SimulationMetadata {
  Nx: number
  Ny: number
  Nz: number
  x_nm: Float32Array
  y_nm: Float32Array
  z_nm: Float32Array
  B_tesla: number
  T_kelvin: number
  n_landau: number
  landau_energies_ev: Float32Array
}

export interface QuantumSceneProps {
  frames: SimulationFrame[]
  metadata: SimulationMetadata
  currentFrame?: number
  autoPlay?: boolean
  frameRate?: number
  colormap?: 'plasma' | 'viridis' | 'phase'
  threshold?: number
  pointSize?: number
  showAxes?: boolean
  showLandauRings?: boolean
}

// ── Colormaps ──────────────────────────────────────────────────────────────────

// Perceptually uniform Matplotlib plasma (dark→yellow)
const PLASMA_STOPS = [
  [0.050383, 0.029803, 0.527975],
  [0.494877, 0.012525, 0.657865],
  [0.798216, 0.280197, 0.469538],
  [0.973416, 0.585761, 0.256415],
  [0.940015, 0.975158, 0.131326],
]

// Perceptually uniform Matplotlib viridis (dark blue→yellow)
const VIRIDIS_STOPS = [
  [0.267004, 0.004874, 0.329415],
  [0.229739, 0.322361, 0.545706],
  [0.127568, 0.566949, 0.550556],
  [0.369214, 0.788888, 0.382914],
  [0.993248, 0.906157, 0.143936],
]

function sampleColormap(stops: number[][], t: number): THREE.Color {
  const n = stops.length - 1
  const i = Math.min(Math.floor(t * n), n - 1)
  const f = t * n - i
  const [r1, g1, b1] = stops[i]
  const [r2, g2, b2] = stops[i + 1]
  return new THREE.Color(r1 + f * (r2 - r1), g1 + f * (g2 - g1), b1 + f * (b2 - b1))
}

// ── Point cloud builder ────────────────────────────────────────────────────────

function buildPointCloud(
  frame: SimulationFrame,
  meta: SimulationMetadata,
  threshold: number,
  colormap: string,
): { positions: Float32Array; colors: Float32Array; count: number } {
  const { Nx, Ny, Nz, x_nm, y_nm, z_nm } = meta
  const prob = frame.probability

  let maxProb = 0
  for (let i = 0; i < prob.length; i++) if (prob[i] > maxProb) maxProb = prob[i]
  if (maxProb === 0) return { positions: new Float32Array(0), colors: new Float32Array(0), count: 0 }

  const absThresh = threshold * maxProb
  let count = 0
  for (let i = 0; i < prob.length; i++) if (prob[i] > absThresh) count++

  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  let idx = 0

  for (let ix = 0; ix < Nx; ix++) {
    for (let iy = 0; iy < Ny; iy++) {
      for (let iz = 0; iz < Nz; iz++) {
        const fi = ix * Ny * Nz + iy * Nz + iz
        const p = prob[fi]
        if (p <= absThresh) continue

        const t = Math.pow(p / maxProb, 0.5)  // sqrt stretch for dynamic range

        positions[idx * 3]     = x_nm[ix]
        positions[idx * 3 + 1] = y_nm[iy]
        positions[idx * 3 + 2] = z_nm[iz]

        let color: THREE.Color
        if (colormap === 'phase' && frame.psi_real && frame.psi_imag) {
          const phi = Math.atan2(frame.psi_imag[fi], frame.psi_real[fi])
          const hue = (phi + Math.PI) / (2 * Math.PI)
          color = new THREE.Color().setHSL(hue, 0.9, 0.3 + 0.35 * t)
        } else if (colormap === 'viridis') {
          color = sampleColormap(VIRIDIS_STOPS, t)
        } else {
          color = sampleColormap(PLASMA_STOPS, t)
        }

        colors[idx * 3]     = color.r
        colors[idx * 3 + 1] = color.g
        colors[idx * 3 + 2] = color.b

        idx++
      }
    }
  }

  return { positions, colors, count }
}

// ── GLSL shaders ───────────────────────────────────────────────────────────────

const VERT = `
  attribute vec3 aColor;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uPointSize;
  uniform float uTime;

  void main() {
    vColor = aColor;
    float brightness = length(aColor);
    float pulse = 1.0 + 0.08 * sin(uTime * 2.5 + position.z * 0.4);
    vAlpha = clamp(brightness * pulse, 0.0, 1.0);
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uPointSize * pulse * (300.0 / -mvPos.z);
    gl_Position = projectionMatrix * mvPos;
  }
`

const FRAG = `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float d = length(coord);
    if (d > 0.5) discard;
    float alpha = vAlpha * (1.0 - smoothstep(0.3, 0.5, d));
    gl_FragColor = vec4(vColor, alpha);
  }
`

// ── Component ─────────────────────────────────────────────────────────────────

export function QuantumScene({
  frames,
  metadata,
  currentFrame = 0,
  autoPlay = false,
  frameRate = 10,
  colormap = 'plasma',
  threshold = 0.05,
  pointSize = 2.0,
  showAxes = true,
  showLandauRings = true,
}: QuantumSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  // Keep latest props accessible inside the animation loop without re-subscribing
  const propsRef = useRef({ frames, metadata, colormap, threshold, pointSize, autoPlay, frameRate, currentFrame })
  propsRef.current = { frames, metadata, colormap, threshold, pointSize, autoPlay, frameRate, currentFrame }

  // Re-run scene setup when metadata changes (grid size / B / n changed)
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    let width = mount.clientWidth
    let height = mount.clientHeight

    // ── Renderer ────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(width, height)
    renderer.setClearColor(0x000510, 1)
    mount.appendChild(renderer.domElement)

    // ── Scene ───────────────────────────────────────────────────────
    const scene = new THREE.Scene()

    // ── Camera ──────────────────────────────────────────────────────
    const meta = propsRef.current.metadata
    const boxSize = Math.max(
      meta.x_nm[meta.x_nm.length - 1] - meta.x_nm[0],
      meta.y_nm[meta.y_nm.length - 1] - meta.y_nm[0],
    )
    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100_000)
    camera.position.set(boxSize * 1.4, boxSize * 0.9, boxSize * 1.4)
    camera.lookAt(0, 0, 0)

    // ── OrbitControls ───────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.6

    // ── Axes helper ─────────────────────────────────────────────────
    if (showAxes) {
      scene.add(new THREE.AxesHelper(boxSize * 0.5))
    }

    // ── Landau ring (analytical peak radius for n-th level) ─────────
    if (showLandauRings && meta.B_tesla > 0) {
      // Peak of |ψₙ|² in x-y plane: r_peak = lB·√(2n+1) for symmetric gauge
      // For n=0 this gives lB — marks the 1/e² extent of the Gaussian
      const lB_nm = Math.sqrt(1.054571817e-34 / (1.602176634e-19 * meta.B_tesla)) * 1e9
      const rPeak = lB_nm * Math.sqrt(2 * meta.n_landau + 1)
      const tubeR = Math.max(0.2, rPeak * 0.015)
      const ringGeom = new THREE.TorusGeometry(rPeak, tubeR, 8, 128)
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x00ffcc,
        wireframe: false,
        opacity: 0.25,
        transparent: true,
      })
      const ring = new THREE.Mesh(ringGeom, ringMat)
      ring.rotation.x = Math.PI / 2
      scene.add(ring)
    }

    // ── Shader material ─────────────────────────────────────────────
    const material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uPointSize: { value: pointSize },
        uTime: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })

    // ── Point cloud ─────────────────────────────────────────────────
    let currentPoints: THREE.Points | null = null

    function rebuildCloud(frame: SimulationFrame) {
      if (currentPoints) {
        scene.remove(currentPoints)
        currentPoints.geometry.dispose()
      }
      const { positions, colors, count } = buildPointCloud(
        frame,
        propsRef.current.metadata,
        propsRef.current.threshold,
        propsRef.current.colormap,
      )
      if (count === 0) return
      const geom = new THREE.BufferGeometry()
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      geom.setAttribute('aColor', new THREE.BufferAttribute(colors, 3))
      currentPoints = new THREE.Points(geom, material)
      scene.add(currentPoints)
    }

    const { frames } = propsRef.current
    if (frames.length > 0) rebuildCloud(frames[0])

    // ── Animation loop ──────────────────────────────────────────────
    let rafId = 0
    let frameIdx = currentFrame
    let lastFrameSwap = 0

    const animate = (ts: number) => {
      rafId = requestAnimationFrame(animate)

      const p = propsRef.current
      material.uniforms.uPointSize.value = p.pointSize
      material.uniforms.uTime.value = ts * 0.001

      controls.update()

      // Auto-advance frames
      if (p.autoPlay && p.frames.length > 1) {
        const interval = 1000 / p.frameRate
        if (ts - lastFrameSwap > interval) {
          frameIdx = (frameIdx + 1) % p.frames.length
          rebuildCloud(p.frames[frameIdx])
          lastFrameSwap = ts
        }
      }

      renderer.render(scene, camera)
    }
    rafId = requestAnimationFrame(animate)

    // ── Resize handler ──────────────────────────────────────────────
    const onResize = () => {
      if (!mount) return
      width = mount.clientWidth
      height = mount.clientHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    window.addEventListener('resize', onResize)

    // ── Cleanup ─────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metadata, showAxes, showLandauRings])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* HUD */}
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 16,
          color: 'rgba(0,255,200,0.75)',
          fontFamily: "'Courier New', monospace",
          fontSize: 12,
          lineHeight: 1.8,
          pointerEvents: 'none',
        }}
      >
        <div>B = {metadata.B_tesla.toFixed(1)} T</div>
        <div>T = {metadata.T_kelvin.toFixed(1)} K</div>
        <div>n = {metadata.n_landau} (Landau)</div>
        {frames[0]?.energy !== undefined && (
          <div>⟨E⟩ = {(frames[0].energy! * 1000).toFixed(4)} meV</div>
        )}
      </div>
    </div>
  )
}

export default QuantumScene
