// visualization/src/components/QuantumScene.tsx
// 
// Main 3D scene for quantum wavefunction visualization.
// Renders precomputed |ψ|² probability density as volumetric point cloud
// with animated frame interpolation.
//
// Landau levels appear as beautiful toroidal/ring distributions in x-y plane,
// with free-particle behavior along z.

import * as THREE from 'three'
import { useRef, useEffect, useMemo, useState } from 'react'

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SimulationFrame {
  t: number          // time in seconds
  probability: Float32Array  // flattened [Nx*Ny*Nz] probability density
  psi_real?: Float32Array
  psi_imag?: Float32Array
  energy?: number    // ⟨E⟩ in eV
}

export interface SimulationMetadata {
  Nx: number
  Ny: number
  Nz: number
  x_nm: Float32Array    // coordinate axes in nm
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
  frameRate?: number          // frames per second for animation
  colormap?: 'plasma' | 'viridis' | 'phase'
  threshold?: number          // minimum |ψ|² to render (0-1, normalized)
  pointSize?: number
  showAxes?: boolean
  showLandauRings?: boolean    // overlay analytical Landau ring positions
}

// ── Colormaps ──────────────────────────────────────────────────────────────────

const PLASMA_COLORS = [
  [0.050383, 0.029803, 0.527975],
  [0.494877, 0.012525, 0.657865],
  [0.798216, 0.280197, 0.469538],
  [0.973416, 0.585761, 0.256415],
  [0.940015, 0.975158, 0.131326],
]

function plasmaColor(t: number): THREE.Color {
  const n = PLASMA_COLORS.length - 1
  const i = Math.min(Math.floor(t * n), n - 1)
  const f = t * n - i
  const [r1, g1, b1] = PLASMA_COLORS[i]
  const [r2, g2, b2] = PLASMA_COLORS[Math.min(i + 1, n)]
  return new THREE.Color(
    r1 + f * (r2 - r1),
    g1 + f * (g2 - g1),
    b1 + f * (b2 - b1)
  )
}

// ── Point Cloud Builder ────────────────────────────────────────────────────────

function buildPointCloud(
  frame: SimulationFrame,
  meta: SimulationMetadata,
  threshold: number,
  colormap: string,
  phase?: Float32Array
): { positions: Float32Array; colors: Float32Array; count: number } {
  const { Nx, Ny, Nz, x_nm, y_nm, z_nm } = meta
  const prob = frame.probability
  
  // Find max for normalization
  let maxProb = 0
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] > maxProb) maxProb = prob[i]
  }
  
  if (maxProb === 0) return { positions: new Float32Array(0), colors: new Float32Array(0), count: 0 }
  
  // Count points above threshold
  let count = 0
  const absThreshold = threshold * maxProb
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] > absThreshold) count++
  }
  
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  let idx = 0
  
  for (let ix = 0; ix < Nx; ix++) {
    for (let iy = 0; iy < Ny; iy++) {
      for (let iz = 0; iz < Nz; iz++) {
        const flatIdx = ix * Ny * Nz + iy * Nz + iz
        const p = prob[flatIdx]
        
        if (p <= absThreshold) continue
        
        const t = p / maxProb  // normalized 0-1
        
        positions[idx * 3 + 0] = x_nm[ix]
        positions[idx * 3 + 1] = y_nm[iy]
        positions[idx * 3 + 2] = z_nm[iz]
        
        let color: THREE.Color
        if (colormap === 'phase' && phase) {
          // Color by quantum phase — beautiful interference patterns
          const phi = phase[flatIdx]  // -π to π
          const hue = (phi + Math.PI) / (2 * Math.PI)
          color = new THREE.Color().setHSL(hue, 0.9, 0.3 + 0.4 * t)
        } else {
          // Plasma colormap by probability magnitude
          color = plasmaColor(Math.pow(t, 0.5))  // sqrt for better dynamic range
        }
        
        colors[idx * 3 + 0] = color.r
        colors[idx * 3 + 1] = color.g
        colors[idx * 3 + 2] = color.b
        
        idx++
      }
    }
  }
  
  return { positions, colors, count }
}

// ── Vertex Shader ──────────────────────────────────────────────────────────────

const VERTEX_SHADER = `
  attribute vec3 color;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float pointSize;
  uniform float time;
  
  void main() {
    vColor = color;
    
    // Subtle breathing animation — size pulses with probability magnitude
    float brightness = length(color);
    float pulse = 1.0 + 0.1 * sin(time * 3.0 + position.z * 0.5);
    vAlpha = brightness * pulse;
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * pulse * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`

// ── Fragment Shader ────────────────────────────────────────────────────────────

const FRAGMENT_SHADER = `
  varying vec3 vColor;
  varying float vAlpha;
  
  void main() {
    // Circular points with soft edges
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    
    float alpha = vAlpha * (1.0 - smoothstep(0.3, 0.5, dist));
    gl_FragColor = vec4(vColor, alpha);
  }
`

// ── Main Component ─────────────────────────────────────────────────────────────

export function QuantumScene({
  frames,
  metadata,
  currentFrame = 0,
  autoPlay = true,
  frameRate = 10,
  colormap = 'plasma',
  threshold = 0.05,
  pointSize = 2.0,
  showAxes = true,
  showLandauRings = true,
}: QuantumSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene>()
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const pointsRef = useRef<THREE.Points>()
  const materialRef = useRef<THREE.ShaderMaterial>()
  const frameIdxRef = useRef(currentFrame)
  const animFrameRef = useRef<number>()
  const lastFrameTimeRef = useRef(0)
  
  // ── Scene setup ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mountRef.current) return
    
    const width = mountRef.current.clientWidth
    const height = mountRef.current.clientHeight
    
    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000510)
    sceneRef.current = scene
    
    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000)
    const boxSize = Math.max(
      metadata.x_nm[metadata.x_nm.length - 1] - metadata.x_nm[0],
      metadata.y_nm[metadata.y_nm.length - 1] - metadata.y_nm[0],
    )
    camera.position.set(boxSize * 1.5, boxSize * 1.0, boxSize * 1.5)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer
    
    // Axes
    if (showAxes) {
      const axesHelper = new THREE.AxesHelper(boxSize * 0.6)
      scene.add(axesHelper)
    }
    
    // Ambient light for any mesh objects
    scene.add(new THREE.AmbientLight(0xffffff, 0.3))
    
    // Landau rings (analytical positions)
    if (showLandauRings && metadata.B_tesla > 0) {
      const lB_nm = Math.sqrt(1.054e-34 / (1.602e-19 * metadata.B_tesla)) * 1e9
      const ringGeom = new THREE.TorusGeometry(lB_nm * Math.sqrt(2 * metadata.n_landau + 1), 0.3, 8, 64)
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, opacity: 0.3, transparent: true })
      const ring = new THREE.Mesh(ringGeom, ringMat)
      ring.rotation.x = Math.PI / 2
      scene.add(ring)
    }
    
    // Initial point cloud
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        pointSize: { value: pointSize },
        time: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    materialRef.current = material
    
    if (frames.length > 0) {
      updatePointCloud(frames[0], scene, material)
    }
    
    // Animation loop
    let lastTime = 0
    const animate = (timestamp: number) => {
      animFrameRef.current = requestAnimationFrame(animate)
      
      // Update shader time for breathing effect
      if (material.uniforms) {
        material.uniforms.time.value = timestamp * 0.001
      }
      
      // Auto-advance frames
      if (autoPlay && frames.length > 1) {
        const frameDt = 1000 / frameRate
        if (timestamp - lastFrameTimeRef.current > frameDt) {
          frameIdxRef.current = (frameIdxRef.current + 1) % frames.length
          updatePointCloud(frames[frameIdxRef.current], scene, material)
          lastFrameTimeRef.current = timestamp
        }
      }
      
      // Slow auto-rotate
      if (sceneRef.current) {
        sceneRef.current.rotation.y += 0.002
      }
      
      renderer.render(scene, camera)
    }
    
    requestAnimationFrame(animate)
    
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      renderer.dispose()
      mountRef.current?.removeChild(renderer.domElement)
    }
  }, [metadata])
  
  function updatePointCloud(
    frame: SimulationFrame,
    scene: THREE.Scene,
    material: THREE.ShaderMaterial
  ) {
    // Remove old points
    if (pointsRef.current) {
      scene.remove(pointsRef.current)
      pointsRef.current.geometry.dispose()
    }
    
    const phaseArr = frame.psi_real && frame.psi_imag
      ? (() => {
          const phase = new Float32Array(frame.psi_real.length)
          for (let i = 0; i < phase.length; i++) {
            phase[i] = Math.atan2(frame.psi_imag![i], frame.psi_real[i])
          }
          return phase
        })()
      : undefined
    
    const { positions, colors, count } = buildPointCloud(
      frame, metadata, threshold, colormap, phaseArr
    )
    
    if (count === 0) return
    
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    
    const points = new THREE.Points(geometry, material)
    scene.add(points)
    pointsRef.current = points
  }
  
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      
      {/* HUD overlay */}
      <div style={{
        position: 'absolute', top: 16, left: 16,
        color: 'rgba(0,255,200,0.8)', fontFamily: 'monospace', fontSize: 12,
        pointerEvents: 'none',
      }}>
        <div>B = {metadata.B_tesla.toFixed(1)} T</div>
        <div>T = {metadata.T_kelvin.toFixed(1)} K</div>
        <div>n = {metadata.n_landau} (Landau level)</div>
        <div>frame {frameIdxRef.current + 1}/{frames.length}</div>
        {frames[frameIdxRef.current]?.energy && (
          <div>⟨E⟩ = {frames[frameIdxRef.current].energy?.toFixed(4)} eV</div>
        )}
      </div>
    </div>
  )
}

export default QuantumScene
