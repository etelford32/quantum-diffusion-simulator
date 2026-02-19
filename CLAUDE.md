# CLAUDE.md — Quantum Diffusion Simulator
# Project context for Claude Code sessions

## Project Identity
Open-source 3D quantum mechanics simulation platform.
Repo: https://github.com/etelford32/quantum-diffusion-simulator
Author: Elliot Telford (elliottelford.com)
License: MIT

## Vision
Simulate a hierarchy of emergent physical complexity:
    Magnetism → Subatomic Behavior → Chemistry → Fluid Dynamics → Convection → Atmosphere

Starting with electrons in magnetic fields (Landau levels), building toward
hydrogen pairing, molecular chemistry, fluid dynamics, and planetary atmospheres.
Scientifically rigorous. Precomputed for accuracy. Rendered in interactive 3D.

---

## Physics Conventions

### Units
- ALL physics computations in SI units (meters, kilograms, seconds, Tesla, Joules)
- Frontend visualization converts to nanometers (nm) for display
- Energy reported in both Joules (internal) and eV (output/display)
- Never mix unit systems within a single module

### Gauge Choice
We use the **Lorenz gauge** as the overarching framework:
    ∇·A + (1/c²)(∂φ/∂t) = 0

For static magnetic fields (magnetostatics), this reduces to the **Coulomb gauge**:
    ∇·A = 0

For the uniform magnetic field B = B·ẑ we use **symmetric gauge**:
    A = (B/2)(-y, x, 0)

This choice (vs Landau gauge A = (0, Bx, 0)) gives:
- Rotational symmetry in x-y plane — eigenstates have definite angular momentum
- Natural ring/torus probability distributions (Landau levels look like rings)
- Direct connection to angular momentum quantum number m

Do NOT switch to Landau gauge without a clear reason and updating this file.

### Hamiltonian
The full single-particle Hamiltonian with minimal coupling:

    H = (p - eA)²/2m + V(r)
      = p²/2m - (e/2m)(p·A + A·p) + e²A²/2m + V(r)

In symmetric gauge with ∇·A = 0:
    p·A = A·p  (they commute)

So:
    H = p²/2m - (e/m)A·p + e²A²/2m + V(r)

The cross term -eA·p/m is the **paramagnetic term** (linear in B).
The e²A²/2m term is the **diamagnetic term** (quadratic in B).

### Landau Levels
Exact energy eigenvalues for electron in uniform B along z:
    Eₙ = ℏωc(n + 1/2)     n = 0, 1, 2, ...

where cyclotron frequency: ωc = eB/m

Magnetic length (natural length scale):
    lB = √(ℏ/eB)
    lB ≈ 25.66 nm at B = 1T

Wavefunction in symmetric gauge (ground state of each level):
    ψₙ(x,y,z) = Cₙ · ((x+iy)/lB)ⁿ · exp(-(x²+y²)/4lB²) · exp(ikz·z)

Key numbers at B = 1T:
    ωc  = 1.7588 × 10¹¹ rad/s
    lB  = 25.66 nm
    E₀  = 0.05788 meV (ground state zero-point energy)
    ΔE  = ℏωc = 0.11576 meV (level spacing)

### Temperature
Temperature enters in two ways:

1. **Initial state population** — Boltzmann distribution over Landau levels:
       Pₙ ∝ exp(-Eₙ/kT)
   At T=0: only n=0 occupied. At T=4.2K (LHe): quantum effects visible.
   At T=300K: need B >> 100T for quantum effects (kT >> ℏωc at room T, B=1T)

2. **Thermal de Broglie wavelength** — quantum length scale:
       λ = ℏ/√(2πmkT)
   When λ > interparticle spacing: quantum statistics matter

3. **Future** — decoherence and dissipation (Lindblad master equation)

### State Variables (tracked per simulation frame)
| Variable | Symbol | Units | Description |
|----------|--------|-------|-------------|
| Wavefunction | ψ(x,y,z,t) | m^(-3/2) | Complex amplitude |
| Probability density | \|ψ\|² | m^(-3) | Observable |
| Phase | arg(ψ) | radians | Interference effects |
| Energy expectation | ⟨E⟩ | J / eV | ⟨ψ\|H\|ψ⟩ |
| Momentum expectation | ⟨p⟩ | kg·m/s | ⟨ψ\|-iℏ∇\|ψ⟩ |
| Temperature | T | K | Thermal population |
| Magnetic field | B | T | Along z-axis |
| Velocity | v = ⟨p⟩/m | m/s | Drift velocity |

---

## Architecture

```
quantum-diffusion-simulator/
├── physics/                    # Core physics (Python + Julia + Rust)
│   ├── utils/constants.py      # ALL physical constants, LorenzFrame, QuantumState
│   ├── solvers/tdse_solver.py  # TDSE split-operator solver
│   ├── fields/                 # Magnetic, electric, gravitational fields
│   └── particles/              # Particle definitions
│
├── simulation/                 # Precomputation pipeline
│   ├── precompute/             # Simulation runners → HDF5/NPZ output
│   ├── data/                   # Gitignored — run scripts to generate
│   └── configs/                # YAML parameter configs
│
├── visualization/              # React + Three.js (Vercel)
│   └── src/components/         # QuantumScene.tsx — main 3D renderer
│
├── docs/                       # Derivations, citations, scientific notes
└── tests/                      # Validation against analytical solutions
```

## Data Flow
```
YAML config
    ↓
Python TDSE solver (physics/)
    ↓
HDF5 / NPZ output (simulation/data/)
    ↓
Three.js point cloud renderer (visualization/)
    ↓
Vercel deployment
```

---

## Numerical Methods

### TDSE Solver: Split-Operator (Trotter-Suzuki)
Factorizes time evolution operator:
    U(dt) ≈ exp(-iT·dt/2ℏ) · exp(-iV·dt/ℏ) · exp(-iT·dt/2ℏ)

Properties:
- Exactly unitary: norm preserved to machine precision
- Accuracy: O(dt²) globally, O(dt³) per step
- Kinetic term T: diagonal in momentum space → apply via FFT
- Potential term V: diagonal in position space → apply directly

Stability condition: dt << 1/ωc (cyclotron period)
At B=1T: T_cyclotron ≈ 35.7 fs → use dt ≤ 1e-18 s

### Grid Resolution Guidelines
| Purpose | Grid | Box size |
|---------|------|----------|
| Quick test | 32³ | 60 nm |
| Standard | 64³ | 80 nm |
| High quality | 128³ | 120 nm |
| Publication | 256³ | 160 nm |

Box size should be ≥ 3×lB in each dimension to avoid boundary artifacts.

### Known Issues / TODOs
- [ ] Magnetic cross-term in `tdse_solver.py` step() is simplified — needs full Peierls phase implementation
- [ ] Absorbing boundary conditions not yet implemented (wavepacket reflection at walls)
- [ ] Z-direction kinetic phase uses simplified FFT approach — verify against Crank-Nicolson

---

## Validation Targets
Before any new feature is merged, verify these benchmarks:

| Test | Expected | Tolerance |
|------|----------|-----------|
| Landau E₀ at B=1T | 0.05788 meV | < 1% |
| Landau E₁ - E₀ | 0.11576 meV | < 1% |
| Wavefunction norm | 1.000000 | < 1e-6 per step |
| H atom ground state | -13.6 eV | < 0.1% |
| H₂ bond length | 0.074 nm | < 2% |

Run: `python tests/validate_landau.py` before committing physics changes.

---

## Tech Stack
| Layer | Tech |
|-------|------|
| Physics solver | Python 3.11+, NumPy, SciPy |
| High-perf compute | Julia (DifferentialEquations.jl), Rust (ndarray) |
| Data format | HDF5 (.h5) via h5py, NPZ fallback |
| Frontend | React 18, TypeScript 5, Three.js |
| Shaders | GLSL — volumetric point cloud rendering |
| Deployment | Vercel (frontend) |
| Version control | GitHub (public) |

Install physics deps: `pip install -r physics/requirements.txt`
Install frontend deps: `cd visualization && npm install`
Run precompute: `python simulation/precompute/landau_precompute.py --B 1.0 --n 0`

---

## Roadmap Summary
- **Phase 1** (current): Electron in B field → Landau levels → 3D visualization ← WE ARE HERE
- **Phase 2**: Hydrogen atom orbitals, H₂ bonding/antibonding, temperature effects
- **Phase 3**: Chemical reaction simulator, radical pairs, magnetic field effects on chemistry
- **Phase 4**: Quantum-to-classical transition, Wigner function, MHD preview
- **Phase 5**: Stellar convection, planetary atmosphere emergence

Full roadmap: see ROADMAP.md

---

## Scientific References
- Landau & Lifshitz, *Quantum Mechanics* §112 (charged particle in magnetic field)
- Griffiths, *Introduction to Quantum Mechanics* Ch. 4 (hydrogen atom)
- Feit, Fleck & Steiger (1982), J. Comput. Phys. 47:412 — split-operator method
- Cohen-Tannoudji, *Quantum Mechanics* Vol. II (perturbation theory, magnetic effects)
- Press et al., *Numerical Recipes* Ch. 19 (PDE methods)
