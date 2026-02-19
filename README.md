# Quantum Diffusion Simulation
### A 3D Open-Source Scientific Visualization Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://www.rust-lang.org/)

> *Simulating quantum mechanical behavior from subatomic particles to emergent atmospheric phenomena — scientific, stunning and publicly open.*

---

## Vision

This project models a hierarchy of emergent physical complexity, each layer arising from the one beneath it:

```
Magnetism → Subatomic Behavior → Chemistry → Water/Fluid Dynamics → Convection → Atmosphere
```

Beginning at the quantum scale — electrons in magnetic fields, Landau levels, wavefunction diffusion — and building upward toward hydrogen bonding, molecular chemistry, fluid dynamics, and ultimately planetary atmospheric convection. Every layer is grounded in real physics, precomputed for scientific accuracy, and rendered in interactive 3D.

---

## Architecture Overview

```
quantum-diffusion-sim/
├── physics/                    # Core physics engine (Python + Julia + Rust)
│   ├── solvers/                # TDSE, Schrödinger, diffusion solvers
│   ├── fields/                 # Magnetic, electric, gravitational field definitions
│   ├── particles/              # Particle definitions: electron, proton, hydrogen
│   └── utils/                  # Constants, coordinate transforms, Lorenz frames
│
├── simulation/                 # Precomputation pipeline
│   ├── precompute/             # Simulation runners — outputs HDF5/JSON data
│   ├── data/                   # Precomputed simulation datasets
│   └── configs/                # Simulation parameter configs (YAML)
│
├── visualization/              # React + Three.js frontend (Vercel deployment)
│   ├── src/
│   │   ├── components/         # React scene components
│   │   ├── shaders/            # GLSL shaders for wavefunction rendering
│   │   └── utils/              # Data loading, animation, interpolation
│   └── public/                 # Static assets, precomputed data files
│
├── docs/                       # Scientific documentation, derivations, citations
├── scripts/                    # Build, deploy, data pipeline scripts
└── tests/                      # Unit + integration tests per layer
```

---

## Physics Roadmap

### Phase 1 — Quantum Foundation
**Lorenz Coordinate System & 3D Scaffolding**
- Establish covariant coordinate framework using Lorenz gauge condition: `∇·A + (1/c)(∂φ/∂t) = 0`
- This ensures electromagnetic potentials are physically consistent and gauge-invariant
- All particle positions, momenta, and fields expressed in this framework

**Electron in a Magnetic Field — Landau Levels**
- Solve the time-dependent Schrödinger equation (TDSE) with magnetic vector potential:
  ```
  iℏ ∂ψ/∂t = [1/2m (p - eA)² + V] ψ
  ```
- Magnetic field B along z-axis → vector potential A = (-By/2, Bx/2, 0) in symmetric gauge
- Energy eigenstates become discrete Landau levels: `Eₙ = ℏωc(n + 1/2)`
- 3D probability density |ψ|² rendered as volumetric point cloud with ring-like distributions
- **Key observables:** energy eigenvalues, cyclotron frequency ωc = eB/m, wavefunction extent

**State variables tracked per simulation:**
- `ψ(x,y,z,t)` — complex wavefunction
- `|ψ|²` — probability density
- `⟨E⟩` — expectation value of energy
- `⟨p⟩` — expectation value of momentum
- `T` — temperature (thermal de Broglie wavelength λ = ℏ/√(2πmkT))
- `v_drift` — drift velocity in crossed E×B fields

---

### Phase 2 — Hydrogen & Atomic Structure
**Hydrogen Atom Wavefunctions**
- Full 3D hydrogen orbitals: ψₙₗₘ(r,θ,φ) in spherical coordinates
- Spherical harmonics Yₗₘ rendered volumetrically
- Radial probability distributions P(r) = r²|Rₙₗ(r)|²

**Hydrogen Pairing / H₂ Molecule**
- Two-center wavefunction: bonding vs antibonding orbitals
- Born-Oppenheimer approximation for nuclear motion
- Electron correlation effects
- Visualization of molecular orbital formation

**Temperature Effects on Atomic States**
- Thermal population of energy levels via Boltzmann distribution: `Pₙ ∝ exp(-Eₙ/kT)`
- Spectral line broadening (Doppler + pressure broadening)
- Ionization probability as function of temperature

---

### Phase 3 — Chemistry & Molecular Dynamics
**Chemical Reaction Simulator**
- Transition state theory: reaction rates `k = A·exp(-Ea/kT)`
- Potential energy surfaces (PES) in 3D
- Reaction coordinate visualization
- Electron transfer reactions — Marcus theory
- Role of magnetic fields in radical pair reactions (cryptochrome / bird navigation analog)

**Water Molecule**
- Full quantum treatment of H₂O: bond angles, dipole moment
- Hydrogen bonding network emergence
- Phase transitions as function of T and P

---

### Phase 4 — Fluid Dynamics & Diffusion
**Classical Diffusion**
- Fick's laws: `∂C/∂t = D∇²C`
- Stochastic particle trajectories (Langevin dynamics)
- Diffusion coefficients as function of temperature

**Quantum-to-Classical Transition**
- Decoherence: wavefunction collapse in thermal environment
- Wigner function phase-space representation
- Emergence of classical random walk from quantum diffusion

**Magnetohydrodynamics (MHD) Preview**
- Lorentz force on conducting fluid: `F = J × B`
- Alfvén waves
- Foundation for stellar convection layer

---

### Phase 5 — Stellar Convection & Atmosphere
**Stellar Convection Zone**
- Mixing length theory for convective energy transport
- Temperature gradient: `dT/dr` vs adiabatic gradient
- Magnetic field amplification via dynamo mechanism
- Ocean-atmosphere energy exchange model

**Planetary Atmosphere Emergence**
- How stellar convection drives atmospheric chemistry
- Photodissociation, recombination
- Molecular escape velocity as function of T and planetary mass

---

## Scientific Grounding

All simulations reference published literature. Key references:

- Landau & Lifshitz, *Quantum Mechanics: Non-Relativistic Theory*
- Griffiths, *Introduction to Quantum Mechanics*
- Cohen-Tannoudji, *Quantum Mechanics* Vol. I & II
- Feynman, Leighton & Sands, *The Feynman Lectures on Physics*
- Strang & Fix, *An Analysis of the Finite Element Method* (numerical methods)
- Press et al., *Numerical Recipes* (computational methods)

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Physics Solver | Python 3.11+, NumPy, SciPy, QuTiP |
| High-Performance Compute | Julia (DifferentialEquations.jl), Rust (ndarray) |
| Data Format | HDF5 (.h5) via h5py, JSON for configs |
| API/Bridge | FastAPI (Python) |
| Frontend | React 18, TypeScript 5, Three.js r155 |
| Shaders | GLSL (custom volumetric rendering) |
| Deployment | Vercel (frontend), Python anywhere / fly.io (API) |
| Version Control | GitHub (public, open source) |

---

## Getting Started

### Physics Engine
```bash
cd physics
pip install -r requirements.txt
python solvers/tdse_solver.py --config ../simulation/configs/electron_landau.yaml
```

### Visualization
```bash
cd visualization
npm install
npm run dev
```

### Run a Precomputation
```bash
python scripts/precompute_landau.py --output simulation/data/landau_levels_B1T.h5
```

---

## Contributing

This is an open science project. Contributions welcome across all layers:
- Physics accuracy improvements and peer review
- Numerical solver optimizations
- Visualization enhancements
- Documentation and derivations

Please open an issue before submitting large PRs. Scientific claims should cite sources.

---

## License

MIT License — free to use, modify, and distribute. Attribution appreciated.

---

*Built by [Elliot Telford](https://elliottelford.com) — astrophysicist, game developer, systems thinker.*
