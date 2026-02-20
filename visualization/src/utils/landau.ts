/**
 * visualization/src/utils/landau.ts
 *
 * Client-side computation of analytical Landau level wavefunctions.
 * Mirrors physics/utils/constants.py — same formulae, implemented in JS.
 *
 * ψₙ(x,y,z) = Cₙ · ((x+iy)/lB)ⁿ · exp(-(x²+y²)/4lB²) · exp(ikz·z)
 *
 * All SI units internally; outputs in nm for visualization.
 */

const HBAR = 1.054571817e-34        // ℏ [J·s]
const E_CHARGE = 1.602176634e-19    // e [C]
const M_ELECTRON = 9.1093837015e-31 // mₑ [kg]

export function magneticLength(B: number): number {
  return Math.sqrt(HBAR / (E_CHARGE * B))  // lB [m]
}

export function cyclotronFrequency(B: number): number {
  return E_CHARGE * B / M_ELECTRON  // ωc [rad/s]
}

export function landauEnergy(n: number, B: number): number {
  return HBAR * cyclotronFrequency(B) * (n + 0.5)  // Eₙ [J]
}

export interface LandauGrid {
  Nx: number
  Ny: number
  Nz: number
  L_nm: number
  probability: Float32Array
  psi_real: Float32Array
  psi_imag: Float32Array
  x_nm: Float32Array
  y_nm: Float32Array
  z_nm: Float32Array
  energy_ev: number
  B_tesla: number
  n_landau: number
  lB_nm: number
  wc: number
}

/**
 * Compute the nth Landau level wavefunction analytically on a 3D grid.
 *
 * @param n       Landau level index (0 = ground state)
 * @param B       Magnetic field [T]
 * @param Nx/Ny/Nz  Grid points per axis
 * @param L_nm    Box half-width [nm] (box spans -L to +L)
 */
export function computeLandauState(
  n: number,
  B: number,
  Nx = 32,
  Ny = 32,
  Nz = 32,
  L_nm = 80,
): LandauGrid {
  const lB = magneticLength(B)
  const lB_nm = lB * 1e9
  const wc = cyclotronFrequency(B)

  // Coordinate axes
  const x_nm = new Float32Array(Nx)
  const y_nm = new Float32Array(Ny)
  const z_nm = new Float32Array(Nz)

  for (let i = 0; i < Nx; i++) x_nm[i] = -L_nm / 2 + (i / (Nx - 1)) * L_nm
  for (let i = 0; i < Ny; i++) y_nm[i] = -L_nm / 2 + (i / (Ny - 1)) * L_nm
  for (let i = 0; i < Nz; i++) z_nm[i] = -L_nm / 2 + (i / (Nz - 1)) * L_nm

  const N = Nx * Ny * Nz
  const psi_real = new Float32Array(N)
  const psi_imag = new Float32Array(N)
  const probability = new Float32Array(N)

  // Compute ψₙ = (x+iy)ⁿ · exp(-(x²+y²)/2) in dimensionless units
  // Dimensionless coords: x̃ = x/lB, ỹ = y/lB
  let normSq = 0.0

  for (let ix = 0; ix < Nx; ix++) {
    const xt = x_nm[ix] / lB_nm  // dimensionless x̃
    for (let iy = 0; iy < Ny; iy++) {
      const yt = y_nm[iy] / lB_nm  // dimensionless ỹ
      const r2 = xt * xt + yt * yt

      // Gaussian envelope: exp(-r²/2) where r is in units of lB
      // This matches exp(-(x²+y²)/4lB²) → in dimensionless: exp(-r̃²/2)
      const gaussian = Math.exp(-r2 / 2)

      // Angular momentum factor: (x+iy)ⁿ = rⁿ·exp(inφ)
      // Write in polar: magnitude = r^n, phase = n*atan2(y,x)
      let re: number, im: number
      if (n === 0) {
        re = gaussian
        im = 0
      } else {
        const r = Math.sqrt(r2)
        const phi = Math.atan2(yt, xt)
        const mag = Math.pow(r, n) * gaussian
        re = mag * Math.cos(n * phi)
        im = mag * Math.sin(n * phi)
      }

      for (let iz = 0; iz < Nz; iz++) {
        // kz = 0 → z part = exp(0) = 1 (stationary in z)
        const idx = ix * Ny * Nz + iy * Nz + iz
        psi_real[idx] = re
        psi_imag[idx] = im
        normSq += re * re + im * im
      }
    }
  }

  // Normalize ψ so Σ|ψ|² = 1
  const norm = Math.sqrt(normSq)
  if (norm > 0) {
    for (let i = 0; i < N; i++) {
      psi_real[i] /= norm
      psi_imag[i] /= norm
      probability[i] = psi_real[i] * psi_real[i] + psi_imag[i] * psi_imag[i]
    }
  }

  const energy_ev = landauEnergy(n, B) / E_CHARGE

  return {
    Nx, Ny, Nz, L_nm,
    probability, psi_real, psi_imag,
    x_nm, y_nm, z_nm,
    energy_ev,
    B_tesla: B,
    n_landau: n,
    lB_nm,
    wc,
  }
}
