"""
physics/utils/constants.py

Fundamental physical constants and coordinate system utilities.
All values in SI units unless otherwise noted.

Lorenz Gauge Framework:
    The Lorenz gauge condition ensures:
        ∇·A + (1/c)(∂φ/∂t) = 0
    
    This makes the wave equations for φ and A symmetric and
    Lorentz-covariant — essential for relativistic consistency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ─── Fundamental Constants ────────────────────────────────────────────────────

HBAR        = 1.054571817e-34   # ℏ, reduced Planck constant [J·s]
H_PLANCK    = 6.62607015e-34    # h, Planck constant [J·s]
C_LIGHT     = 2.99792458e8      # c, speed of light [m/s]
E_CHARGE    = 1.602176634e-19   # e, elementary charge [C]
M_ELECTRON  = 9.1093837015e-31  # mₑ, electron mass [kg]
M_PROTON    = 1.67262192369e-27 # mp, proton mass [kg]
K_BOLTZMANN = 1.380649e-23      # kB, Boltzmann constant [J/K]
EPSILON_0   = 8.8541878128e-12  # ε₀, vacuum permittivity [F/m]
MU_0        = 1.25663706212e-6  # μ₀, vacuum permeability [H/m]
A_BOHR      = 5.29177210903e-11 # a₀, Bohr radius [m]
E_HARTREE   = 4.3597447222071e-18  # Eh, Hartree energy [J]
ALPHA_FINE  = 7.2973525693e-3   # α, fine structure constant [dimensionless]

# ─── Derived Quantities ───────────────────────────────────────────────────────

def cyclotron_frequency(B: float, mass: float = M_ELECTRON) -> float:
    """
    Cyclotron frequency: ωc = eB/m
    
    The fundamental frequency of circular motion of a charged
    particle in a uniform magnetic field. Landau level spacing
    is exactly ℏωc.
    
    Args:
        B: Magnetic field strength [T]
        mass: Particle mass [kg], default electron
    
    Returns:
        ωc in [rad/s]
    """
    return E_CHARGE * B / mass


def landau_energy(n: int, B: float, mass: float = M_ELECTRON) -> float:
    """
    Landau level energy: Eₙ = ℏωc(n + 1/2)
    
    Discrete energy levels of a charged particle in a uniform
    magnetic field. The zero-point energy ℏωc/2 has no classical analog.
    
    Args:
        n: Landau level index (0, 1, 2, ...)
        B: Magnetic field strength [T]
        mass: Particle mass [kg]
    
    Returns:
        Energy in [J]
    """
    wc = cyclotron_frequency(B, mass)
    return HBAR * wc * (n + 0.5)


def magnetic_length(B: float) -> float:
    """
    Magnetic length: lB = √(ℏ/eB)
    
    The natural length scale for electrons in a magnetic field.
    Sets the spatial extent of Landau level wavefunctions.
    Typical value: ~26nm at B=1T.
    
    Args:
        B: Magnetic field strength [T]
    
    Returns:
        lB in [m]
    """
    return np.sqrt(HBAR / (E_CHARGE * B))


def thermal_de_broglie(T: float, mass: float = M_ELECTRON) -> float:
    """
    Thermal de Broglie wavelength: λ = ℏ / √(2πmkT)
    
    The quantum length scale below which quantum effects dominate.
    When λ >> interparticle spacing, quantum statistics matter.
    
    Args:
        T: Temperature [K]
        mass: Particle mass [kg]
    
    Returns:
        λ in [m]
    """
    return HBAR / np.sqrt(2 * np.pi * mass * K_BOLTZMANN * T)


def boltzmann_population(energies: np.ndarray, T: float) -> np.ndarray:
    """
    Thermal population via Boltzmann distribution: Pₙ ∝ exp(-Eₙ/kT)
    
    Args:
        energies: Array of energy levels [J]
        T: Temperature [K]
    
    Returns:
        Normalized probability array
    """
    if T <= 0:
        raise ValueError("Temperature must be positive")
    beta = 1.0 / (K_BOLTZMANN * T)
    weights = np.exp(-beta * (energies - energies.min()))
    return weights / weights.sum()


# ─── Lorenz Gauge Coordinate System ──────────────────────────────────────────

@dataclass
class LorenzFrame:
    """
    3D coordinate frame in Lorenz gauge.
    
    The Lorenz gauge condition:
        ∇·A + (1/c²)(∂φ/∂t) = 0
    
    makes the potentials (φ, A) satisfy symmetric wave equations:
        □²φ = -ρ/ε₀
        □²A = -μ₀J
    
    where □² = ∇² - (1/c²)∂²/∂t² is the d'Alembertian operator.
    
    This frame tracks both spatial coordinates and the electromagnetic
    4-potential (φ/c, Ax, Ay, Az) for full covariance.
    """
    # Spatial grid parameters
    Nx: int = 64
    Ny: int = 64
    Nz: int = 64
    Lx: float = 20 * A_BOHR   # Box size in x [m]
    Ly: float = 20 * A_BOHR
    Lz: float = 20 * A_BOHR

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        
        # Build coordinate arrays
        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
        self.z = np.linspace(-self.Lz/2, self.Lz/2, self.Nz)
        
        # 3D meshgrid (for field computations)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

    def vector_potential_symmetric_gauge(self, B: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Magnetic vector potential in symmetric gauge for uniform B along z:
            A = B/2 × (-y, x, 0)
        
        Satisfies Lorenz gauge: ∇·A = 0 (Coulomb gauge coincides here
        for static fields — both gauges equivalent for magnetostatics).
        
        Args:
            B: Magnetic field strength [T]
        
        Returns:
            (Ax, Ay, Az) arrays on the 3D grid
        """
        Ax = -0.5 * B * self.Y
        Ay =  0.5 * B * self.X
        Az = np.zeros_like(self.X)
        return Ax, Ay, Az

    def gradient(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradient of scalar field f on grid."""
        gx = np.gradient(f, self.dx, axis=0)
        gy = np.gradient(f, self.dy, axis=1)
        gz = np.gradient(f, self.dz, axis=2)
        return gx, gy, gz

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Compute Laplacian ∇²f on grid using central differences."""
        d2x = np.gradient(np.gradient(f, self.dx, axis=0), self.dx, axis=0)
        d2y = np.gradient(np.gradient(f, self.dy, axis=1), self.dy, axis=1)
        d2z = np.gradient(np.gradient(f, self.dz, axis=2), self.dz, axis=2)
        return d2x + d2y + d2z

    def kinetic_momentum_operator(
        self,
        psi: np.ndarray,
        Ax: np.ndarray,
        Ay: np.ndarray,
        Az: np.ndarray
    ) -> np.ndarray:
        """
        Kinetic momentum operator: (p - eA)²ψ / 2m
        
        The minimal coupling prescription replaces p → p - eA,
        ensuring gauge invariance of the Schrödinger equation.
        
        Returns the kinetic energy term of Hψ.
        """
        # Momentum operator components: p = -iℏ∇
        # (p - eA)ψ component by component
        gpx, gpy, gpz = self.gradient(psi)
        
        # Kinetic momentum: π = p - eA
        # (p - eA)ψ = -iℏ∇ψ - eA·ψ
        pi_psi_x = -1j * HBAR * gpx - E_CHARGE * Ax * psi
        pi_psi_y = -1j * HBAR * gpy - E_CHARGE * Ay * psi
        pi_psi_z = -1j * HBAR * gpz - E_CHARGE * Az * psi

        # (p - eA)²ψ = ∇·(π·ψ) — apply operator again
        def div_component(f, axis, d):
            return np.gradient(f, d, axis=axis)

        result = (
            div_component(-1j * HBAR * gpx - E_CHARGE * Ax * psi, 0, self.dx) +
            div_component(-1j * HBAR * gpy - E_CHARGE * Ay * psi, 1, self.dy) +
            div_component(-1j * HBAR * gpz - E_CHARGE * Az * psi, 2, self.dz)
        )

        # Hmm — need to apply (p-eA) twice properly via chain rule
        # Full expansion: (p-eA)²ψ = p²ψ - e(p·A + A·p)ψ + e²A²ψ
        p2_psi = -HBAR**2 * self.laplacian(psi)
        
        # In Coulomb/Lorenz gauge for static uniform B: ∇·A = 0
        # So p·A = A·p and cross term = -2eA·pψ = -2eA·(-iℏ∇ψ)
        A2 = Ax**2 + Ay**2 + Az**2
        
        cross = -2 * E_CHARGE * (
            Ax * (-1j * HBAR * gpx) +
            Ay * (-1j * HBAR * gpy) +
            Az * (-1j * HBAR * gpz)
        )
        
        e2A2_psi = E_CHARGE**2 * A2 * psi
        
        return (p2_psi + cross + e2A2_psi) / (2 * M_ELECTRON)


# ─── State Vector ─────────────────────────────────────────────────────────────

@dataclass
class QuantumState:
    """
    Full quantum state container for a single particle.
    
    Tracks wavefunction and all derived observables at a given
    simulation timestep. Designed for serialization to HDF5.
    """
    psi: np.ndarray          # Complex wavefunction ψ(x,y,z)
    t: float                 # Current time [s]
    T: float                 # Temperature [K]
    B: float                 # Magnetic field strength [T]
    
    @property
    def probability_density(self) -> np.ndarray:
        """|ψ|² — probability density"""
        return np.abs(self.psi)**2
    
    @property
    def phase(self) -> np.ndarray:
        """arg(ψ) — wavefunction phase"""
        return np.angle(self.psi)
    
    def expectation_energy(self, H_psi: np.ndarray) -> float:
        """⟨E⟩ = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩"""
        norm = np.sum(self.probability_density)
        return np.real(np.sum(np.conj(self.psi) * H_psi)) / norm
    
    def expectation_momentum(self, frame: LorenzFrame) -> Tuple[float, float, float]:
        """⟨p⟩ = ⟨ψ|-iℏ∇|ψ⟩"""
        gpx, gpy, gpz = frame.gradient(self.psi)
        norm = np.sum(self.probability_density)
        px = np.real(np.sum(np.conj(self.psi) * (-1j * HBAR * gpx))) / norm
        py = np.real(np.sum(np.conj(self.psi) * (-1j * HBAR * gpy))) / norm
        pz = np.real(np.sum(np.conj(self.psi) * (-1j * HBAR * gpz))) / norm
        return px, py, pz
    
    def normalize(self) -> None:
        """Normalize ψ so that ∫|ψ|²dV = 1"""
        norm = np.sqrt(np.sum(self.probability_density))
        if norm > 0:
            self.psi /= norm


if __name__ == "__main__":
    # Quick sanity check
    B = 1.0  # 1 Tesla
    wc = cyclotron_frequency(B)
    lB = magnetic_length(B)
    
    print(f"B = {B} T")
    print(f"Cyclotron frequency ωc = {wc:.4e} rad/s")
    print(f"Magnetic length lB = {lB*1e9:.3f} nm")
    print(f"Landau level E0 = {landau_energy(0, B)/E_CHARGE*1000:.4f} meV")
    print(f"Landau level E1 = {landau_energy(1, B)/E_CHARGE*1000:.4f} meV")
    print(f"Level spacing = {HBAR*wc/E_CHARGE*1000:.4f} meV")
    
    T = 300  # Room temperature
    lam = thermal_de_broglie(T)
    print(f"\nT = {T} K")
    print(f"Thermal de Broglie λ = {lam*1e9:.4f} nm")
    print(f"Bohr radius a0 = {A_BOHR*1e9:.4f} nm")
    print(f"Quantum effects {'dominate' if lam > A_BOHR else 'are subdominant'} at this T")
