"""
simulation/precompute/landau_precompute.py

Precomputation pipeline for Landau level simulations.

Runs TDSE solver and saves output to HDF5 format for
consumption by the Three.js visualization frontend.

Output schema (HDF5):
    /metadata/
        B           — magnetic field [T]
        T           — temperature [K]
        n_landau    — Landau level index
        dt          — time step [s]
        n_steps     — total steps
        grid/Nx, Ny, Nz, Lx, Ly, Lz
    /frames/
        /t          — time values array [n_frames]
        /psi_real   — Re(ψ) [n_frames, Nx, Ny, Nz]
        /psi_imag   — Im(ψ) [n_frames, Nx, Ny, Nz]
        /probability — |ψ|² [n_frames, Nx, Ny, Nz]
        /energy     — ⟨E⟩ [n_frames]
        /px, /py, /pz — ⟨p⟩ components [n_frames]
    /observables/
        /landau_energies — analytical Landau level energies
        /thermal_populations — Boltzmann populations at T
"""

import sys
import os
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from physics.utils.constants import (
    LorenzFrame, landau_energy, E_CHARGE, M_ELECTRON
)
from physics.solvers.tdse_solver import TDSESolver

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Warning: h5py not installed. Output will be NumPy .npz format.")


def run_landau_simulation(config: dict, output_path: str):
    """
    Run full Landau level precomputation and save to file.
    
    Args:
        config: Simulation config dict
        output_path: Path to output .h5 or .npz file
    """
    # ── Parse config ──────────────────────────────────────────────
    grid_cfg = config['grid']
    sim_cfg  = config['simulation']
    phys_cfg = config['physics']
    
    print(f"Quantum Diffusion Simulation — Landau Level Precompute")
    print(f"{'='*55}")
    print(f"Grid: {grid_cfg['Nx']}³  B={phys_cfg['B']}T  T={phys_cfg['T']}K  n={phys_cfg['n_landau']}")
    
    # ── Build frame ───────────────────────────────────────────────
    frame = LorenzFrame(
        Nx=grid_cfg['Nx'], Ny=grid_cfg['Ny'], Nz=grid_cfg['Nz'],
        Lx=grid_cfg['Lx_nm'] * 1e-9,
        Ly=grid_cfg['Ly_nm'] * 1e-9,
        Lz=grid_cfg['Lz_nm'] * 1e-9,
    )
    
    # ── Build solver ──────────────────────────────────────────────
    solver = TDSESolver(
        frame=frame,
        B=phys_cfg['B'],
        T_temperature=phys_cfg['T'],
        dt=sim_cfg['dt'],
    )
    
    # ── Initial state ─────────────────────────────────────────────
    n = phys_cfg['n_landau']
    if phys_cfg['T'] > 0 and phys_cfg.get('thermal_mix', False):
        psi_init = solver.thermal_initial_state(n_max=phys_cfg.get('n_max_thermal', 5))
        print(f"Initial state: thermal mixture at T={phys_cfg['T']}K")
    else:
        psi_init = solver.landau_ground_state(n=n)
        print(f"Initial state: Landau level n={n}")
    
    # ── Run simulation ────────────────────────────────────────────
    n_steps    = sim_cfg['n_steps']
    save_every = sim_cfg['save_every']
    n_frames   = n_steps // save_every
    
    print(f"Running {n_steps} steps, saving every {save_every} → {n_frames} frames")
    
    # Pre-allocate arrays
    t_arr    = np.zeros(n_frames, dtype=np.float64)
    prob_arr = np.zeros((n_frames, frame.Nx, frame.Ny, frame.Nz), dtype=np.float32)
    psi_real = np.zeros_like(prob_arr)
    psi_imag = np.zeros_like(prob_arr)
    energy_arr = np.zeros(n_frames, dtype=np.float64)
    
    t0 = time.time()
    frame_idx = 0
    
    for state in solver.evolve(psi_init, n_steps, save_every):
        if frame_idx >= n_frames:
            break
        
        t_arr[frame_idx]    = state.t
        prob_arr[frame_idx] = state.probability_density.astype(np.float32)
        psi_real[frame_idx] = state.psi.real.astype(np.float32)
        psi_imag[frame_idx] = state.psi.imag.astype(np.float32)
        
        elapsed = time.time() - t0
        pct = frame_idx / n_frames * 100
        print(f"  Frame {frame_idx+1}/{n_frames} ({pct:.0f}%)  t={state.t:.2e}s  elapsed={elapsed:.1f}s")
        
        frame_idx += 1
    
    print(f"\nSimulation complete in {time.time()-t0:.1f}s")
    
    # ── Analytical observables ────────────────────────────────────
    n_levels = 10
    landau_energies = np.array([landau_energy(i, phys_cfg['B']) / E_CHARGE
                                for i in range(n_levels)])  # in eV
    
    # ── Save output ───────────────────────────────────────────────
    print(f"Saving to {output_path}...")
    
    if HDF5_AVAILABLE and output_path.endswith('.h5'):
        _save_hdf5(output_path, {
            'config': config,
            'frame': frame,
            't_arr': t_arr,
            'prob_arr': prob_arr,
            'psi_real': psi_real,
            'psi_imag': psi_imag,
            'energy_arr': energy_arr,
            'landau_energies': landau_energies,
            'x': frame.x * 1e9,  # convert to nm for frontend
            'y': frame.y * 1e9,
            'z': frame.z * 1e9,
        })
    else:
        # Fallback to npz
        npz_path = output_path.replace('.h5', '.npz')
        np.savez_compressed(npz_path,
            t=t_arr,
            probability=prob_arr,
            psi_real=psi_real,
            psi_imag=psi_imag,
            energy=energy_arr,
            landau_energies=landau_energies,
            x_nm=frame.x * 1e9,
            y_nm=frame.y * 1e9,
            z_nm=frame.z * 1e9,
        )
        print(f"Saved to {npz_path}")
    
    print("Done.")
    return prob_arr, t_arr


def _save_hdf5(path: str, data: dict):
    """Save simulation data to HDF5 format."""
    import h5py
    config = data['config']
    frame = data['frame']
    
    with h5py.File(path, 'w') as f:
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['B_tesla']     = config['physics']['B']
        meta.attrs['T_kelvin']    = config['physics']['T']
        meta.attrs['n_landau']    = config['physics']['n_landau']
        meta.attrs['dt_seconds']  = config['simulation']['dt']
        meta.attrs['n_steps']     = config['simulation']['n_steps']
        meta.attrs['description'] = "Landau level wavefunction evolution"
        
        grid = meta.create_group('grid')
        grid.attrs['Nx'] = frame.Nx
        grid.attrs['Ny'] = frame.Ny
        grid.attrs['Nz'] = frame.Nz
        
        # Coordinate axes (in nm)
        coords = f.create_group('coordinates')
        coords.create_dataset('x_nm', data=data['x'])
        coords.create_dataset('y_nm', data=data['y'])
        coords.create_dataset('z_nm', data=data['z'])
        
        # Time frames
        frames = f.create_group('frames')
        frames.create_dataset('t_seconds',   data=data['t_arr'], compression='gzip')
        frames.create_dataset('probability', data=data['prob_arr'], compression='gzip', dtype='float32')
        frames.create_dataset('psi_real',    data=data['psi_real'], compression='gzip', dtype='float32')
        frames.create_dataset('psi_imag',    data=data['psi_imag'], compression='gzip', dtype='float32')
        frames.create_dataset('energy_ev',   data=data['energy_arr'] / E_CHARGE, compression='gzip')
        
        # Analytical observables
        obs = f.create_group('observables')
        obs.create_dataset('landau_energies_ev', data=data['landau_energies'])
        
    print(f"HDF5 saved: {path}")


# ── Default config ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'grid': {
        'Nx': 48, 'Ny': 48, 'Nz': 48,
        'Lx_nm': 60.0, 'Ly_nm': 60.0, 'Lz_nm': 60.0,
    },
    'physics': {
        'B': 1.0,          # Tesla
        'T': 0.0,          # Kelvin (0 = ground state)
        'n_landau': 0,     # Landau level index
        'thermal_mix': False,
        'n_max_thermal': 5,
    },
    'simulation': {
        'dt': 1e-18,       # seconds
        'n_steps': 1000,
        'save_every': 50,  # → 20 frames
    }
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute Landau level simulation')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--output', type=str, default='simulation/data/landau_n0_B1T.npz')
    parser.add_argument('--B', type=float, default=1.0, help='Magnetic field [T]')
    parser.add_argument('--T', type=float, default=0.0, help='Temperature [K]')
    parser.add_argument('--n', type=int, default=0, help='Landau level index')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG.copy()
        config['physics']['B'] = args.B
        config['physics']['T'] = args.T
        config['physics']['n_landau'] = args.n
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_landau_simulation(config, args.output)
