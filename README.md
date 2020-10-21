# qmje.github.io

# Explains the function of each of the files in this directory 

# 1d_ho.py -- Solves the 1D quantum harmonic oscillator using a 
# spectral [Fourier split-step] method. Plots i) density |\psi|^2 
# sampled during imaginary time propagation, ii) comparison between  
# analytical and numerical solution, iii) realtime space-time
# density plot

# igt_2d_hpc.py -- Computes the ground state of a trapped, rotating
# 2D condensate. Here the rotation comprises a rigid as well as nonlinear
# rotation. The numerical method is a basic finite difference method. The
# code saves the output as a npz Python data file for later processing.

# vortex_tracker.py -- Sorts the identical phase defects [vortices] present in a 
# 2D wavefunction \psi(x,y) into the correct order as a function of time, t. 
# Looks for vortex at t = t + 1 with shortest distance from current vortex.
# Produces (x,y,t) trajectory plot of the phase defects dynamics in the trap.
