# Lattice-Boltzmann-Method-GPU
This program is a GPU CUDA version D3Q19 BGK Lattice Boltzmann Method Computational Fluid Dynamics solver to simulate steady/unsteady 3D single-phase flows. Compared with CPU serial code, this GPU code is more than 200 times faster and has same accuracy. To run it, a NVIDIA GPU with CUDA Toolkit is a must. It contains four cases:

1. Lid_driven_cavity: steady Laminar flow
2. Poiseulle_flow: steady Laminar flow
3. bifurcation: steady Laminar flow
4. curved vessel: unsteady Laminar flow with arbitrary geometry. This case is a good example to start if you want to use your geometry and inlet/outlet boundary conditions.
