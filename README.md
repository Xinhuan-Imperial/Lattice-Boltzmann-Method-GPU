# Lattice-Boltzmann-Method-GPU
This program is a GPU CUDA version D3Q19 BGK Lattice Boltzmann Method Computational Fluid Dynamics solver to simulate steady/unsteady 3D single-phase flows. Compared with CPU serial code, this GPU code is more than 200 times faster and has same accuracy. To run it, a NVIDIA GPU with CUDA Toolkit is a must. It contains a lecture notes, the author's PhD thesis (Chapter 4 contains a detailed description about implementation of LBM), three Matlab tools (MyCrustOpen, fitNormal and smoothpatch) for geometry preprocessing, four simulation cases:

1. Lid_driven_cavity: steady Laminar flow
2. Poiseulle_flow: steady Laminar flow
3. bifurcation: steady Laminar flow
4. curved vessel: unsteady Laminar flow with arbitrary geometry. This case is a good example to start if you want to use your geometry and inlet/outlet boundary conditions.

For geometry preprecessing, it is important to generate 3D uniform grid, and then denote each grid with an integer number - there are 6 kinds of computational grids: inlet 2, outlet 3, fluid 4, wall 1, the outer neighbour of wall -1, not used 0. The 3D computational grid is denoted by the 6 integers from -1 to 4:
1. For inlet and outlet grids, we use the non-equilibrium extrapolation method
2. For wall grids, we use the half-way bounce back method
3. For fluid grids, SRT LBM streaming and collision happens
4. For outer neighbour grids of wall, those grids are only used for easier implementation of wall half-way bounce back.
5. For not used grids, they are abondoned in function index_transform to reduce the large memory consumption.

To use the code, you must cite:
1. The author's PhD thesis
2. Zhou, X., Vincent, P., Riemer, K., Zhou, Xiaowei, Leow, C. H., Tang, M. X*. Measurement Augmented 3D Lattice Boltzmann Flow Simulation for Convergence Acceleration and Uncertainty Suppression. (Submitted to the Computers & Fluids).
