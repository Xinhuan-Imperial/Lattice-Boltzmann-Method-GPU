# Lattice-Boltzmann-Method-GPU
This program is a GPU CUDA version D3Q19 BGK Lattice Boltzmann Method Computational Fluid Dynamics solver to simulate steady/unsteady 3D single-phase Newtonian flows, where moving boundary and body force (i.e., gravity) are NOT considered. Compared with CPU serial code, this GPU code is >250 times faster with NVIDIA Geforce 2080ti and >140times with NVIDIA Geforce 1050ti, and has same accuracy. To run it, a NVIDIA GPU with CUDA Toolkit is a must. It contains: A. a LBM lecture notes;B. the author's PhD thesis (Chapter 4 contains a detailed description about implementation of LBM);C. a paper about uniform Cartesian grid generation(CartGen: Robust, efﬁcient and easy to implementuniform/octree/embedded boundary Cartesiangrid generator);D. three Matlab tools (MyCrustOpen, fitNormal and smoothpatch) for surface reconstruction/smoothing; E. a Matlab function geo_preprocess to generate uniform Cartesian non-body-fitting grid; F. four simulation cases:

1. Lid_driven_cavity: steady Laminar flow
2. Poiseulle_flow: steady Laminar flow
3. bifurcation: steady Laminar flow with arbitrary geometry. This case is a good example to start if you want to use your geometry and steady inlet/outlet boundary conditions.
4. curved vessel: unsteady Laminar flow with arbitrary geometry. This case is a good example to start if you want to use your geometry and unsteady inlet/outlet boundary conditions.

To use your own geometry, geometry preprecessing is important and we must generate 3D uniform Cartesian non-body-fitted computational grid to fit the geometry and then denote each grid with an integer number - there are six kinds of computational grids (denoted by the 6 integers from -1 to 4): inlet 2, outlet 3, fluid 4, wall 1, not used 0, outer neighbour of wall -1. A Matlab function geo_preprocess is provided to generate such 3D uniform Cartesian grids (refer to the paper 'CartGen: Robust,...' for the algorithm used) and decide each grid's mask:
1. For inlet and outlet grids, we use the non-equilibrium extrapolation method
2. For wall grids, we use the half-way bounce back method
3. For fluid grids, SRT LBM streaming and collision happens
4. For not used grids, they are out of the flow domain while not adjacent to wall grids, and are abondoned in C function index_transform to reduce the large memory usage.
5. For outer neighbour grids of wall, those grids are out of the flow domain and only used for easier implementation of wall half-way bounce back. As wall grids pull information from their 19 neighbour grids. A C function geo_pre is used to define if a grid is outer neighbour of wall, because this requires for-loop and inefficient using Matlab.


To use the code, you must cite:
1. Zhou, X., Ultrasound Imaging Augmented 3D Flow Reconstruction and Computational Fluid Dynamics Simulation. (The author's PhD thesis)
2. Zhou, X., Vincent, P., Riemer, K., Zhou, Xiaowei, Leow, C. H., Tang, M. X*. Measurement Augmented 3D Lattice Boltzmann Flow Simulation for Convergence Acceleration and Uncertainty Suppression. (Submitted to the Computers & Fluids)
