Steady Poiseulle flow in a circular vessle, whose diameter is 0.004m and maximum velocity is 0.15m/s. Boundary condition: given velocity and pressure distribution at the inlet/outlet. To compile, in Windows command terminal or Linux shell, enter: nvcc Poiseulle.cu -o Poiseulle

physical units:
temperature: 36 degree

UMAX=0.15 m/s --maximum velocity at the center

H=0.004 m --The diameter and length of the vessel

rho=1060 kg/m3

kinematic viscosity:2.7e-6 m2/s

Re=222.2

converter:

CH=6.557377049180328e-05

C_rho=1060

C_T=4.2469e-05

C_U=1.5441

nondimensional:

U_MAX=UMAX/C_U

rho_bar=1

tau=0.58

BOUNDARY CONDITION:

Inlet: velocity & pressure

Outlet: velocity & pressure
