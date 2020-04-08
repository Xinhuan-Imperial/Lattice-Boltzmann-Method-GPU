/**
 * Copyright by the author.  All rights reserved.
 *
 * Please refer to the author (zhouxinhuan0205@126.com or x.zhou16@ic.ac.uk)
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation is strictly prohibited.
 *
 * This software is Lattice Boltzmann Modelling for lid driven cavity flow
 * based on NVIDIA GPU.
 * http://www.bg.ic.ac.uk/research/m.tang/ulis/
 * Boundary Condition: non-equlibrium extrapolation
 * Wall: hall-way bounce back
 * Lid:moving at 0.15m/s, other 5 lids are static 
 * Author: Dr. Xinhuan Zhou, PhD of Imperial College London
 * Date: 12 December 2018
 * 
**/

#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <direct.h>
#include <stdlib.h>
#include <sstream>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

using namespace std;

string output_direc = "./out";
string filename = "lid";
string logname;

const int Q = 19, NX = 64, NY = 64, NZ = 64;//D3Q19
const float CH = 0.0000655737f, C_rho = 1060.f, C_U = 2.4705f;
float *h_ux, *h_uy, *h_uz, *h_rho,*d_dst, *d_scr,*h_dst, *h_scr, *d_tmp,*d_ux,*d_uy,*d_uz,*d_rho,*d_velsum;
int *h_geo,*d_geo;
float C_pre = C_rho*C_U*C_U,u_max=0.15f/C_U;
int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y,bz=1+(NZ-1)/BLOCK_Z;
int NLATTICE=bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z;
float tau = 0.55f;
		
__global__ void update(int NLATTICE,int* __restrict__ d_geo,float* __restrict__ d_scr,float* __restrict__ d_dst,float* __restrict__ d_ux,float* __restrict__ d_uy,float* __restrict__ d_uz,float* __restrict__ d_rho,float tau) {
	//ghost layer=0, wall=1, lid=2, fluid=3
	//non-equilibrium extrapolation of wall
	float feq[19],fnq[19];

	float tmp_ux, tmp_uy, tmp_uz,tmp_rho;
	int i,j,k,ind,ind2,koff;
	int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y;
	
	for(koff=BLOCK_Z-1;koff>=0;koff--){
		i=threadIdx.x+blockIdx.x*blockDim.x;
		j=threadIdx.y+blockIdx.y*blockDim.y;
		k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			
		ind=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
				
		//streaming: pull from neighbour cells
				
		if (d_geo[ind] ==1) {
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[1] = d_scr[NLATTICE*1+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[2] = d_scr[NLATTICE*2+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[3] = d_scr[NLATTICE*3+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[4] = d_scr[NLATTICE*4+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[5] = d_scr[NLATTICE*5+ind2];
	
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[6] = d_scr[NLATTICE*6+ind2];
	
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[7] = d_scr[NLATTICE*7+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[8] = d_scr[NLATTICE*8+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[9] = d_scr[NLATTICE*9+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[10] = d_scr[NLATTICE*10+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[11] = d_scr[NLATTICE*11+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[12] = d_scr[NLATTICE*12+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[13] = d_scr[NLATTICE*13+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[14] = d_scr[NLATTICE*14+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[15] = d_scr[NLATTICE*15+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[16] = d_scr[NLATTICE*16+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[17] = d_scr[NLATTICE*17+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[18] = d_scr[NLATTICE*18+ind2];
	
			d_scr[NLATTICE*1+ind] = fnq[2];
			d_scr[NLATTICE*2+ind] = fnq[1];
			d_scr[NLATTICE*3+ind] = fnq[4];
			d_scr[NLATTICE*4+ind] = fnq[3];
			d_scr[NLATTICE*5+ind] = fnq[6];
			d_scr[NLATTICE*6+ind] = fnq[5];
			d_scr[NLATTICE*7+ind] = fnq[10];
			d_scr[NLATTICE*8+ind] = fnq[9];
			d_scr[NLATTICE*9+ind] = fnq[8];
			d_scr[NLATTICE*10+ind] = fnq[7];
			d_scr[NLATTICE*11+ind] = fnq[14];
			d_scr[NLATTICE*12+ind] = fnq[13];
			d_scr[NLATTICE*13+ind] = fnq[12];
			d_scr[NLATTICE*14+ind] = fnq[11];
			d_scr[NLATTICE*15+ind] = fnq[18];
			d_scr[NLATTICE*16+ind] = fnq[17];
			d_scr[NLATTICE*17+ind] = fnq[16];
			d_scr[NLATTICE*18+ind] = fnq[15];
		}
		
		if (d_geo[ind] ==3) {
			fnq[0]=d_scr[ind];

			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[1] = d_scr[NLATTICE+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[2] = d_scr[NLATTICE*2+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[3] = d_scr[NLATTICE*3+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[4] = d_scr[NLATTICE*4+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[5] = d_scr[NLATTICE*5+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[6] = d_scr[NLATTICE*6+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[7] = d_scr[NLATTICE*7+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[8] = d_scr[NLATTICE*8+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[9] = d_scr[NLATTICE*9+ind2];			
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[10] = d_scr[NLATTICE*10+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[11] = d_scr[NLATTICE*11+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[12]= d_scr[NLATTICE*12+ind2];
		
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[13]= d_scr[NLATTICE*13+ind2];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[14]= d_scr[NLATTICE*14+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[15]= d_scr[NLATTICE*15+ind2];			
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[16]= d_scr[NLATTICE*16+ind2];		
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[17]= d_scr[NLATTICE*17+ind2];	
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			fnq[18]= d_scr[NLATTICE*18+ind2];		

			//update
			tmp_rho = 0.f;
			for (k = 0; k < Q; k++) {
				tmp_rho = tmp_rho + fnq[k];
			}
			tmp_ux = (fnq[1] - fnq[2] + fnq[7] + fnq[8] - fnq[9] - fnq[10] + fnq[11] + fnq[12] - fnq[13] - fnq[14]) / tmp_rho;
			tmp_uy = (fnq[3] - fnq[4] + fnq[7] - fnq[8] + fnq[9] - fnq[10] + fnq[15] - fnq[16] + fnq[17] - fnq[18]) / tmp_rho;
			tmp_uz = (fnq[5] - fnq[6] + fnq[11] - fnq[12] + fnq[13] - fnq[14] + fnq[15] + fnq[16] - fnq[17] - fnq[18]) / tmp_rho;

			d_ux[ind]=tmp_ux;
			d_uy[ind]=tmp_uy;
			d_uz[ind]=tmp_uz;
			d_rho[ind]=tmp_rho;			
			//colliding
			
			feq[0] = tmp_rho/3.0f * (1.0f - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy -1.5f* tmp_uz*tmp_uz);
			feq[1] = tmp_rho /18.0f * (1.0f + 3.0f* tmp_ux + 3.0f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy -1.5f* tmp_uz*tmp_uz);
			feq[2] = tmp_rho /18.0f * (1.0f - 3.0f* tmp_ux + 3.0f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy -1.5f* tmp_uz*tmp_uz);
			feq[3] = tmp_rho /18.0f * (1.0f + 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			feq[4] = tmp_rho /18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			feq[5] = tmp_rho /18.0f * (1.0f + 3.0f* tmp_uz + 3.0f*tmp_uz*tmp_uz - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy);
			feq[6] = tmp_rho /18.0f* (1.0f - 3.0f* tmp_uz + 3.0f*tmp_uz*tmp_uz - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy);
			feq[7] = tmp_rho /36.0f* (1.0f + 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy -1.5f* tmp_uz*tmp_uz);
			feq[8] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			feq[9] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_ux) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			feq[10] = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			feq[11] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux + tmp_uz) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_ux*tmp_uz-1.5f* tmp_uy*tmp_uy);
			feq[12] = tmp_rho /36.0f* (1.0f + 3.0f* (tmp_ux - tmp_uz) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_ux*tmp_uz-1.5f* tmp_uy*tmp_uy);
			feq[13] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_ux) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_ux*tmp_uz-1.5f* tmp_uy*tmp_uy);
			feq[14] = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uz) + 3.0f* tmp_ux*tmp_ux + 3.0*tmp_uz*tmp_uz + 9.0f*tmp_ux*tmp_uz -1.5f* tmp_uy*tmp_uy);
			feq[15] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz- 1.5f*tmp_ux*tmp_ux);
			feq[16] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			feq[17] = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			feq[18] = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			
			d_dst[ind] = fnq[0] - (fnq[0] - feq[0]) / tau;
			d_dst[NLATTICE*1+ind] = fnq[1] - (fnq[1] - feq[1]) / tau;
			d_dst[NLATTICE*2+ind] = fnq[2] - (fnq[2] - feq[2]) / tau;
			d_dst[NLATTICE*3+ind] = fnq[3] - (fnq[3] - feq[3]) / tau;
			d_dst[NLATTICE*4+ind] = fnq[4] - (fnq[4] - feq[4]) / tau;
			d_dst[NLATTICE*5+ind] = fnq[5] - (fnq[5] - feq[5]) / tau;
			d_dst[NLATTICE*6+ind] = fnq[6] - (fnq[6] - feq[6]) / tau;
			d_dst[NLATTICE*7+ind] = fnq[7] - (fnq[7] - feq[7]) / tau;
			d_dst[NLATTICE*8+ind] = fnq[8] - (fnq[8] - feq[8]) / tau;
			d_dst[NLATTICE*9+ind] = fnq[9] - (fnq[9] - feq[9]) / tau;
			d_dst[NLATTICE*10+ind] = fnq[10] - (fnq[10] - feq[10]) / tau;
			d_dst[NLATTICE*11+ind] = fnq[11] - (fnq[11] - feq[11]) / tau;
			d_dst[NLATTICE*12+ind] = fnq[12] - (fnq[12] - feq[12]) / tau;
			d_dst[NLATTICE*13+ind] = fnq[13] - (fnq[13] - feq[13]) / tau;
			d_dst[NLATTICE*14+ind] = fnq[14] - (fnq[14] - feq[14]) / tau;
			d_dst[NLATTICE*15+ind] = fnq[15] - (fnq[15] - feq[15]) / tau;
			d_dst[NLATTICE*16+ind] = fnq[16] - (fnq[16] - feq[16]) / tau;
			d_dst[NLATTICE*17+ind] = fnq[17] - (fnq[17] - feq[17]) / tau;
			d_dst[NLATTICE*18+ind] = fnq[18] - (fnq[18] - feq[18]) / tau;
		}
	}
}

__global__ void boundary_stream(int NLATTICE,int* __restrict__ d_geo,float* __restrict__ d_dst,float* __restrict__ d_ux,float* __restrict__ d_uy,float* __restrict__ d_uz,float* __restrict__ d_rho,float tau,float C_U) {
	//ghost layer=0, wall=1, lid=2, fluid=3,data augmentation=4
	//non-equilibrium extrapolation of wall
	
	float feq;
	float u_max=0.15f/C_U;
	float tmp,tmp_ux, tmp_uy, tmp_uz,tmp_rho;
	int i,j,k,ind,ind2,koff;
	int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y;
	
	for(koff=BLOCK_Z-1;koff>=0;koff--){
		i=threadIdx.x+blockIdx.x*blockDim.x;
		j=threadIdx.y+blockIdx.y*blockDim.y;
		k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;

		ind=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
		
		//non-equilibrium extrapolation of lid
		if (d_geo[ind] == 2) {
			//4
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			tmp_rho=d_rho[ind2];
			tmp_ux=d_ux[ind2];
			tmp_uy=d_uy[ind2];
			tmp_uz=d_uz[ind2];
			feq = tmp_rho/18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f* tmp_ux*tmp_ux -1.5f*tmp_uz*tmp_uz);
			tmp = tmp_rho/18.0f* (1.0f - 1.5f*u_max*u_max);
			d_dst[NLATTICE*4+ind] = tmp + (d_dst[NLATTICE*4+ind2] - feq)*(1.0f - 1.0f / tau);
		
			//8
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			tmp_rho=d_rho[ind2];
			tmp_ux=d_ux[ind2];
			tmp_uy=d_uy[ind2];
			tmp_uz=d_uz[ind2];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy - 1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho/36.0f  * (1.0f - 1.5f* u_max*u_max);
			d_dst[NLATTICE*8+ind] = tmp + (d_dst[NLATTICE*8+ind2] - feq)*(1.0f - 1.0f / tau);
		
			//10
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			tmp_rho=d_rho[ind2];
			tmp_ux=d_ux[ind2];
			tmp_uy=d_uy[ind2];
			tmp_uz=d_uz[ind2];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy -1.5f*tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f  * (1.0f - 1.5f* u_max*u_max);
			d_dst[NLATTICE*10+ind] = tmp + (d_dst[NLATTICE*10+ind2] - feq)*(1.0f - 1.0f / tau);
			
			//16
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			tmp_rho=d_rho[ind2];
			tmp_ux=d_ux[ind2];
			tmp_uy=d_uy[ind2];
			tmp_uz=d_uz[ind2];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f* tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f*u_max + 3.0f*u_max*u_max);
			d_dst[NLATTICE*16+ind] = tmp + (d_dst[NLATTICE*16+ind2] - feq)*(1.0f - 1.0f / tau);
		
			//18
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			tmp_rho=d_rho[ind2];
			tmp_ux=d_ux[ind2];
			tmp_uy=d_uy[ind2];
			tmp_uz=d_uz[ind2];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f* tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f*u_max + 3.0f*u_max*u_max);
			d_dst[NLATTICE*18+ind] = tmp + (d_dst[NLATTICE*18+ind2] - feq)*(1.0f - 1.0f / tau);
		}
	}
}

__global__ void calc_vel_square(float*  __restrict__ d_velsum,float*  __restrict__ d_ux,float*  __restrict__ d_uy,float*  __restrict__ d_uz,int NLATTICE){
	int const ind = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(ind<NLATTICE){
		d_velsum[ind]=sqrt(powf(d_ux[ind],2.f)+powf(d_uy[ind],2.f)+powf(d_uz[ind],2.f));
	}
}

void geo_pre() {
	//ghost layer=0, wall=1, lid=2, fluid=3,data augmentation=4
	int ind;
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				h_geo[ind] = 0;
			}
		}
	}
	for (int z = 1; z < NZ-1; z++) {
		for (int y = 1; y < NY-1; y++) {
			for (int x = 1; x < NX-1; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				h_geo[ind] = 1;
			}
		}
	}
	for (int x = 2; x < NX - 2; x++) {
		for (int y = 2; y < NY - 2; y++) {
			for (int z = 2; z < NZ - 2; z++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				h_geo[ind] = 3;
			}
		}
	}
	int y=NY-2;
	for (int x = 1; x < NX - 1; x++) {
		for (int z = 1; z < NZ - 1; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			h_geo[ind] = 2;
		}
	}
}

void initialize() {
	float feq[19];
	float  tmp_rho, tmp_ux, tmp_uy, tmp_uz, ux2, uy2, uz2, uxyz2, uxy2, uxz2, uyz2, uxy, uxz, uyz;
	float wi[19] = { 1.0f / 3.0f, 1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f };
	int ind;

	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				h_ux[ind]=0.0f;
				h_uy[ind]=0.0f;
				h_uz[ind]=0.0f;
				h_rho[ind]=1.0f;
			}
		}
	}
	
	int y;					
	for (int z = 0; z < NZ; z++) {
		for (int x = 0; x < NX; x++) {
			y=NY-1;
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			h_uz[ind]=u_max;
			y=NY-2;
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			h_uz[ind]=u_max;
		}
	}	
	
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				tmp_rho = h_rho[ind];
				tmp_ux = h_ux[ind];
				tmp_uy = h_uy[ind];
				tmp_uz = h_uz[ind];
				ux2 = tmp_ux*tmp_ux;
				uy2 = tmp_uy*tmp_uy;
				uz2 = tmp_uz*tmp_uz;
				uxyz2 = ux2 + uy2 + uz2;
				uxy2 = ux2 + uy2;
				uxz2 = ux2 + uz2;
				uyz2 = uy2 + uz2;
				uxy = 2.0f*tmp_ux*tmp_uy;
				uxz = 2.0f*tmp_ux*tmp_uz;
				uyz = 2.0f*tmp_uy*tmp_uz;

				feq[0] = tmp_rho * wi[0] * (1.0f - 1.5f*uxyz2);
				feq[1] = tmp_rho * wi[1] * (1.0f + 3.0f* tmp_ux + 4.5f*ux2 - 1.5f*uxyz2);
				feq[2] = tmp_rho * wi[2] * (1.0f - 3.0f* tmp_ux + 4.5f*ux2 - 1.5f*uxyz2);
				feq[3] = tmp_rho * wi[3] * (1.0f + 3.0f* tmp_uy + 4.5f*uy2 - 1.5f*uxyz2);
				feq[4] = tmp_rho * wi[4] * (1.0f - 3.0f* tmp_uy + 4.5f*uy2 - 1.5f*uxyz2);
				feq[5] = tmp_rho * wi[5] * (1.0f + 3.0f* tmp_uz + 4.5f*uz2 - 1.5f*uxyz2);
				feq[6] = tmp_rho * wi[6] * (1.0f - 3.0f* tmp_uz + 4.5f*uz2 - 1.5f*uxyz2);
				feq[7] = tmp_rho * wi[7] * (1.0f + 3.0f* (tmp_ux + tmp_uy) + 4.5f* (uxy2 + uxy) - 1.5f*uxyz2);
				feq[8] = tmp_rho * wi[8] * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 4.5f* (uxy2 - uxy) - 1.5f* uxyz2);
				feq[9] = tmp_rho * wi[9] * (1.0f + 3.0f* (tmp_uy - tmp_ux) + 4.5f* (uxy2 - uxy) - 1.5f* uxyz2);
				feq[10] = tmp_rho * wi[10] * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 4.5f* (uxy2 + uxy) - 1.5f* uxyz2);
				feq[11] = tmp_rho * wi[11] * (1.0f + 3.0f* (tmp_ux + tmp_uz) + 4.5f* (uxz2 + uxz) - 1.5f* uxyz2);
				feq[12] = tmp_rho * wi[12] * (1.0f + 3.0f* (tmp_ux - tmp_uz) + 4.5f* (uxz2 - uxz) - 1.5f* uxyz2);
				feq[13] = tmp_rho * wi[13] * (1.0f + 3.0f* (tmp_uz - tmp_ux) + 4.5f* (uxz2 - uxz) - 1.5f* uxyz2);
				feq[14] = tmp_rho * wi[14] * (1.0f - 3.0f* (tmp_ux + tmp_uz) + 4.5f* (uxz2 + uxz) - 1.5f* uxyz2);
                feq[15] = tmp_rho * wi[15] * (1.0f + 3.0f* (tmp_uy + tmp_uz) + 4.5f* (uyz2 + uyz) - 1.5f* uxyz2);
				feq[16] = tmp_rho * wi[16] * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 4.5f* (uyz2 - uyz) - 1.5f* uxyz2);
				feq[17] = tmp_rho * wi[17] * (1.0f + 3.0f* (tmp_uy - tmp_uz) + 4.5f* (uyz2 - uyz) - 1.5f* uxyz2);
				feq[18] = tmp_rho * wi[18] * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 4.5f* (uyz2 + uyz) - 1.5f* uxyz2);

				for (int k = 0; k<Q; k++) {
					h_dst[NLATTICE*k+ind] = feq[k];
					h_scr[NLATTICE*k+ind] = feq[k];
				}
			}
		}
	}
}

void outputSave(string output_direc,int t) {
	stringstream ss;
	int ind;
	ss << output_direc << '/' << "lid" << '_' << t << ".vtk";
	string datafilename = ss.str();
	ofstream ofs(datafilename);
	ofs<<"# vtk DataFile Version 2.0"<<endl;
	ofs<<"<-- LBM flow with UIV acceleration, http://www.bg.ic.ac.uk/research/m.tang/ulis/ -->"<<endl;
	ofs<<"ASCII"<<endl;
	ofs<<"DATASET STRUCTURED_POINTS"<<endl;
	ofs << "DIMENSIONS " << NX-4 << ' ' << NY-4 << ' ' << NZ-4 << endl;
	ofs<< "SPACING "<< CH<<' '<< CH<<' '<< CH<<endl;
	ofs<<"ORIGIN "<< round(NX / 2-1)*CH<<' '<< round(NY / 2-1)*CH<<' '<< .0<<endl;
	ofs<<"POINT_DATA  "<<(NX-4)*(NY-4)*(NZ-4)<<endl;

	ofs << "VECTORS VELOCITY float" << endl;
	for (int z = 2; z < NZ-2; z ++) {
		for (int y = 2; y < NY-2; y ++) {
			for (int x = 2; x < NX-2; x ++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				ofs << h_ux[ind] * C_U << " ";
				ofs << h_uy[ind] * C_U << " ";
				ofs << h_uz[ind] * C_U << " ";
			}
		}
	}
	
	ofs.close();
}

int main(int argc,const char **argv) {
	float residual,milli,sum_current,sum_next;
	float tol=1e-6f;
	int stag_max=50,k=0,tol_count=0,max_it=10000,time_save=500;
	logname=output_direc+'/'+ "CONVERGENCE.log";
	ofstream logfile(logname);	
	
	dim3 dimGrid(bx,by,bz);
	dim3 dimBlock(BLOCK_X,BLOCK_Y,1);
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//allocate memory
	h_geo=(int*)malloc(sizeof(int)*NLATTICE);
	h_ux=(float*)malloc(sizeof(float)*NLATTICE);
	h_uy=(float*)malloc(sizeof(float)*NLATTICE);
	h_uz=(float*)malloc(sizeof(float)*NLATTICE);
	h_rho=(float*)malloc(sizeof(float)*NLATTICE);
	h_dst=(float*)malloc(sizeof(float)*NLATTICE*Q);
	h_scr=(float*)malloc(sizeof(float)*NLATTICE*Q);
	
	cudaMalloc(&d_ux,sizeof(float)*NLATTICE);
	cudaMalloc(&d_uy,sizeof(float)*NLATTICE);
	cudaMalloc(&d_uz,sizeof(float)*NLATTICE);
	cudaMalloc(&d_rho,sizeof(float)*NLATTICE);	
	cudaMalloc(&d_geo,sizeof(int)*NLATTICE);
	cudaMalloc(&d_dst,sizeof(float)*NLATTICE*Q);
	cudaMalloc(&d_scr,sizeof(float)*NLATTICE*Q);
	cudaMalloc(&d_velsum,sizeof(float)*NLATTICE);
	thrust::device_ptr<float> dev_d_velsum(d_velsum);

	geo_pre();
	initialize();
	
 	cudaMemcpy(d_geo,h_geo,sizeof(int)*NLATTICE,cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst,h_dst,sizeof(float)*NLATTICE*Q,cudaMemcpyHostToDevice);
	cudaMemcpy(d_scr,h_scr,sizeof(float)*NLATTICE*Q,cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	sum_current=0.f;
	while(k<=max_it&&tol_count<=stag_max){
		update<<<dimGrid,dimBlock>>>(NLATTICE,d_geo,d_scr,d_dst,d_ux,d_uy,d_uz,d_rho,tau);
		cudaDeviceSynchronize();
		
		boundary_stream<<<dimGrid,dimBlock>>>(NLATTICE,d_geo,d_dst,d_ux,d_uy,d_uz,d_rho,tau,C_U);
		cudaDeviceSynchronize();

		calc_vel_square<<<(NLATTICE+128-1)/128,128>>>(d_velsum,d_ux,d_uy,d_uz,NLATTICE);
		cudaDeviceSynchronize();
		sum_next=thrust::reduce(thrust::device,dev_d_velsum,dev_d_velsum+NLATTICE,0.f, thrust::plus<float>());
		
		d_tmp=d_scr;
		d_scr = d_dst;
		d_dst=d_tmp;
		
		residual=abs(sum_next-sum_current)/sum_next;		
		if(k%time_save==0){
			cudaDeviceSynchronize();
			cudaMemcpy(h_ux,d_ux,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uy,d_uy,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uz,d_uz,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rho,d_rho,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);			
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milli,start,stop);
			cout<<"ITERATION # " << k << ", collapse time: "<< milli <<" ms, residual:"<< residual<<endl;
			logfile<<residual<<endl;
			outputSave(output_direc,k);
		}
		k++;
		sum_current=sum_next;
		if(residual<=tol)tol_count++;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli,start,stop);
	cout << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << "#LATTICE" << NLATTICE << endl;
	cout<<"Residual is "<<residual<<endl;
	logfile << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << "#LATTICE" << NLATTICE << " ERROR IS"<<residual<<endl;
	logfile.close();
	cudaMemcpy(h_ux,d_ux,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_uy,d_uy,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_uz,d_uz,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho,d_rho,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);	
	outputSave(output_direc,k);
	
	//free memory
	free(h_ux);
	free(h_uy);
	free(h_uz);
	free(h_rho);
	free(h_dst);
	free(h_scr);
	free(h_geo);
	cudaFree(d_dst);
	cudaFree(d_scr);
	cudaFree(d_geo);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_uz);
	cudaFree(d_rho);	
	cudaDeviceReset();
	system("pause");
	return 0;
}
