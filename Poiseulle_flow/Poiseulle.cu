/**
 * Copyright by the author.  All rights reserved.
 *
 * Please refer to the author (zhouxinhuan0205@126.com or x.zhou16@ic.ac.uk)
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation is strictly prohibited.
 *
 * This software is Lattice Boltzmann Modelling for straight vessel flow using SRT model
 * based on GPU, writen by Xinhuan Zhou from Ultrasound Lab for Imaging and Sensing
 * http://www.bg.ic.ac.uk/research/m.tang/ulis/
 * Boundary Condition: non-equlibrium extrapolation
 * Wall: hall-way bounce back.
 * Inlet:velocity & pressure; Outlet:velocity & pressure
 * Author: Xinhuan Zhou, PhD of Imperial College London
 * Date: 21 January 2019
 * 
**/

#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <algorithm>
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
#define BLOCK_Z 4

using namespace std;

const int Q = 19, NX = 64, NY = 64, NZ = 64;//D3Q19
const float CH = 0.0000655737f, C_rho = 1060.f, C_U = 1.5441f,tau=0.58f;
float wi[19] = { 1.0f / 3.0f, 1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f };

float *h_ux, *h_uy, *h_uz, *h_rho,*d_dst, *d_scr,*h_dst, *h_scr, *d_tmp,*d_ux,*d_uy,*d_uz,*d_rho,*d_velsum;
int *h_geo,*d_geo,*h_index,*d_index;
float C_pre = C_rho*C_U*C_U,u_max=0.15f/C_U;
int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y,bz=1+(NZ-1)/BLOCK_Z;
dim3 dimGrid(bx,by,bz);
dim3 dimBlock(BLOCK_X,BLOCK_Y,1);
int NLATTICE=0;
texture<int,cudaTextureType1D,cudaReadModeElementType> d_index_text;
texture<int,cudaTextureType1D,cudaReadModeElementType> d_geo_text;

void geo_pre() {
	//not useful=0,ghost cells=-1, wall=1,left end=2,right end=3,fluid=4
	float dist;
	int minn4, minx, miny, minz, minn8, ind, ind2, i, j, k;
	float radius = (NX - 1) / 2.0f;
	float center_x = (NX - 1) / 2.0f, center_z = (NZ - 1) / 2.0f;
	int ***flag;
	flag = (int ***)malloc(NX * sizeof(*flag));

	for (int x = 0; x < NX; x++) {
		flag[x] = (int **)malloc(NY * sizeof(**flag));
		for (int y = 0; y < NY; y++) {
			flag[x][y] = (int *)malloc(NZ * sizeof(***flag));
		}
	}

	//ghost layer
	for (int x = 0; x < NX; x++) {
		for (int y = 0; y < NY; y++) {
			for (int z = 0; z < NZ; z++) {
				ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				h_geo[ind] = 0;
				flag[x][y][z] = 0;
			}
		}
	}

	//to binary matrix
	for (int x = 0; x < NX; x++) {
		for (int y = 1; y < NY - 1; y++) {
			for (int z = 0; z < NZ; z++) {
				dist = sqrt(pow(x - center_x, 2) + pow(z - center_z, 2));
				if (dist <= radius) {
					ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
					flag[x][y][z] = 1;
					h_geo[ind] = 1;
				}
			}
		}
	}

	//distance transform, fluid=4
	for (int t = 0; t < 3; t++) {
		for (int x = 1; x < NX - 1; x++) {
			for (int y = 2; y < NY - 2; y++) {
				for (int z = 1; z < NZ - 1; z++) {
					ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
					minx = min(flag[x + 1][y][z], flag[x - 1][y][z]);
					miny = min(flag[x][y - 1][z], flag[x][y + 1][z]);
					minz = min(flag[x][y][z - 1], flag[x][y][z + 1]);
					minn4 = min(minx, miny);
					minn8 = min(minn4, minz);
					h_geo[ind] = h_geo[ind] + minn8;
				}
			}
		}
	}

	int y = 1;
	//distance transform, left end=2
	for (int x = 1; x < NX - 1; x++) {
		for (int z = 1; z < NZ - 1; z++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			minx = min(flag[x + 1][y][z], flag[x - 1][y][z]);
			minz = min(flag[x][y][z - 1], flag[x][y][z + 1]);
			minn4 = min(minx, minz);
			h_geo[ind] = h_geo[ind] + minn4;
		}
	}

	y = NY - 2;
	//distance transform, right end=3
	for (int t = 0; t < 2; t++) {
		for (int x = 1; x < NX - 1; x++) {
			for (int z = 1; z < NZ - 1; z++) {
				ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				minx = min(flag[x + 1][y][z], flag[x - 1][y][z]);
				minz = min(flag[x][y][z - 1], flag[x][y][z + 1]);
				minn4 = min(minx, minz);
				h_geo[ind] = h_geo[ind] + minn4;
			}
		}
	}

	free(flag);

	for (int z = 1; z < NZ-1; z++) {
		for (int y = 1; y < NY-1; y++) {
			for (int x = 1; x < NX-1; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				if (h_geo[ind]==1||h_geo[ind]==2||h_geo[ind]==3){
					//1
					i=x+1;
					j=y;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//2
					i=x-1;
					j=y;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//3
					i=x;
					j=y+1;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//4
					i=x;
					j=y-1;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//5
					i=x;
					j=y;
					k=z+1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//6
					i=x;
					j=y;
					k=z-1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//7
					i=x+1;
					j=y+1;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//8
					i=x+1;
					j=y-1;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//9
					i=x-1;
					j=y+1;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//10
					i=x-1;
					j=y-1;
					k=z;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//11
					i=x+1;
					j=y;
					k=z+1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//12
					i=x+1;
					j=y;
					k=z-1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//13
					i=x-1;
					j=y;
					k=z+1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//14
					i=x-1;
					j=y;
					k=z-1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//15
					i=x;
					j=y+1;
					k=z+1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//16
					i=x;
					j=y-1;
					k=z+1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//17
					i=x;
					j=y+1;
					k=z-1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
					//18
					i=x;
					j=y-1;
					k=z-1;
					ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if(h_geo[ind2]==0)h_geo[ind2]=-1;
				}
			}
		}
	}
}

void index_transform(){
	int ind;
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				if (h_geo[ind]!=0){
					h_index[ind]=NLATTICE;
					NLATTICE++;
				}
				else h_index[ind]=-1;
			}
		}
	}
}

void initialize() {
	float center_x = (NX - 1) / 2.0f, center_z = (NZ - 1) / 2.0f;
	float  radius = (NX - 1) / 2.0f,tmp;
	int ind,idx;
	float feq[19];
	float  tmp_rho, tmp_ux, tmp_uy, tmp_uz;
	
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];
				if(idx>=0){
					h_ux[idx] = 0.0f;
					h_uy[idx] = 0.0f;
					h_uz[idx] = 0.0f;
					h_rho[idx] = 1.0f;
				}
			}
		}
	}
	
	int y=0;
	for (int x = 0; x<NX; x++) {
		for (int z = 0; z<NZ; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];
			if (idx>=0) {
				tmp=u_max*(1.0f-(pow(x-center_x,2)+pow(z-center_z,2))/pow(radius,2));
				h_uy[idx] =tmp;
			}
		}
	}
	
	y=1;
	for (int x = 0; x<NX; x++) {
		for (int z = 0; z<NZ; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];				
			if(idx>=0){
				tmp=u_max*(1.0f-(pow(x-center_x,2)+pow(z-center_z,2))/pow(radius,2));
				h_uy[idx] = tmp;
			}
		}
	}
	
	y=NY-1;
	for (int x = 0; x<NX; x++) {
		for (int z = 0; z<NZ; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];				
			if(idx>=0){
				tmp=u_max*(1.0f-(pow(x-center_x,2)+pow(z-center_z,2))/pow(radius,2));
				h_uy[idx] = tmp;
			}
		}
	}
	
	y=NY-2;
	for (int x = 0; x<NX; x++) {
		for (int z = 0; z<NZ; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];				
			if(idx>=0){
				tmp=u_max*(1.0f-(pow(x-center_x,2)+pow(z-center_z,2))/pow(radius,2));
				h_uy[idx] = tmp;
			}
		}
	}

	for (int x=0; x < NX; x++) {
		for (int y=0; y < NY; y++) {
			for (int z=0; z < NZ ; z++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];				
				if(idx>=0){
					tmp_rho = h_rho[idx];
					tmp_ux = h_ux[idx];
					tmp_uy = h_uy[idx];
					tmp_uz = h_uz[idx];

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
			
					for (int k = 0; k<Q; k++) {
						h_dst[NLATTICE*k+idx] = feq[k];
						h_scr[NLATTICE*k+idx] = feq[k];
					}
				}
			}
		}
	}
}

__global__ void update(int NLATTICE,float*  __restrict__ d_scr,float*  __restrict__ d_dst,float*  __restrict__ d_ux,float*  __restrict__ d_uy,float*  __restrict__ d_uz,float*  __restrict__ d_rho,float tau){
	float fnq[19],feq[19];
	int i,j,k,ind,ind2,idx,koff,geo_tmp;
	int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y;
	float tmp_rho,tmp_ux, tmp_uy, tmp_uz;
	
	for(koff=BLOCK_Z-1;koff>=0;koff--){
		i=threadIdx.x+blockIdx.x*blockDim.x;
		j=threadIdx.y+blockIdx.y*blockDim.y;
		k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
		ind=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
		idx=tex1Dfetch(d_index_text,ind);
		
		geo_tmp=tex1Dfetch(d_geo_text,ind);
		if (geo_tmp ==4) {
			fnq[0] = d_scr[idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[1] = d_scr[NLATTICE*1+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[2] = d_scr[NLATTICE*2+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[3] = d_scr[NLATTICE*3+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[4] = d_scr[NLATTICE*4+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[5] = d_scr[NLATTICE*5+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[6] = d_scr[NLATTICE*6+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[7] = d_scr[NLATTICE*7+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[8] = d_scr[NLATTICE*8+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[9] = d_scr[NLATTICE*9+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[10] = d_scr[NLATTICE*10+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[11] = d_scr[NLATTICE*11+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[12] = d_scr[NLATTICE*12+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[13] = d_scr[NLATTICE*13+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[14] = d_scr[NLATTICE*14+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[15] = d_scr[NLATTICE*15+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[16] = d_scr[NLATTICE*16+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[17] = d_scr[NLATTICE*17+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[18] = d_scr[NLATTICE*18+idx];

			
			tmp_rho = 0.f;
			for (int k = 0; k < Q; k++) {
				tmp_rho = tmp_rho + fnq[k];
			}

			tmp_ux = (fnq[1] - fnq[2] + fnq[7] + fnq[8] - fnq[9] - fnq[10] + fnq[11] + fnq[12] - fnq[13] - fnq[14]) / tmp_rho;
			tmp_uy = (fnq[3] - fnq[4] + fnq[7] - fnq[8] + fnq[9] - fnq[10] + fnq[15] - fnq[16] + fnq[17] - fnq[18]) / tmp_rho;
			tmp_uz = (fnq[5] - fnq[6] + fnq[11] - fnq[12] + fnq[13] - fnq[14] + fnq[15] + fnq[16] - fnq[17] - fnq[18]) / tmp_rho;

			idx=tex1Dfetch(d_index_text,ind);
			d_rho[idx] = tmp_rho;			
			d_ux[idx] = tmp_ux;
			d_uy[idx] = tmp_uy;
			d_uz[idx] = tmp_uz;			

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
			d_dst[idx] = fnq[0] - (fnq[0] - feq[0]) / tau;
			d_dst[NLATTICE*1+idx] = fnq[1] - (fnq[1] - feq[1]) / tau;
			d_dst[NLATTICE*2+idx] = fnq[2] - (fnq[2] - feq[2]) / tau;
			d_dst[NLATTICE*3+idx] = fnq[3] - (fnq[3] - feq[3]) / tau;
			d_dst[NLATTICE*4+idx] = fnq[4] - (fnq[4] - feq[4]) / tau;
			d_dst[NLATTICE*5+idx] = fnq[5] - (fnq[5] - feq[5]) / tau;
			d_dst[NLATTICE*6+idx] = fnq[6] - (fnq[6] - feq[6]) / tau;
			d_dst[NLATTICE*7+idx] = fnq[7] - (fnq[7] - feq[7]) / tau;
			d_dst[NLATTICE*8+idx] = fnq[8] - (fnq[8] - feq[8]) / tau;
			d_dst[NLATTICE*9+idx] = fnq[9] - (fnq[9] - feq[9]) / tau;
			d_dst[NLATTICE*10+idx] = fnq[10] - (fnq[10] - feq[10]) / tau;
			d_dst[NLATTICE*11+idx] = fnq[11] - (fnq[11] - feq[11]) / tau;
			d_dst[NLATTICE*12+idx] = fnq[12] - (fnq[12] - feq[12]) / tau;
			d_dst[NLATTICE*13+idx] = fnq[13] - (fnq[13] - feq[13]) / tau;
			d_dst[NLATTICE*14+idx] = fnq[14] - (fnq[14] - feq[14]) / tau;
			d_dst[NLATTICE*15+idx] = fnq[15] - (fnq[15] - feq[15]) / tau;
			d_dst[NLATTICE*16+idx] = fnq[16] - (fnq[16] - feq[16]) / tau;
			d_dst[NLATTICE*17+idx] = fnq[17] - (fnq[17] - feq[17]) / tau;
			d_dst[NLATTICE*18+idx] = fnq[18] - (fnq[18] - feq[18]) / tau;
		}
	}
}

__global__ void boundary_stream(int NLATTICE,float*  __restrict__ d_scr,float*  __restrict__ d_dst,float*  __restrict__ d_ux,float*  __restrict__ d_uy,float*  __restrict__ d_uz,float*  __restrict__ d_rho,float tau){
	float feq,fnq[19];
	int i,j,k,ind,idx,ind2,idx2,koff,geo_tmp;
	int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y;
	float tmp, tmp_rho,tmp_ux, tmp_uy, tmp_uz,uygt;
	float u_max=0.09714700668f;
	
	for(koff=BLOCK_Z-1;koff>=0;koff--){
		i=threadIdx.x+blockIdx.x*blockDim.x;
		j=threadIdx.y+blockIdx.y*blockDim.y;
		k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
		ind=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
		uygt=u_max*(1.0f-(pow(i-(NX - 1) / 2.0f,2)+pow(k-(NZ - 1) / 2.0f,2))/pow((NX - 1) / 2.0f,2));
		geo_tmp=tex1Dfetch(d_geo_text,ind);
		idx2=tex1Dfetch(d_index_text,ind);
		
		if (geo_tmp == 1) {
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[1] = d_dst[NLATTICE+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[2] = d_dst[NLATTICE*2+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[3] = d_dst[NLATTICE*3+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[4] = d_dst[NLATTICE*4+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[5] = d_dst[NLATTICE*5+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[6] = d_dst[NLATTICE*6+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[7] = d_dst[NLATTICE*7+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[8] = d_dst[NLATTICE*8+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[9] = d_dst[NLATTICE*9+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[10] = d_dst[NLATTICE*10+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[11] = d_dst[NLATTICE*11+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[12] = d_dst[NLATTICE*12+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[13] = d_dst[NLATTICE*13+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[14] = d_dst[NLATTICE*14+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[15] = d_dst[NLATTICE*15+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[16] = d_dst[NLATTICE*16+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[17] = d_dst[NLATTICE*17+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			fnq[18] = d_dst[NLATTICE*18+idx];
			
			d_dst[NLATTICE*1+idx2] = fnq[2];
			d_dst[NLATTICE*2+idx2] = fnq[1];
			d_dst[NLATTICE*3+idx2] = fnq[4];
			d_dst[NLATTICE*4+idx2] = fnq[3];
			d_dst[NLATTICE*5+idx2] = fnq[6];
			d_dst[NLATTICE*6+idx2] = fnq[5];
			d_dst[NLATTICE*7+idx2] = fnq[10];
			d_dst[NLATTICE*8+idx2] = fnq[9];
			d_dst[NLATTICE*9+idx2] = fnq[8];
			d_dst[NLATTICE*10+idx2] = fnq[7];
			d_dst[NLATTICE*11+idx2] = fnq[14];
			d_dst[NLATTICE*12+idx2] = fnq[13];
			d_dst[NLATTICE*13+idx2] = fnq[12];
			d_dst[NLATTICE*14+idx2] = fnq[11];
			d_dst[NLATTICE*15+idx2] = fnq[18];
			d_dst[NLATTICE*16+idx2] = fnq[17];
			d_dst[NLATTICE*17+idx2] = fnq[16];
			d_dst[NLATTICE*18+idx2] = fnq[15];
		}
		
		if (geo_tmp == 3) {
			//4
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
						
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /18.0f * (1.0f - 3.0f* uygt + 3.0f*uygt*uygt);
			d_dst[NLATTICE*4+idx2] = tmp + (d_dst[NLATTICE*4+idx] - feq)*(1.0f - 1.0f / tau);

			//8
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f* uygt+ 3.0f*uygt*uygt);
			d_dst[NLATTICE*8+idx2] = tmp + (d_dst[NLATTICE*8+idx] - feq)*(1.0f - 1.0f / tau);

			//10
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f*uygt + 3.0f*uygt*uygt);
			d_dst[NLATTICE*10+idx2] = tmp + (d_dst[NLATTICE*10+idx] - feq)*(1.0f - 1.0f / tau);

			//16
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f* uygt + 3.0f* uygt*uygt);
			d_dst[NLATTICE*16+idx2] = tmp + (d_dst[NLATTICE*16+idx] - feq)*(1.0f - 1.0f / tau);
			
			//18
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f* uygt + 3.0f* uygt*uygt );
			d_dst[NLATTICE*18+idx2] = tmp + (d_dst[NLATTICE*18+idx] - feq)*(1.0f - 1.0f / tau);
		}

		if (geo_tmp == 2) {
			//3
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /18.0f * (1.0f + 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /18.0f * (1.0f + 3.0f* uygt + 3.0f*uygt*uygt);
			d_dst[NLATTICE*3+idx2] = tmp + (d_dst[NLATTICE*3+idx] - feq)*(1.0f - 1.0f / tau);

			//7
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f* (1.0f + 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy -1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f* (1.0f + 3.0f* uygt + 3.0f*uygt*uygt);
			d_dst[NLATTICE*7+idx2] = tmp + (d_dst[NLATTICE*7+idx] - feq)*(1.0f - 1.0f / tau);

			//9
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_ux) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f* uygt+ 3.0f*uygt*uygt);
			d_dst[NLATTICE*9+idx2] = tmp + (d_dst[NLATTICE*9+idx] - feq)*(1.0f - 1.0f / tau);
							
			//15
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz- 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f* uygt + 3.0f* uygt*uygt);
			d_dst[NLATTICE*15+idx2] = tmp + (d_dst[NLATTICE*15+idx] - feq)*(1.0f - 1.0f / tau);

			//17
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index_text,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f* uygt + 3.0f* uygt*uygt );
			d_dst[NLATTICE*17+idx2] = tmp + (d_dst[NLATTICE*17+idx] - feq)*(1.0f - 1.0f / tau);
		}
	}
}

__global__ void calc_vel_square(float*  __restrict__ d_velsum,float*  __restrict__ d_ux,float*  __restrict__ d_uy,float*  __restrict__ d_uz,int NLATTICE){
	int const ind = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(ind<NLATTICE){
		d_velsum[ind]=sqrt(powf(d_ux[ind],2.f)+powf(d_uy[ind],2.f)+powf(d_uz[ind],2.f));
	}
}
		
void outputSave(string output_direc,int t) {
	int ind,idx;
	stringstream ss;
	ss << output_direc << '/' << "pos" << '_' << t << ".vtk";
	string datafilename = ss.str();
	ofstream ofs(datafilename);
	ofs<<"# vtk DataFile Version 2.0"<<endl;
	ofs<<"<-- LBM flow with UIV acceleration, http://www.bg.ic.ac.uk/research/m.tang/ulis/ -->"<<endl;
	ofs<<"ASCII"<<endl;
	ofs<<"DATASET STRUCTURED_POINTS"<<endl;
	ofs << "DIMENSIONS " << NX-2 << ' ' << NY-4 << ' ' << NZ-2 << endl;
	ofs<< "SPACING "<< CH<<' '<< CH<<' '<< CH<<endl;
	ofs<<"ORIGIN "<< round(NX / 2)*CH<<' '<< round(NY / 2)*CH<<' '<< .0<<endl;
	ofs<<"POINT_DATA  "<<(NX-2) *(NY-4)*(NZ-2)<<endl;

	ofs << "VECTORS VELOCITY float" << endl;
	for (int z=1; z < NZ-1; z++) {
		for (int y=2; y < NY-2; y++) {
			for (int x=1; x < NX-1; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];
				if(idx>=0){
					ofs << h_ux[idx] * C_U << ' ';
					ofs << h_uy[idx] * C_U << ' ';
					ofs << h_uz[idx] * C_U << ' ';
				}else{
					ofs << 0.0f << ' ';
					ofs << 0.0f << ' ';
					ofs << 0.0f << ' ';
				}
			}
		}
	}

	ofs.close();
}

int main(int argc,const char **argv) {
	string output_direc = "./out";
	float residual,milli,sum_current,sum_next;
	float tol=1e-5;
	int stag_max=20,k=0,tol_count=0,max_it=10000,time_save=200;
	string logname=output_direc+'/'+ "CONVERGENCE.log";
	ofstream logfile(logname);	

	h_index=(int*)malloc(sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z);
	h_geo=(int*)malloc(sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z);
	geo_pre();
	index_transform();
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
		
	//allocate memory
	h_ux=(float*)malloc(sizeof(float)*NLATTICE);
	h_uy=(float*)malloc(sizeof(float)*NLATTICE);
	h_uz=(float*)malloc(sizeof(float)*NLATTICE);
	h_rho=(float*)malloc(sizeof(float)*NLATTICE);
	h_dst=(float*)malloc(sizeof(float)*NLATTICE*Q);
	h_scr=(float*)malloc(sizeof(float)*NLATTICE*Q);
	
	cudaMalloc(&d_index,sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z);
	cudaMalloc(&d_ux,sizeof(float)*NLATTICE);
	cudaMalloc(&d_uy,sizeof(float)*NLATTICE);
	cudaMalloc(&d_uz,sizeof(float)*NLATTICE);
	cudaMalloc(&d_velsum,sizeof(float)*NLATTICE);
	cudaMalloc(&d_rho,sizeof(float)*NLATTICE);		
	cudaMalloc(&d_geo,bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z*sizeof(int));
	cudaMalloc(&d_dst,sizeof(float)*NLATTICE*Q);
	cudaMalloc(&d_scr,sizeof(float)*NLATTICE*Q);
	thrust::device_ptr<float> dev_d_velsum(d_velsum);
	
	initialize();
	cudaMemcpy(d_index,h_index,sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z,cudaMemcpyHostToDevice);
 	cudaMemcpy(d_geo,h_geo,bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z*sizeof(int),cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_dst,h_dst,sizeof(float)*NLATTICE*Q,cudaMemcpyHostToDevice);
	cudaMemcpy(d_scr,h_scr,sizeof(float)*NLATTICE*Q,cudaMemcpyHostToDevice);

	cudaBindTexture(NULL,d_index_text,d_index);	
	cudaBindTexture(NULL,d_geo_text,d_geo);
	cudaEventRecord(start);
	sum_current=0.f;
	while(k<=max_it&&tol_count<=stag_max){		
		update<<<dimGrid,dimBlock>>>(NLATTICE,d_scr,d_dst,d_ux,d_uy,d_uz,d_rho,tau);
		cudaDeviceSynchronize();

		boundary_stream<<<dimGrid,dimBlock>>>(NLATTICE,d_scr,d_dst,d_ux,d_uy,d_uz,d_rho,tau);
		cudaDeviceSynchronize();
		
		calc_vel_square<<<(NLATTICE+128-1)/128,128>>>(d_velsum,d_ux,d_uy,d_uz,NLATTICE);
		cudaDeviceSynchronize();
		
		sum_next=thrust::reduce(thrust::device,dev_d_velsum,dev_d_velsum+NLATTICE,0.f, thrust::plus<float>());

		d_tmp=d_scr;
		d_scr = d_dst;
		d_dst=d_tmp;
		if(k%time_save==0){
			cudaDeviceSynchronize();
			cudaMemcpy(h_ux,d_ux,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uy,d_uy,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uz,d_uz,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rho,d_rho,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);			
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milli,start,stop);
			residual=abs(sum_next-sum_current)/sum_next;
			cout<<"ITERATION # " << k << ", collapse time: "<< milli <<" ms, residual:"<< residual<<endl;
			logfile<<residual<<endl;
			outputSave(output_direc,k);
		}
		k++;
		sum_current=sum_next;
		if(residual<=tol)tol_count++;
	}
	cudaMemcpy(h_ux,d_ux,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_uy,d_uy,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_uz,d_uz,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho,d_rho,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
	outputSave(output_direc,k);
			
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli,start,stop);
	cout << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << "#LATTICE" << NLATTICE << endl;
	cout<<"Residual is "<<residual<<endl;
	logfile << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << "#LATTICE" << NLATTICE << " ERROR IS"<<residual<<endl;
	logfile.close();
	
	//free memory
	free(h_index);
	free(h_ux);
	free(h_uy);
	free(h_uz);
	free(h_rho);
	free(h_dst);
	free(h_scr);
	free(h_geo);
	cudaUnbindTexture(d_index_text);	
	cudaUnbindTexture(d_geo_text);
	cudaFree(d_index);
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
