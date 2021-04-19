#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4

using namespace std;

string output_direc = "./out";
string filename = "bif";
string logname;

const int Q = 19, NX = 64, NY = 83, NZ = 32,REPEAT=4400,time_save=4400;//D3Q19
const float CH = 0.000248925f, C_rho = 998.2f, C_U = 0.24159041f;
float *h_ux, *h_uy, *h_uz, *h_rho,*h_inletx,*h_inlety,*h_outletx,*h_outlety,*h_meax,*h_meay, *d_dst, *d_scr,*h_dst, *h_scr, *d_tmp,*d_ux,*d_uy,*d_uz,*d_rho,*d_meax1,*d_meay1,*d_inletx1,*d_inlety1,*d_outletx1,*d_outlety1;
int *h_geo,*d_geo1,*h_index,*d_indexdev;
float C_pre = C_rho*C_U*C_U;
int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y,bz=1+(NZ-1)/BLOCK_Z;
int NLATTICE=0;
float wi[19] = { 1.0f / 3.0f, 1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 18.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f,1.0f / 36.0f };
texture<float,cudaTextureType1D,cudaReadModeElementType> d_meax;
texture<float,cudaTextureType1D,cudaReadModeElementType> d_meay;
texture<int,cudaTextureType1D,cudaReadModeElementType> d_index;
texture<float,cudaTextureType1D,cudaReadModeElementType> d_inletx;
texture<float,cudaTextureType1D,cudaReadModeElementType> d_inlety;
texture<float,cudaTextureType1D,cudaReadModeElementType> d_outletx;
texture<float,cudaTextureType1D,cudaReadModeElementType> d_outlety;
texture<int,cudaTextureType1D,cudaReadModeElementType> d_geo;

void geo_pre() {
	//not useful=0,ghost cells=-1, wall=1,left end=2,right end=3,fluid=4,augmentation=5
	int minn4, minx, miny, minz, minn8, ind, ind2, i, j, k, tmp;
	int ***flag;
	FILE *file;
	flag = (int ***)malloc(NX * sizeof(*flag));

	for (int x = 0; x < NX; x++) {
		flag[x] = (int **)malloc(NY * sizeof(**flag));
		for (int y = 0; y < NY; y++) {
			flag[x][y] = (int *)malloc(NZ * sizeof(***flag));
		}
	}

	file=fopen("./geo.txt", "r");
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				fscanf(file, "%d ", &tmp);
				h_geo[ind] = tmp;
				flag[x][y][z] = tmp;
			}
		}
	}
	fclose(file);

	int y;
	for (int x = 1; x < NX - 1; x++) {
		for (int z = 1; z < NZ - 1; z++) {
			y = 0;
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			h_geo[ind] = 0;
			y = NY - 1;
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			h_geo[ind] = 0;
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

	//distance transform, left end=2
	for (int x = 1; x < NX - 1; x++) {
		for (int z = 1; z < NZ - 1; z++) {
			y = 1;
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			y=2;
			ind2= (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			h_geo[ind]=0;
			if(h_geo[ind2]==1)h_geo[ind]=1;
			if(h_geo[ind2]==4)h_geo[ind]=2;
		}
	}
	
	//distance transform, right end=3
	for (int t = 0; t < 2; t++) {
		for (int x = 1; x < NX - 1; x++) {
			for (int z = 1; z < NZ - 1; z++) {
				y = NY - 2;
				ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				y=NY-3;
				ind2= (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				h_geo[ind]=0;
				if(h_geo[ind2]==1)h_geo[ind]=1;
				if(h_geo[ind2]==4)h_geo[ind]=3;
			}
		}
	}

	free(flag);
	
	for (int z = 1; z < NZ - 1; z++) {
		for (int y = 1; y < NY - 1; y++) {
			for (int x = 1; x < NX - 1; x++) {
				ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				if (h_geo[ind] == 1 ) {
					//1
					i = x + 1;
					j = y;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//2
					i = x - 1;
					j = y;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//3
					i = x;
					j = y + 1;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//4
					i = x;
					j = y - 1;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//5
					i = x;
					j = y;
					k = z + 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//6
					i = x;
					j = y;
					k = z - 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//7
					i = x + 1;
					j = y + 1;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//8
					i = x + 1;
					j = y - 1;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//9
					i = x - 1;
					j = y + 1;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//10
					i = x - 1;
					j = y - 1;
					k = z;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//11
					i = x + 1;
					j = y;
					k = z + 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//12
					i = x + 1;
					j = y;
					k = z - 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//13
					i = x - 1;
					j = y;
					k = z + 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//14
					i = x - 1;
					j = y;
					k = z - 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//15
					i = x;
					j = y + 1;
					k = z + 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//16
					i = x;
					j = y - 1;
					k = z + 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//17
					i = x;
					j = y + 1;
					k = z - 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
					//18
					i = x;
					j = y - 1;
					k = z - 1;
					ind2 = (i / BLOCK_X + j / BLOCK_Y*bx + k / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + i%BLOCK_X + j%BLOCK_Y*BLOCK_X + k%BLOCK_Z*BLOCK_X*BLOCK_Y;
					if (h_geo[ind2] == 0)h_geo[ind2] = -1;
				}
			}
		}
	}

	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				if (h_geo[ind] != 0) {
					h_index[ind] = NLATTICE;
					NLATTICE++;
				}
				else h_index[ind] = -1;
			}
		}
	}
}

void read_vel() {
	int ind, ind1, z = NZ/2;
	FILE *file, *file1;
	float tmp;
	
	/**
	file=fopen("./measurements.txt", "r");	
	for (int y = 0; y < NY; y++) {
		for (int x = 0; x < NX; x++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			ind1 = x + y*NX;
			fscanf(file, "%f ", &tmp);
			if (h_geo[ind] == 4) {
				h_meay[ind1] = tmp;
			}
			else {
				h_meay[ind1] = 0;
			}
		}
	}
	
	for (int y = 0; y < NY; y++) {
		for (int x = 0; x < NX; x++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			ind1 = x + y*NX;
			fscanf(file, "%f ", &tmp);

			if (h_geo[ind] == 4) {
				h_meax[ind1] = tmp;
			}
			else {
				h_meax[ind1] = 0;
			}

		}
	}
	fclose(file);
	**/
	
	int y = 1;
	file1=fopen("./bc.txt", "r");
	for (int z = 0; z < NZ; z++) {
		for (int x = 0; x < NX; x++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			ind1 = x + z*NX;
			fscanf(file1, "%f ", &tmp);
			if (h_geo[ind] == 2) {
				h_inlety[ind1] = tmp;
			}
			else {
				h_inlety[ind1] = 0;
			}
			h_inletx[ind1]=0;
		}
	}
	
	y = NY - 2;
	for (int z = 0; z < NZ; z++) {
		for (int x = 0; x < NX; x++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			ind1 = x + z*NX;
			fscanf(file1, "%f ", &tmp);
			if (h_geo[ind] == 3) {
				h_outlety[ind1] = tmp;
			}
			else {
				h_outlety[ind1] = 0;
			}
			h_outletx[ind1]=0;
		}
	}
	fclose(file1);
}

void initialize() {
	int ind,idx,ind1;
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
	
	int y=1;
	for (int x = 0; x<NX; x++) {
		for (int z = 0; z<NZ; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];				
			ind1 = x + z*NX;
			if(idx>=0){
				h_ux[idx] =0.0f;
				h_uy[idx] =h_inlety[ind1];
			}
		}
	}
	
	y=NY-2;
	for (int x = 0; x<NX; x++) {
		for (int z = 0; z<NZ; z++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];				
			ind1 = x + z*NX;
			if(idx>=0){
				h_ux[idx] =0.0f;
				h_uy[idx] =h_outlety[ind1];
			}
		}
	}
	/**
	int z=NZ/2;
	for (int x = 0; x<NX; x++) {
		for (int y = 0; y<NY; y++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];
			ind1 = x + y*NX;			
			if(h_geo[ind]==4){
				h_ux[idx] = h_meax[ind1];
				h_uy[idx] = h_meay[ind1];
			}
		}
	}
	**/
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

__global__ void update(int NLATTICE,float*  __restrict__ d_scr,float*  __restrict__ d_dst,float*  __restrict__ d_ux,float*  __restrict__ d_uy,float*  __restrict__ d_uz,float*  __restrict__ d_rho){
	float fnq[19],feq;
	int i,j,k,ind,ind2,idx,koff,geo_tmp;
	int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y;
	float tmp_rho,tmp_ux, tmp_uy, tmp_uz;
	float tau=0.55f;
	
	for(koff=BLOCK_Z-1;koff>=0;koff--){
		i=threadIdx.x+blockIdx.x*blockDim.x;
		j=threadIdx.y+blockIdx.y*blockDim.y;
		k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
		ind=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
		idx=tex1Dfetch(d_index,ind);
		
		geo_tmp=tex1Dfetch(d_geo,ind);
		if (geo_tmp ==4) {
			fnq[0] = d_scr[idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[1] = d_scr[NLATTICE*1+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[2] = d_scr[NLATTICE*2+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[3] = d_scr[NLATTICE*3+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[4] = d_scr[NLATTICE*4+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[5] = d_scr[NLATTICE*5+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[6] = d_scr[NLATTICE*6+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[7] = d_scr[NLATTICE*7+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[8] = d_scr[NLATTICE*8+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[9] = d_scr[NLATTICE*9+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[10] = d_scr[NLATTICE*10+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[11] = d_scr[NLATTICE*11+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[12] = d_scr[NLATTICE*12+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[13] = d_scr[NLATTICE*13+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[14] = d_scr[NLATTICE*14+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[15] = d_scr[NLATTICE*15+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[16] = d_scr[NLATTICE*16+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[17] = d_scr[NLATTICE*17+idx];
				
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[18] = d_scr[NLATTICE*18+idx];

			
			tmp_rho = 0.f;
			for (int k = 0; k < Q; k++) {
				tmp_rho = tmp_rho + fnq[k];
			}

			tmp_ux = (fnq[1] - fnq[2] + fnq[7] + fnq[8] - fnq[9] - fnq[10] + fnq[11] + fnq[12] - fnq[13] - fnq[14]) / tmp_rho;
			tmp_uy = (fnq[3] - fnq[4] + fnq[7] - fnq[8] + fnq[9] - fnq[10] + fnq[15] - fnq[16] + fnq[17] - fnq[18]) / tmp_rho;
			tmp_uz = (fnq[5] - fnq[6] + fnq[11] - fnq[12] + fnq[13] - fnq[14] + fnq[15] + fnq[16] - fnq[17] - fnq[18]) / tmp_rho;

			idx=tex1Dfetch(d_index,ind);
			/**
			if(geo_tmp==5){
				k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
				ind2=i + k*NX;
				tmp_ux=tex1Dfetch(d_meax,ind2);
				tmp_uy=tex1Dfetch(d_meay,ind2);
			}
			**/
			d_rho[idx] = tmp_rho;			
			d_ux[idx] = tmp_ux;
			d_uy[idx] = tmp_uy;
			d_uz[idx] = tmp_uz;			

			feq = tmp_rho/3.0f * (1.0f - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy -1.5f* tmp_uz*tmp_uz);
			d_dst[idx] = fnq[0] - (fnq[0] - feq) / tau;
			feq = tmp_rho /18.0f * (1.0f + 3.0f* tmp_ux + 3.0f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy -1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*1+idx] = fnq[1] - (fnq[1] - feq) / tau;
			feq = tmp_rho /18.0f * (1.0f - 3.0f* tmp_ux + 3.0f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy -1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*2+idx] = fnq[2] - (fnq[2] - feq) / tau;
			feq = tmp_rho /18.0f * (1.0f + 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*3+idx] = fnq[3] - (fnq[3] - feq) / tau;
			feq = tmp_rho /18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*4+idx] = fnq[4] - (fnq[4] - feq) / tau;
			feq = tmp_rho /18.0f * (1.0f + 3.0f* tmp_uz + 3.0f*tmp_uz*tmp_uz - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy);
			d_dst[NLATTICE*5+idx] = fnq[5] - (fnq[5] - feq) / tau;
			feq = tmp_rho /18.0f* (1.0f - 3.0f* tmp_uz + 3.0f*tmp_uz*tmp_uz - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uy*tmp_uy);
			d_dst[NLATTICE*6+idx] = fnq[6] - (fnq[6] - feq) / tau;
			feq = tmp_rho /36.0f* (1.0f + 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy -1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*7+idx] = fnq[7] - (fnq[7] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*8+idx] = fnq[8] - (fnq[8] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_ux) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*9+idx] = fnq[9] - (fnq[9] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*10+idx] = fnq[10] - (fnq[10] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux + tmp_uz) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_ux*tmp_uz-1.5f* tmp_uy*tmp_uy);
			d_dst[NLATTICE*11+idx] = fnq[11] - (fnq[11] - feq) / tau;
			feq = tmp_rho /36.0f* (1.0f + 3.0f* (tmp_ux - tmp_uz) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_ux*tmp_uz-1.5f* tmp_uy*tmp_uy);
			d_dst[NLATTICE*12+idx] = fnq[12] - (fnq[12] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_ux) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_ux*tmp_uz-1.5f* tmp_uy*tmp_uy);
			d_dst[NLATTICE*13+idx] = fnq[13] - (fnq[13] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uz) + 3.0f* tmp_ux*tmp_ux + 3.0*tmp_uz*tmp_uz + 9.0f*tmp_ux*tmp_uz -1.5f* tmp_uy*tmp_uy);
			d_dst[NLATTICE*14+idx] = fnq[14] - (fnq[14] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz- 1.5f*tmp_ux*tmp_ux);
			d_dst[NLATTICE*15+idx] = fnq[15] - (fnq[15] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			d_dst[NLATTICE*16+idx] = fnq[16] - (fnq[16] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			d_dst[NLATTICE*17+idx] = fnq[17] - (fnq[17] - feq) / tau;
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			d_dst[NLATTICE*18+idx] = fnq[18] - (fnq[18] - feq) / tau;
		}
	}
}

__global__ void boundary_stream(int NLATTICE,float*  __restrict__ d_scr,float*  __restrict__ d_dst,float*  __restrict__ d_ux,float*  __restrict__ d_uy,float*  __restrict__ d_uz,float*  __restrict__ d_rho){
	float feq,fnq[19];
	int i,j,k,ind,idx,ind2,idx2,koff,geo_tmp;
	int bx=1+(NX-1)/BLOCK_X,by=1+(NY-1)/BLOCK_Y;
	float tmp, tmp_rho,tmp_ux, tmp_uy, tmp_uz,uygt_in,uygt_out,tau=0.55f;
	
	for(koff=BLOCK_Z-1;koff>=0;koff--){
		i=threadIdx.x+blockIdx.x*blockDim.x;
		j=threadIdx.y+blockIdx.y*blockDim.y;
		k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
		ind=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
		ind2=i + k*NX;
		geo_tmp=tex1Dfetch(d_geo,ind);
		idx2=tex1Dfetch(d_index,ind);
		
		if (geo_tmp == 1) {
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[1] = d_dst[NLATTICE+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[2] = d_dst[NLATTICE*2+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[3] = d_dst[NLATTICE*3+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[4] = d_dst[NLATTICE*4+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[5] = d_dst[NLATTICE*5+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[6] = d_dst[NLATTICE*6+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[7] = d_dst[NLATTICE*7+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[8] = d_dst[NLATTICE*8+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[9] = d_dst[NLATTICE*9+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[10] = d_dst[NLATTICE*10+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[11] = d_dst[NLATTICE*11+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[12] = d_dst[NLATTICE*12+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[13] = d_dst[NLATTICE*13+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[14] = d_dst[NLATTICE*14+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[15] = d_dst[NLATTICE*15+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[16] = d_dst[NLATTICE*16+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y-1+NY)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			fnq[17] = d_dst[NLATTICE*17+idx];
			
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=(threadIdx.y+blockIdx.y*blockDim.y+1)%NY;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
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
		
		/**
		if (geo_tmp == 3) {
			uygt_out=tex1Dfetch(d_outlety,ind2);
			//4
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
						
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /18.0f * (1.0f - 3.0f* uygt_out + 3.0f*uygt_out*uygt_out);
			d_dst[NLATTICE*4+idx2] = tmp + (d_dst[NLATTICE*4+idx] - feq)*(1.0f - 1.0f / tau);

			//8
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f* uygt_out+ 3.0f*uygt_out*uygt_out);
			d_dst[NLATTICE*8+idx2] = tmp + (d_dst[NLATTICE*8+idx] - feq)*(1.0f - 1.0f / tau);

			//10
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f*uygt_out + 3.0f*uygt_out*uygt_out);
			d_dst[NLATTICE*10+idx2] = tmp + (d_dst[NLATTICE*10+idx] - feq)*(1.0f - 1.0f / tau);

			//16
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f* uygt_out + 3.0f* uygt_out*uygt_out);
			d_dst[NLATTICE*16+idx2] = tmp + (d_dst[NLATTICE*16+idx] - feq)*(1.0f - 1.0f / tau);
			
			//18
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f - 3.0f* uygt_out + 3.0f* uygt_out*uygt_out );
			d_dst[NLATTICE*18+idx2] = tmp + (d_dst[NLATTICE*18+idx] - feq)*(1.0f - 1.0f / tau);
		}
		**/
		
		if (geo_tmp == 3) {
			//4
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
						
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			tmp = 1.f /18.0f * (1.0f - 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*4+idx2] = tmp + (d_dst[NLATTICE*4+idx] - feq)*(1.0f - 1.0f / tau);

			//8
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = 1.f /36.0f * (1.0f + 3.0f* (tmp_ux - tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*8+idx2] = tmp + (d_dst[NLATTICE*8+idx] - feq)*(1.0f - 1.0f / tau);

			//10
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = 1.f /36.0f * (1.0f - 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			d_dst[NLATTICE*10+idx2] = tmp + (d_dst[NLATTICE*10+idx] - feq)*(1.0f - 1.0f / tau);

			//16
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = 1.f /36.0f * (1.0f + 3.0f* (tmp_uz - tmp_uy) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			d_dst[NLATTICE*16+idx2] = tmp + (d_dst[NLATTICE*16+idx] - feq)*(1.0f - 1.0f / tau);
			
			//18
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y-1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = 1.f /36.0f * (1.0f - 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			d_dst[NLATTICE*18+idx2] = tmp + (d_dst[NLATTICE*18+idx] - feq)*(1.0f - 1.0f / tau);
		}
		
		if (geo_tmp == 2) {
			uygt_in=tex1Dfetch(d_inlety,ind2);
			//3
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /18.0f * (1.0f + 3.0f* tmp_uy + 3.0f*tmp_uy*tmp_uy - 1.5f*tmp_ux*tmp_ux -1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /18.0f * (1.0f + 3.0f* uygt_in + 3.0f*uygt_in*uygt_in);
			d_dst[NLATTICE*3+idx2] = tmp + (d_dst[NLATTICE*3+idx] - feq)*(1.0f - 1.0f / tau);

			//7
			i=threadIdx.x+blockIdx.x*blockDim.x+1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f* (1.0f + 3.0f* (tmp_ux + tmp_uy) + 3.0f*tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy + 9.0f*tmp_ux*tmp_uy -1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f* (1.0f + 3.0f* uygt_in + 3.0f*uygt_in*uygt_in);
			d_dst[NLATTICE*7+idx2] = tmp + (d_dst[NLATTICE*7+idx] - feq)*(1.0f - 1.0f / tau);

			//9
			i=threadIdx.x+blockIdx.x*blockDim.x-1;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_ux) + 3.0f* tmp_ux*tmp_ux + 3.0f*tmp_uy*tmp_uy - 9.0f*tmp_ux*tmp_uy-1.5f* tmp_uz*tmp_uz);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f* uygt_in+ 3.0f*uygt_in*uygt_in);
			d_dst[NLATTICE*9+idx2] = tmp + (d_dst[NLATTICE*9+idx] - feq)*(1.0f - 1.0f / tau);
							
			//15
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff+1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy + tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz + 9.0f*tmp_uy*tmp_uz- 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f* uygt_in + 3.0f* uygt_in*uygt_in);
			d_dst[NLATTICE*15+idx2] = tmp + (d_dst[NLATTICE*15+idx] - feq)*(1.0f - 1.0f / tau);

			//17
			i=threadIdx.x+blockIdx.x*blockDim.x;
			j=threadIdx.y+blockIdx.y*blockDim.y+1;
			k=threadIdx.z+blockIdx.z*BLOCK_Z+koff-1;
			ind2=(i/BLOCK_X+j/BLOCK_Y*bx+k/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+i%BLOCK_X+j%BLOCK_Y*BLOCK_X+k%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=tex1Dfetch(d_index,ind2);
			tmp_rho = d_rho[idx];
			tmp_ux = d_ux[idx];
			tmp_uy = d_uy[idx];
			tmp_uz = d_uz[idx];
			feq = tmp_rho /36.0f * (1.0f + 3.0f* (tmp_uy - tmp_uz) + 3.0f* tmp_uy*tmp_uy + 3.0f*tmp_uz*tmp_uz - 9.0f*tmp_uy*tmp_uz - 1.5f*tmp_ux*tmp_ux);
			tmp = tmp_rho /36.0f * (1.0f + 3.0f* uygt_in + 3.0f* uygt_in*uygt_in );
			d_dst[NLATTICE*17+idx2] = tmp + (d_dst[NLATTICE*17+idx] - feq)*(1.0f - 1.0f / tau);
		}
	}
}
			
void outtxt(){
	int ind,idx;
	int z=NZ/2;
	ofstream ofs("s1_out.txt");	
	for (int y = 0; y < NY; y++) {
		for (int x = 0; x < NX; x++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			if (h_geo[ind] == 4) {
				idx=h_index[ind];
				ofs<<h_uy[idx]<<' ';
			}else{
				ofs<<0<<' ';
			}
		}
	}
	
	for (int y = 0; y < NY; y++) {
		for (int x = 0; x < NX; x++) {
			ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			if (h_geo[ind] == 4) {
				idx=h_index[ind];
				ofs<<h_ux[idx]<<' ';
			}else{
				ofs<<0<<' ';
			}
		}
	}
	ofs.close();
}

void write_once() {
	int ind, idx, z = NZ/2;
	ofstream ofs("./meas1.txt");	
	for (int y = 0; y < NY; y++) {
		for (int x = 0; x < NX; x++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];
			ofs<<h_uy[idx]<<' ';
		}
	}
	
	for (int y = 0; y < NY; y++) {
		for (int x = 0; x < NX; x++) {
			ind = (x / BLOCK_X + y / BLOCK_Y*bx + z / BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z + x%BLOCK_X + y%BLOCK_Y*BLOCK_X + z%BLOCK_Z*BLOCK_X*BLOCK_Y;
			idx=h_index[ind];
			ofs<<h_ux[idx]<<' ';
		}
	}
	ofs.close();
}

void write_vel() {
	int ind,idx;
	string velo_file = "./scenario3a.txt";
	ofstream velfile(velo_file);
	
	for (int z=0; z < NZ; z++) {
		for (int y=0; y < NY; y++) {
			for (int x=0; x < NX; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];					
				if(idx>=0)velfile<<h_ux[idx]<<' '<<h_uy[idx] <<' '<<h_uz[idx] <<' ';
				//else velfile<<0<<' '<<0 <<' '<<0 <<' ';
			}
		}
		velfile<<endl;
	}
	velfile.close();
}

void outputSave(int t) {
	int ind,idx;
	string datafilename=output_direc + "/bif_" +to_string(t)+".vtk";
	ofstream ofs(datafilename);
	ofs<<"# vtk DataFile Version 2.0"<<endl;
	ofs<<"<-- LBM flow with UIV acceleration, http://www.bg.ic.ac.uk/research/m.tang/ulis/ -->"<<endl;
	ofs<<"ASCII"<<endl;
	ofs<<"DATASET STRUCTURED_POINTS"<<endl;
	ofs << "DIMENSIONS " << NX-2 << ' ' << NY-4 << ' ' << NZ-2 << endl;
	ofs<< "SPACING "<< CH<<' '<< CH<<' '<< CH<<endl;
	ofs<<"ORIGIN "<< round(NX / 2)*CH<<' '<< round(NY / 2)*CH<<' '<< .0<<endl;
	ofs<<"POINT_DATA  "<<(NX-2) *(NY-4)*(NZ-2)<<endl;
/**
	ofs << "SCALARS DENSITY float" << endl;
	ofs << "LOOKUP_TABLE default" << endl;
	for (int z=1; z < NZ-1; z++) {
		for (int y=2; y < NY-2; y++) {
			for (int x=1; x < NX-1; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];
				if(idx>=0)ofs << h_rho[idx] * C_rho << ' ';
				else ofs << 0.0f<< ' ';
			}
		}
	}
	ofs << endl;

	ofs << "SCALARS PRESSURE float" << endl;
	ofs << "LOOKUP_TABLE default" << endl;
	for (int z=1; z < NZ-1; z++) {
		for (int y=2; y < NY-2; y++) {
			for (int x=1; x < NX-1; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];	
				if(idx>=0)ofs << h_rho[idx] * C_pre / 3.0 << ' ';
				else ofs << 0.0f<< ' ';
			}
		}
	}
	ofs << endl;
**/
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
					ofs << 0 << ' ';
					ofs << 0 << ' ';
					ofs << 0 << ' ';
				}
			}
		}
	}

	ofs.close();
}

long double calc_res() {
	int ind,idx;
	float vtmp;
	long double sum1=0.0L;
	for (int z=1; z < NZ-1; z++) {
		for (int y=2; y < NY-2; y++) {
			for (int x=1; x < NX-1; x++) {
				ind=(x/BLOCK_X+y/BLOCK_Y*bx+z/BLOCK_Z*bx*by)*BLOCK_X*BLOCK_Y*BLOCK_Z+x%BLOCK_X+y%BLOCK_Y*BLOCK_X+z%BLOCK_Z*BLOCK_X*BLOCK_Y;
				idx=h_index[ind];
				if(h_geo[ind]>=4){
					vtmp = powf(h_ux[idx], 2.f) + powf(h_uy[idx], 2.f) + powf(h_uz[idx], 2.f);
					sum1 = sum1 + vtmp;				
				}
			}
		}
	}
	return sum1;
}

int main(int argc,const char **argv) {
	float residual,milli;
	long double sum1,sum2;
	logname=output_direc+'/'+ "CONVERGENCE.log";
	ofstream logfile(logname);	
	dim3 dimGrid(bx,by,bz);
	dim3 dimBlock(BLOCK_X,BLOCK_Y,1);

	h_index=(int*)malloc(sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z);
	h_geo=(int*)malloc(sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z);
	geo_pre();
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	
	//allocate memory
	h_ux=(float*)malloc(sizeof(float)*NLATTICE);
	h_uy=(float*)malloc(sizeof(float)*NLATTICE);
	h_uz=(float*)malloc(sizeof(float)*NLATTICE);
	h_rho=(float*)malloc(sizeof(float)*NLATTICE);
	h_meax=(float*)malloc(sizeof(float)*NX*NY);
	h_meay=(float*)malloc(sizeof(float)*NX*NY);
	h_inletx = (float*)malloc(sizeof(float)*NX*NZ);
	h_inlety = (float*)malloc(sizeof(float)*NX*NZ);
	h_outletx = (float*)malloc(sizeof(float)*NX*NZ);
	h_outlety = (float*)malloc(sizeof(float)*NX*NZ);
	h_dst=(float*)malloc(sizeof(float)*NLATTICE*Q);
	h_scr=(float*)malloc(sizeof(float)*NLATTICE*Q);
	
	cudaMalloc((void**)&d_indexdev,sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z);
	cudaMalloc((void**)&d_ux,sizeof(float)*NLATTICE);
	cudaMalloc((void**)&d_uy,sizeof(float)*NLATTICE);
	cudaMalloc((void**)&d_uz,sizeof(float)*NLATTICE);
	cudaMalloc((void**)&d_rho,sizeof(float)*NLATTICE);	
	
	cudaMalloc(&d_meax1,NX*NY*sizeof(float));
	cudaMalloc(&d_meay1,NX*NY*sizeof(float));
	cudaMalloc(&d_inletx1,NX*NZ*sizeof(float));
	cudaMalloc(&d_inlety1,NX*NZ*sizeof(float));
	cudaMalloc(&d_outletx1,NX*NZ*sizeof(float));
	cudaMalloc(&d_outlety1,NX*NZ*sizeof(float));	
	cudaMalloc(&d_geo1,bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z*sizeof(int));

	cudaMalloc((void**)&d_dst,sizeof(float)*NLATTICE*Q);
	cudaMalloc((void**)&d_scr,sizeof(float)*NLATTICE*Q);
	read_vel();
	initialize();
	cudaMemcpy(d_indexdev,h_index,sizeof(int)*bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z,cudaMemcpyHostToDevice);
 	cudaMemcpy(d_meax1,h_meax,NX*NY*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_meay1,h_meay,NX*NY*sizeof(float),cudaMemcpyHostToDevice);
 	cudaMemcpy(d_inletx1,h_inletx,NX*NZ*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_inlety1,h_inlety,NX*NZ*sizeof(float),cudaMemcpyHostToDevice);
 	cudaMemcpy(d_outletx1,h_outletx,NX*NZ*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_outlety1,h_outlety,NX*NZ*sizeof(float),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_geo1,h_geo,bx*by*bz*BLOCK_X*BLOCK_Y*BLOCK_Z*sizeof(int),cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_dst,h_dst,sizeof(float)*NLATTICE*Q,cudaMemcpyHostToDevice);
	cudaMemcpy(d_scr,h_scr,sizeof(float)*NLATTICE*Q,cudaMemcpyHostToDevice);

	cudaBindTexture(NULL,d_index,d_indexdev);	
	cudaBindTexture(NULL,d_meax,d_meax1);
	cudaBindTexture(NULL,d_meay,d_meay1);
	cudaBindTexture(NULL,d_inletx,d_inletx1);
	cudaBindTexture(NULL,d_inlety,d_inlety1);
	cudaBindTexture(NULL,d_outletx,d_outletx1);
	cudaBindTexture(NULL,d_outlety,d_outlety1);	
	cudaBindTexture(NULL,d_geo,d_geo1);
	cudaEventRecord(start);
	for(int i=0;i<=REPEAT;i++){
		cudaDeviceSynchronize();
	
		update<<<dimGrid,dimBlock>>>(NLATTICE,d_scr,d_dst,d_ux,d_uy,d_uz,d_rho);
		cudaDeviceSynchronize();

		boundary_stream<<<dimGrid,dimBlock>>>(NLATTICE,d_scr,d_dst,d_ux,d_uy,d_uz,d_rho);
		cudaDeviceSynchronize();
		
		d_tmp=d_scr;
		d_scr = d_dst;
		d_dst=d_tmp;
		if(i%time_save==0){
			cudaDeviceSynchronize();
			sum1=calc_res();
			cudaMemcpy(h_ux,d_ux,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uy,d_uy,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_uz,d_uz,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);
			cudaMemcpy(h_rho,d_rho,sizeof(float)*NLATTICE,cudaMemcpyDeviceToHost);			
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milli,start,stop);
			sum2=calc_res();
			residual=(float)(abs(sum1-sum2)/sum2);
			logfile<<residual<<endl;
			cout << "ITERATION # " << i << ", collapse time: " << milli <<" ms, residual:" << residual << endl;
			outputSave(i);
		}
	}
	//outtxt();
	//write_vel();
	write_once();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli,start,stop);
	cout << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << "#LATTICE" << NLATTICE << endl;
	logfile << "TOTAL RUNNING TIME: " << milli << " MILLI SECONDS" << "#LATTICE" << NLATTICE << " ERROR IS"<<residual<<endl;
	logfile.close();
	
	cudaUnbindTexture(d_index);	
	cudaUnbindTexture(d_geo);
	cudaUnbindTexture(d_meax);
	cudaUnbindTexture(d_meay);
	cudaUnbindTexture(d_inletx);
	cudaUnbindTexture(d_inlety);
	cudaUnbindTexture(d_outletx);
	cudaUnbindTexture(d_outlety);	
	//free memory
	free(h_index);
	free(h_ux);
	free(h_uy);
	free(h_uz);
	free(h_rho);
	free(h_meax);
	free(h_meay);
	free(h_dst);
	free(h_scr);
	free(h_geo);
	free(h_inletx);
	free(h_inlety);
	free(h_outletx);
	free(h_outlety);
	
	cudaFree(d_indexdev);
	cudaFree(d_dst);
	cudaFree(d_scr);
	cudaFree(d_geo1);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_uz);
	cudaFree(d_rho);	
	cudaFree(d_meax1);
	cudaFree(d_meay1);
	cudaFree(d_inletx1);
	cudaFree(d_inlety1);
	cudaFree(d_outletx1);
	cudaFree(d_outlety1);	
	cudaDeviceReset();
	system("pause");
	return 0;
}
