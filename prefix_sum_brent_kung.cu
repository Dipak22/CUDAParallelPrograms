#include<iostream>
using namespace std;
#define BLOCKDIM 512
#define LENGTH 1024

__global__ void prefix_sum(int *X, int *Y){
    __shared__ int XY[LENGTH];
    int id = 2*blockDim.x*blockIdx.x + threadIdx.x;
    if(id<LENGTH)
        XY[threadIdx.x] = X[id];
    if(id+blockDim.x<LENGTH)
        XY[threadIdx.x + blockDim.x] = X[id + blockDim.x];

    // Reduction Phase
    for(int stride = 1; stride<=blockDim.x; stride *=2){
        __syncthreads();
        int index = (threadIdx.x +1) * 2 * stride -1;
        if(index < LENGTH)
            XY[index] += XY[index - stride];

    }
    //Distribution phase
    for(int stride = LENGTH/4; stride>=1; stride /=2){
        __syncthreads();
        int index = (threadIdx.x +1) * 2 * stride -1;
        if(index + stride< LENGTH)
            XY[index+ stride] += XY[index];

    }
    __syncthreads();
    if(id<LENGTH){
        Y[id] = XY[threadIdx.x];
    }
    if(id + blockDim.x<LENGTH)
        Y[id+ blockDim.x] = XY[threadIdx.x + blockDim.x];

}

int main(){
    int *A, *out;
    A = (int*)malloc(LENGTH * sizeof(int));
    out =(int*)malloc(LENGTH * sizeof(int));
    memset(out,0 , LENGTH * sizeof(int));
    for(int i=0;i<LENGTH;i++)
        A[i] = i+1;
    int *A_d, *out_d;
    cudaMalloc((void**)&A_d, LENGTH * sizeof(int));
    cudaMalloc((void**)&out_d, LENGTH * sizeof(int));
    cudaMemset(out_d, 0 , LENGTH * sizeof(int));
    cudaMemcpy(A_d, A, LENGTH * sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid(1,1,1);
    prefix_sum<<<dimGrid,BLOCKDIM>>>(A_d, out_d);
    cudaMemcpy(out, out_d, LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i =0;i<LENGTH;i++)
        cout<<out[i]<<" ";

    cudaFree(A_d);
    cudaFree(out_d);

    free(A);
    free(out);

    return 0;
}