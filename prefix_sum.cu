#include<iostream>
using namespace std;
#define BLOCKDIM 1024
#define LENGTH 1024

__global__ void prefix_sum(int *X, int *Y){
    __shared__ int XY[BLOCKDIM];
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<LENGTH)
        XY[threadIdx.x] = X[i];
    else
        XY[threadIdx.x] =0;
    for(int stride = 1; stride<blockDim.x; stride *=2){
        __syncthreads();
        float temp;
        if(threadIdx.x>= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads();
        if(threadIdx.x>=stride)
            XY[threadIdx.x] = temp;

    }
    if(i<LENGTH){
        Y[i] = XY[threadIdx.x];
    }
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
    dim3 dimGrid((LENGTH+BLOCKDIM -1)/BLOCKDIM);
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