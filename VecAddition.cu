#include<iostream>
#define N 100
using namespace std;

__global__ void vecAdd(float* A, float* B, float* C){
    int i = blockIdx.x * blockDim.x+ threadIdx.x;
    if(i<N)
        C[i] = A[i] + B[i];
}

__global__ void vecAdd_alternate(float* A, float* B, float* C){
    int i = (blockIdx.x * blockDim.x+ threadIdx.x)*2;
    if((i+1)<N){
        C[i] = A[i] + B[i];
        C[i+1] = A[i+1] + B[i+1];
    }
}

int main(){
    float A_h[N];
    for(int i=0;i<N;i++)
        A_h[i] = i+1;
    float B_h[N];
    for(int i=0;i<N;i++)
        B_h[i] = i+2;
    float C_h[N];
    float *A_d,*B_d,*C_d;
    cudaMalloc((void**)&A_d,N*sizeof(float));
    cudaMalloc((void**)&B_d,N*sizeof(float));
    cudaMalloc((void**)&C_d,N*sizeof(float));
    cudaMemcpy(A_d,A_h,N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,N*sizeof(float), cudaMemcpyHostToDevice);
    vecAdd_alternate<<<ceil(N/32.0),32>>>(A_d,B_d,C_d);
    cudaMemcpy(C_h,C_d,N*sizeof(float), cudaMemcpyDeviceToHost);
    
    cout<<"A"<<endl;
    for(int i=0;i<N;i++)
        cout<<A_h[i]<<" ";
    cout<<endl<<"B"<<endl;
    for(int i=0;i<N;i++)
        cout<<B_h[i]<<" ";
    cout<<endl<<"C"<<endl;
    for(int i=0;i<N;i++)
        cout<<C_h[i]<<" ";
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}