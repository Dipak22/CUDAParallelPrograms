#include<iostream>
#define M 4
#define N 4
using namespace std;

__global__ void matmul(int *A,int *B, int *C){
    int x = blockIdx.x* blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(y<M && x<N){
        int result = 0;
        for(int i=0;i<N;i++)
            result += A[y*N +i]* B[i*N + x];
        C[y * N + x] = result;
    }
}

int main(){

    int* A_h, *B_h, *C_h;
    A_h = (int*)malloc(M*N*sizeof(int));
    B_h = (int*)malloc(M*N*sizeof(int));
    C_h = (int*)malloc(M*N*sizeof(int));
    for(int i =0;i<M;i++){
        for(int j =0;j<N;j++)
            A_h[i*N+j] = rand()%5;
    }
    for(int i =0;i<N;i++){
        for(int j =0;j<M;j++)
            B_h[i*M+j] = rand()%5;
    }

    int* A_d, *B_d, *C_d;
    int memSize = M*N*sizeof(int);
    cudaMalloc((void**)&A_d,memSize);
    cudaMalloc((void**)&B_d,memSize);
    cudaMalloc((void**)&C_d,memSize);
    cudaMemcpy(A_d,A_h,memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,memSize, cudaMemcpyHostToDevice);
    dim3 dimBlock(16,32,1);
    dim3 dimGrid(ceil(N/16.0), ceil(M/32.0),1);
    matmul<<<dimGrid, dimBlock>>>(A_d,B_d,C_d);
    cudaMemcpy(C_h,C_d,memSize, cudaMemcpyDeviceToHost);

    cout<<"A"<<endl;
    for(int i =0;i<M;i++){
        for(int j =0;j<N;j++)
            cout<<A_h[i*N+j]<<" ";
        cout<<endl;
    }
    cout<<"\nB"<<endl;
    for(int i =0;i<N;i++){
        for(int j =0;j<M;j++)
            cout<<B_h[i*M+j] <<" ";
        cout<<endl;
    }
    cout<<"\nC"<<endl;
    for(int i =0;i<M;i++){
        for(int j =0;j<N;j++)
            cout<<C_h[i*N+j]<<" ";
        cout<<endl;
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}