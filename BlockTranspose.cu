#include<iostream>

#define BLOCK_WIDTH 2
#define WIDTH 4
#define HEIGHT 4

using namespace std;

__global__ void blockTranspose(int *A, int *B) {
    // Calculate global memory indices
    int x = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    int y = blockIdx.y * BLOCK_WIDTH + threadIdx.y;

    // Perform transposition directly in global memory
    if (x < WIDTH && y < HEIGHT) {
        int index_in = y * WIDTH + x;
        int index_out = x * WIDTH + y;  // Transposed index

        B[index_out] = A[index_in];  // No shared memory needed
    }
}

int main(){
    size_t size = WIDTH * HEIGHT * sizeof(int);
    int *A = (int*)malloc(size);
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDTH;j++)
            A[i*WIDTH +j] = rand()%5;
    }
    cout<<"A"<<endl;
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDTH;j++)
            cout<<A[i*WIDTH +j]<<" ";
        cout<<"\n";
    }
    int *A_d, *B_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMemcpy(A_d,A, size, cudaMemcpyHostToDevice);
    dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
    dim3 gridDim(WIDTH/blockDim.x,HEIGHT/blockDim.y);
    blockTranspose<<<gridDim,blockDim>>>(A_d,B_d);
    cudaMemcpy(A,B_d,size, cudaMemcpyDeviceToHost);
    cout<<"Transposed A"<<endl;
    for(int i=0;i<HEIGHT;i++){
        for(int j=0;j<WIDTH;j++)
            cout<<A[i*WIDTH +j]<<" ";
        cout<<"\n";
    }
    cudaFree(A_d);
    free(A);

    return 0;
}