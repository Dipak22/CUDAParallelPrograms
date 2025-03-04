#include<iostream>
#define TILE_WIDTH 2
#define WIDTH 4
using namespace std;

__global__ void matmul(int *A, int *B, int *P){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ int A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ int B_s[TILE_WIDTH][TILE_WIDTH];
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    int pValue = 0;
    for(int ph =0;ph<WIDTH/TILE_WIDTH;ph++){
        if(row<WIDTH && (ph*TILE_WIDTH + tx)<WIDTH)
            A_s[ty][tx] = A[row*WIDTH + ph*TILE_WIDTH +tx];
        else
            A_s[ty][tx] =0;
        if(col<WIDTH && (ph*TILE_WIDTH +ty)<WIDTH )
            B_s[ty][tx] = B[(ph*TILE_WIDTH +ty)*WIDTH + col];
        else
            B_s[ty][tx] = 0;
        __syncthreads();

        for(int k =0;k<TILE_WIDTH; k++)
            pValue +=A_s[ty][k] * B_s[k][tx];
        __syncthreads();
    }

    if(row<WIDTH && col<WIDTH)
        P[row*WIDTH + col] = pValue;
}

int main(){
    int *A,*B,*P;
    size_t size = WIDTH*WIDTH*sizeof(int);
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    P = (int*)malloc(size);
    int *A_d, *B_d, *P_d;
    for(int i=0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++){
            A[i*WIDTH + j] = rand()%5;
            B[i*WIDTH + j] = rand()%3;
        }
            
    }
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&P_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH);
    dim3 gridDim((WIDTH + TILE_WIDTH -1)/TILE_WIDTH,(WIDTH + TILE_WIDTH -1)/TILE_WIDTH);
    matmul<<<gridDim, blockDim>>>(A_d,B_d, P_d);
    cudaMemcpy(P,P_d, size, cudaMemcpyDeviceToHost);
    cout<<"A\n";
    for(int i=0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++){
            cout<<A[i*WIDTH + j]<<" ";
            
        }
        cout<<"\n";
            
    }
    cout<<"B\n";
    for(int i=0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++){
            cout<<B[i*WIDTH + j]<<" ";
            
        }
        cout<<"\n";
            
    }
    cout<<"P\n";
    for(int i=0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++){
            cout<<P[i*WIDTH + j]<<" ";
            
        }
        cout<<"\n";
            
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(P_d);

    free(A);
    free(B);
    free(P);
    return 0;
}