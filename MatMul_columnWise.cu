#include<iostream>
using namespace std;
#define WIDTH 4

__global__ void matmul(int *A, int *B,int *C){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(col<WIDTH){
        for(int r=0;r<WIDTH;r++){
            int result = 0;
            for(int k = 0;k<WIDTH;k++)
                result +=A[r*WIDTH+k] * B[k*WIDTH+col];
            C[r*WIDTH + col] = result;
        }
    }
    
}
int main(){
    int *A, *B, *C;
    int memsize = WIDTH * WIDTH * sizeof(int);
    C = (int*)malloc(memsize);
    A = (int*)malloc(memsize);
    B = (int*)malloc(memsize);
    for(int i=0;i<WIDTH;i++){
        for (int j = 0;j<WIDTH;j++){
            A[i*WIDTH +j] = rand()%5;
            B[i*WIDTH + j] = rand()%5;
        }
    }

    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, memsize);
    cudaMalloc((void**)&B_d, memsize);
    cudaMalloc((void**)&C_d, memsize);

    cudaMemcpy(A_d, A, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, memsize, cudaMemcpyHostToDevice);

    matmul<<<ceil(WIDTH/16.0),16>>>(A_d,B_d,C_d);

    cudaMemcpy(C,C_d,memsize, cudaMemcpyDeviceToHost);
    cout<<"A"<<endl;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++)
            cout<<A[i*WIDTH +j]<<"  ";
        cout<<"\n";
    }
    cout<<"B"<<endl;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++)
            cout<<B[i*WIDTH +j]<<"  ";
        cout<<"\n";
    }
    cout<<"C"<<endl;
    for(int i=0;i<WIDTH;i++){
        for(int j=0;j<WIDTH;j++)
            cout<<C[i*WIDTH +j]<<"  ";
        cout<<"\n";
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);

    return 0;
}