#include<iostream>
using namespace std;

#define TILE_WIDTH 4
#define COARSE_FACTOR 4
#define WIDTH (4 * TILE_WIDTH)

__global__ void matmul(float *A, float *B, float *P) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR]; // Accommodate coarsening

    int row = by * TILE_WIDTH + ty;
    int start_col = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float pValue[COARSE_FACTOR] = {0};

    for (int ph = 0; ph < WIDTH / TILE_WIDTH; ph++) {
        // Load A into shared memory
        int indexA = row * WIDTH + (ph * TILE_WIDTH + tx);
        M[ty][tx] = (indexA < WIDTH * WIDTH) ? A[indexA] : 0;

        // Load multiple elements of B into shared memory
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = start_col + c * TILE_WIDTH;
            int indexB = (ph * TILE_WIDTH + ty) * WIDTH + col;
            N[ty][tx + c * TILE_WIDTH] = (indexB < WIDTH * WIDTH) ? B[indexB] : 0;
        }

        __syncthreads(); // Ensure all threads have loaded data

        // Perform matrix multiplication
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int c = 0; c < COARSE_FACTOR; c++) {
                pValue[c] += M[ty][k] * N[k][tx + c * TILE_WIDTH];
            }
        }

        __syncthreads(); // Ensure no overwrites in next iteration
    }

    // Store results in output matrix
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = start_col + c * TILE_WIDTH;
        if (row < WIDTH && col < WIDTH) {
            P[row * WIDTH + col] = pValue[c];
        }
    }
}


int main(){
    float *A,*B,*C;
    size_t size = WIDTH * WIDTH * sizeof(float);
    A  = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for(int i =0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            A[i*WIDTH+j] = rand()%3;
            B[i*WIDTH+j] = rand()%3;
        }
    }

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
    dim3 dimGrid((WIDTH+TILE_WIDTH-1)/TILE_WIDTH,(WIDTH+TILE_WIDTH-1)/TILE_WIDTH);
    matmul<<<dimGrid, dimBlock>>>(A_d,B_d,C_d);
    cudaMemcpy(C,C_d, size, cudaMemcpyDeviceToHost);
    cout<<"A\n";
    for(int i =0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            cout<<A[i*WIDTH+j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"B\n";
    for(int i =0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            cout<<B[i*WIDTH+j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"C\n";
    for(int i =0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            cout<<C[i*WIDTH+j]<<" ";
        }
        cout<<"\n";
    }

    bool isCorrect = true;
    int idx = 0;
    cout<<"CPU Answer\n";
    for(int i =0;i<WIDTH;i++){
        for(int j =0;j<WIDTH;j++){
            float pValue = 0;
            for(int k=0;k<WIDTH;k++){
                pValue +=A[i*WIDTH +k] * B[k*WIDTH + j];
            }
            if(pValue != C[i*WIDTH+j]){
                idx = i*WIDTH+j;
                isCorrect = false;
            }
            cout<<pValue<<" ";
        }
        cout<<"\n";
    }
    if(!isCorrect)
        cout<<"\n GPU calculation incorrect at index "<<idx;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C);
    return 0;
}