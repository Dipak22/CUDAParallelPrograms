#include<iostream>
using namespace std;
#define R 1
#define OUT_TILE_WIDTH 16
#define FILTER_WIDTH (2*R+1)
#define IN_TILE_WIDTH (OUT_TILE_WIDTH + FILTER_WIDTH - 1)
#define WIDTH 5

__constant__ float F[FILTER_WIDTH][FILTER_WIDTH];


__global__ void conv2D(float *N, float *P){
    int row = blockIdx.y* OUT_TILE_WIDTH + threadIdx.y -R;
    int col = blockIdx.x*OUT_TILE_WIDTH + threadIdx.x -R;
    __shared__ float Ns[IN_TILE_WIDTH][IN_TILE_WIDTH];
    if(row>=0 && row<WIDTH && col>=0 && col<WIDTH)
        Ns[threadIdx.y][threadIdx.x] = N[row*WIDTH + col];
    else
        Ns[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    int tileRow = threadIdx.y - R;
    int tileCol = threadIdx.x - R;
    if(row>=0 && row<WIDTH && col>=0 && col<WIDTH){
        if(tileRow>=0 && tileRow<OUT_TILE_WIDTH && tileCol>=0 && tileCol<OUT_TILE_WIDTH){
            float pValue = 0.0f;
            for(int r=0;r<FILTER_WIDTH;r++){
                for(int c=0;c<FILTER_WIDTH;c++){
                    pValue +=Ns[tileRow+r][tileCol+c] * F[r][c];
                }
            }
            P[row*WIDTH + col] = pValue;
        }
        
    }

}

int main(){
    float *N, *F_h, *P;
    N = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    P = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    F_h = (float*)malloc(FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    for(int i =0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++)
            N[i*WIDTH+j] = rand()%3;
    }
    for(int i=0;i<FILTER_WIDTH;i++){
        for(int j =0;j<FILTER_WIDTH;j++)
            F_h[i*FILTER_WIDTH+j] = rand()%3;
    }

    float *N_d,  *P_d;
    cudaMalloc((void**)&N_d, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void**)&P_d, WIDTH * WIDTH * sizeof(float));

    cudaMemcpy(N_d, N, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, F_h, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH);
    dim3 gridDim((WIDTH + IN_TILE_WIDTH -1)/IN_TILE_WIDTH , (WIDTH + IN_TILE_WIDTH -1)/IN_TILE_WIDTH);
    conv2D<<<gridDim, blockDim>>>(N_d, P_d);
    cudaMemcpy(P,P_d, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"N\n";
    for(int i =0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++)
            cout<<N[i*WIDTH+j]<<" ";
        cout<<"\n";
    }
    cout<<"F\n";
    for(int i=0;i<FILTER_WIDTH;i++){
        for(int j =0;j<FILTER_WIDTH;j++)
            cout<<F_h[i*FILTER_WIDTH+j]<<" ";
        cout<<"\n";
    }
    cout<<"P\n";
    for(int i =0;i<WIDTH;i++){
        for(int j = 0; j<WIDTH;j++)
            cout<<P[i*WIDTH+j]<<" ";
        cout<<"\n";
    }

    cudaFree(N_d);
    cudaFree(P_d);
    cudaFree(F);

    free(N);
    free(P);
    free(F_h);

    return 0;
}

