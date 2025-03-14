#include<iostream>
using namespace std;


#define NUM_BINS (26/4 + 1)
#define COARSE_FACTOR 3
__global__ void hist_private(char *str, int *hist, int length){
    __shared__ int hist_s[NUM_BINS];
    for(int i = threadIdx.x; i<NUM_BINS;i+=blockDim.x)
        hist_s[i] = 0;
    __syncthreads();
    //populate the shared hist
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx*COARSE_FACTOR; i<min(length,(idx+1)*COARSE_FACTOR); i++){
        int chIdx = str[i] - 'a';
        if(chIdx>=0 && chIdx<26){
            atomicAdd(&(hist_s[chIdx/4]), 1);
        }
    }
    __syncthreads();
    //populate the global histogram
    for(int i = threadIdx.x;i<NUM_BINS;i+=blockDim.x){
        int val = hist_s[i];
        if(val>0)
            atomicAdd(&hist[i], val);
    }


}

int main(){
    string s = "programming massively parallel processors";
    int length = s.length();
    int* hist = (int*)malloc(NUM_BINS * sizeof(int));
    memset(hist, 0, NUM_BINS* sizeof(int));
    char *str;
    cudaMalloc((void**)&str, length * sizeof(char));
    cudaMemcpy(str,s.c_str(),length*sizeof(char), cudaMemcpyHostToDevice);
    int *hist_d;
    cudaMalloc((void**)&hist_d, NUM_BINS*sizeof(int));
    cudaMemset(hist_d, 0 , NUM_BINS * sizeof(int));
    dim3 dimBlock(32);
    dim3 dimGrid(ceil(length/32.0));
    hist_private<<<dimGrid,dimBlock>>>(str,hist_d, length);
    cudaMemcpy(hist, hist_d, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<NUM_BINS;i++)
        cout<<hist[i]<<" ";

    cudaFree(hist_d);
    cudaFree(str);
    free(hist);
    return 0;
}