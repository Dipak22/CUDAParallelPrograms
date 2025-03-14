#include<iostream>
#define NUM_BINS (26/4 + 1)
using namespace std;

__global__ void calculate_hist(char *str, int* hist, int length){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ int hist_s[NUM_BINS];

    if(threadIdx.x<NUM_BINS){
        hist_s[threadIdx.x] = 0;
    }
    __syncthreads();
    if(idx<length){
        int chIdx = str[idx] - 'a';
        if(chIdx>=0 && chIdx<26)
            atomicAdd(&(hist_s[chIdx/4]), 1);
    }
    __syncthreads();
    //commit to global memory
    for(int bin = threadIdx.x; bin<NUM_BINS;bin+=blockDim.x){
        int val = hist_s[bin];
        if(val>0){
            atomicAdd(&hist[bin], val);
        }
    }
}

int main(){
    string s = "programming massively parallel processors";
    //string s = "aaaaaa eeeeee zzzzzzz";

    int length = s.length();
    int* hist = (int*)malloc(NUM_BINS * sizeof(int));
    memset(hist, 0, NUM_BINS* sizeof(int));
    char *str;
    int *hist_d;
    cudaMalloc((void**)&str, length * sizeof(char));
    cudaMalloc((void**)&hist_d, NUM_BINS * sizeof(int));
    cudaMemset(hist_d, 0 , NUM_BINS*sizeof(int));
    cudaMemcpy(str, s.c_str(), length* sizeof(char), cudaMemcpyHostToDevice);
    dim3 dimBlock(32);
    dim3 dimGrid(ceil(length/32.0));
    calculate_hist<<<dimGrid, dimBlock>>>(str, hist_d, length);
    cudaMemcpy(hist, hist_d, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i =0;i<NUM_BINS;i++)
        cout<<hist[i]<<" ";

    cudaFree(str);
    cudaFree(hist_d);
    free(hist);
    return 0;
}