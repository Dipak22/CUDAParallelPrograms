#include<iostream>
using namespace std;

__global__ void calc_hist(char *s, unsigned int* hist, int length){
    int idx = blockIdx.x* blockDim.x + threadIdx.x;
    if(idx<length){
        int hist_idx = s[idx] - 'a';
        if(hist_idx>=0 && hist_idx<26)
            atomicAdd(&hist[hist_idx/4], 1);
    }
}

int main(){
    string s = "programming massively parallel processors aaaa  zzzzz";
    int length = s.length();
    char *str;
    unsigned int *hist_d;
    unsigned int *hist_h;
    int hist_length = (int)26/4 +1;
    hist_h = (unsigned int*)malloc(hist_length* sizeof(int));
    memset(hist_h, 0, hist_length*sizeof(int));
    cudaMalloc((void**)&str, length*sizeof(char));
    cudaMalloc((void**)&hist_d, hist_length*sizeof(int));
    cudaMemcpy(str, s.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(hist_d, 0, hist_length * sizeof(int));
    dim3 dimBlock(256,1,1);
    dim3 dimGrid(ceil(length/256.0),1,1);
    calc_hist<<<dimGrid,dimBlock>>>(str,hist_d, length);
    cudaMemcpy(hist_h, hist_d,hist_length*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<hist_length;i++)
        cout<<hist_h[i]<<" ";

    cudaFree(str);
    cudaFree(hist_d);
    free(hist_h);
    return 0;
}