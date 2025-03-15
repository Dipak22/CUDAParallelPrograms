#include<iostream>
#define LENGTH 4096
#define BLOCKDIM 512
#define COARSE_FACTOR 2
using namespace std;

__global__ void sumReduce(int *arr, int *out){
   __shared__ int arr_s[BLOCKDIM];
   int segment = COARSE_FACTOR* 2 * blockDim.x * blockIdx.x;
   int idx = threadIdx.x;
   int i = segment + idx;
   float sum = arr[i];
   for(int tile = 1; tile<COARSE_FACTOR*2;tile++)
        sum +=arr[i + tile*BLOCKDIM];
   arr_s[idx] = sum;
   for(int stride = blockDim.x/2; stride>=1; stride /=2){
        __syncthreads();
       if(threadIdx.x<stride){
           arr_s[idx] +=arr_s[idx+stride];
       }
   }
   if(threadIdx.x==0)
       atomicAdd(&out[0], arr_s[0]);
}

int main(){
   int *arr, *sum;
   arr = (int*)malloc(LENGTH * sizeof(int));
   sum = (int*)malloc(sizeof(int));
   for(int i=0;i<LENGTH;i++)
       arr[i] = i+1;
   int *arr_d, *output;
   cudaMalloc((void**)&arr_d, LENGTH * sizeof(int));
   cudaMalloc((void**)&output, sizeof(int));
   cudaMemcpy(arr_d, arr, LENGTH*sizeof(int), cudaMemcpyHostToDevice);
   dim3 dimBlock(BLOCKDIM);
   dim3 dimGrid((LENGTH + BLOCKDIM -1)/BLOCKDIM);
   sumReduce<<<dimGrid,dimBlock>>>(arr_d, output);
   cudaMemcpy(sum, output, sizeof(int), cudaMemcpyDeviceToHost);
   cout<<sum[0];

   cudaFree(arr_d);
   cudaFree(output);

   free(arr);
   free(sum);

   return 0;
}