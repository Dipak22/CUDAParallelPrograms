 #include<iostream>
 #define LENGTH 2048
 using namespace std;

 __global__ void sumReduce(int *arr, int *out){
    int idx = 2*threadIdx.x;
    for(int stride = 1; stride<=blockDim.x; stride *=2){
        if(threadIdx.x%stride ==0){
            arr[idx] +=arr[idx+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x==0)
        out[0] = arr[0];
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
    dim3 dimBlock(1024);
    dim3 dimGrid(1);
    sumReduce<<<dimGrid,dimBlock>>>(arr_d, output);
    cudaMemcpy(sum, output, sizeof(int), cudaMemcpyDeviceToHost);
    cout<<sum[0];

    cudaFree(arr_d);
    cudaFree(output);

    free(arr);
    free(sum);

    return 0;
 }