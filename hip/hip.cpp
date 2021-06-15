#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define M 4
#define K 4
#define N 4

void initial(double* list,int row,int col)
{
    double *num = list;
    for (int i=0; i<row*col; i++)
    {
        num[i] = rand()%10;
    }
}

void CpuMatrix(double *A,double *B,double *C)
{
    int i,j,k;

    for( i=0; i<M; i++)
    {
        for(j=0; j<N; j++)
        {
            double sum = 0;
            for(int k=0; k<K; k++)
            {
                sum += A[i*K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void GpuMatrix(double *dev_A,double *dev_B,double *dev_C)
{
    int ix = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int iy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(ix<K && iy<M)
    {
    double sum = 0;
    for( int k = 0; k < K;k++)
    {
        sum += dev_A[iy*K + k] * dev_B[k*N + ix];
    }
    dev_C[iy * N + ix] = sum;
}
}

void printMatrix(double *list,int row,int col)
{
    double *p = list;
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            printf("%10lf",p[j]);
        }
        p = p + col;
        printf("\n");
    }
}
int main(int argc,char **argv)
{
    int Axy = M*K;
    int Abytes = Axy * sizeof(double);

    int Bxy = K*N;
    int Bbytes = Bxy * sizeof(double);

    int nxy = M*N;
    int nbytes = nxy * sizeof(double);
    
    float time_cpu,time_gpu;
    
    clock_t start_cpu,stop_cpu;
    
    hipEvent_t start_GPU,stop_GPU;

    double *host_A, *host_B, *host_C, *c_CPU;
    host_A = (double*)malloc(Abytes);
    host_B = (double*)malloc(Bbytes);
    host_C = (double*)malloc(nbytes);
    c_CPU = (double*)malloc(nbytes);


    initial(host_A,M,K);

    printf("A:(%d,%d):\n",M,K);
    printMatrix(host_A,M,K);

    initial(host_B,K,N);

    printf("B:(%d,%d):\n",K,N);
    printMatrix(host_B,K,N);

// start_cpu = clock();
    CpuMatrix(host_A,host_B,host_C);
//   stop_cpu = clock();

    printf("Host_C:(%d,%d):\n",M,N);
// printf("\nCPU time is %f(ms)\n",(float)(stop_cpu-start_cpu)/CLOCKS_PER_SEC);
    printMatrix(host_C,M,N);
    double *dev_A,*dev_B,*dev_C;
    hipMalloc(&dev_A,Axy*sizeof(double));
    hipMalloc(&dev_B,Bxy*sizeof(double));
    hipMalloc(&dev_C,nxy*sizeof(double));

    dim3 block(1024,1);
    dim3 grid(64,64);

    hipMemcpy(dev_A,host_A,Abytes,hipMemcpyDeviceToHost);
    hipMemcpy(dev_B,host_B,Bbytes,hipMemcpyDeviceToHost);

    hipEventCreate(&start_GPU);
    hipEventCreate(&stop_GPU);
    hipEventRecord(start_GPU,0);
    hipLaunchKernelGGL(GpuMatrix,grid,block,0,0,dev_A,dev_B,dev_C);
    hipEventRecord(stop_GPU,0);
    hipEventSynchronize(start_GPU);
    hipEventSynchronize(stop_GPU);
    hipEventElapsedTime(&time_gpu, start_GPU,stop_GPU);
    printf("\nThe time from GPU:\t%f(ms)\n", time_GPU/1000);
    hipDeviceSynchronize();
    hipEventDestroy(start_GPU);
    hipEventDestroy(stop_GPU);

    hipMemcpy(c_CPU,dev_C,nbytes,hipMemcpyDeviceToHost);
    printf("device_C:(%d,%d):\n",M,N);
    printMatrix(c_CPU,M,N);


    hipFree(dev_A);
    hipFree(dev_B);
    hipFree(dev_C);
    free(host_A);
    free(host_B);
    free(host_C);
    free(c_CPU);

    return 0;
}