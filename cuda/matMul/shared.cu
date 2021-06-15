#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern __shared__ double share_b[];

__global__ void MatMul(double *A,double *b,double *C,const int N,int num)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;	

	for (int i = 0; i < 4096; i++)
	share_b[i] = b[i + num * 4096];
	
	__syncthreads();

	double elem1,elem2,value=0;
	for(int k=0 ; k<4096 ; k++){
		elem1 = A[x * N + k +num *4096];
		elem2 = share_b[k];
		
		value += elem1 * elem2;	//求和
	}

	C[x]+=value;

}
 
int main()
{
	const int N = pow(2,14);
    
    double **A,*b,*C;
    //定义GPU内存指针
    double *Dev_A,*Dev_b,*Dev_C;

	printf("分配CPU内存空间..\n");
    //矩阵A的内存空间分配
	A=(double**)malloc(sizeof(double*) * N);		//分配二维数组
    A[0]=(double*)malloc(sizeof(double) * N * N);	//分配一维数组
    for(int i=1;i<N;i++) A[i]=A[i-1]+N;

    //向量b的内存空间分配
    b=(double*)malloc(sizeof(double) * N);

    //矩阵C的内存空间分配
	C=(double*)malloc(sizeof(double) * N);

	printf("初始化矩阵..\n");
    //矩阵A的初始化
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
        {
            A[i][j] = i-0.1*j+1;
        }
        b[i]=log(sqrt(i*i-i+2));
    }


	//使用event计算时间
	cudaEvent_t start,stop;
	float elapsedTime = 0;
	//创建Event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("分配GPU内存空间..\n");
	//设备端内存分配
	cudaMalloc((void**)&Dev_A, N * N * sizeof(double));
	cudaMalloc((void**)&Dev_b, N * sizeof(double));
	cudaMalloc((void**)&Dev_C, N * sizeof(double));

	cudaEventRecord(start,0);

	printf("内存拷贝..\n");
	//数据拷贝，主机到设备
	cudaMemcpy(Dev_A, A[0], N * N * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_b, b, N * 1 * sizeof(double),cudaMemcpyHostToDevice);
	
	printf("开始计算..\n");

	dim3 block(16);
	dim3 grid(N/block.x);

	for (int i = 0; i < N / 4096; ++i)
    {
        dim3 block(32);
        dim3 grid(N / block.x);
        MatMul<<<grid, block, 4096 * sizeof(double)>>>(Dev_A, Dev_b, Dev_C, N, i);
    }
	cudaMemcpy(C,Dev_C,N * sizeof(double),cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop,0);

	// cudaError_t cudaError = cudaGetLastError();
    // printf("CUDA error: %s\n", cudaGetErrorString(cudaError));

	cudaDeviceSynchronize();

	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime,start,stop);

	cudaEventDestroy(start);    
	cudaEventDestroy(stop);
	

	printf("计算完成..\n");

	printf("运行时间:%f(ms)\n",elapsedTime);

	double *test_c=(double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; ++i){

        for (int k = 0; k < N; ++k)
         {
            test_c[i]+=A[i][k]*b[k];
         }
    }

	bool flag = true;
    for (int i = 0; i < N; ++i){
        float a=test_c[i];
        float b=C[i];
        if (a!=b)
        {
            flag = false;
			break;
        }
    }
    if (flag == true)
        printf("result correct\n");
    else{
        printf("resul wrong\n");
    }

	//释放GPU内存
	cudaFree(Dev_A);
	cudaFree(Dev_b);
	cudaFree(Dev_C);
	
	//释放CPU内存
	free(A[0]);
    free(A);
    free(b);
	free(C);

	return 0;
}