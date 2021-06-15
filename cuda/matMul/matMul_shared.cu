#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double eps = 1e-7;
__global__ void MatMul(double *A,double *B,double *C,int n,int ch=32)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	//显式声明共享内存a，b子矩阵块
	__shared__ double shareA[32][32];
	__shared__ double shareB[32][32];
	double  sum = 0;

	//计算矩阵乘法 Kahan’s Summation Formula
	for (int m = 0; m < n/ch; m++)
	{
		// load data from global memory to shared memory
		shareA[threadIdx.y][threadIdx.x] = A[row * n + (m * ch + threadIdx.x)];
		shareB[threadIdx.y][threadIdx.x] = B[(m * ch + threadIdx.y) * n + col];
		// sync to wait for all threads in one block to finish loading datas
		__syncthreads();

		for (int i = 0; i < ch; i++)
		{
			sum += shareA[threadIdx.y][i] * shareB[i][threadIdx.x];

		}
		// sync to wait for all threads in one block to finish compute
		__syncthreads();
	}
	// store results into global memory
	if (row < n && col < n)
		C[row * n + col] = sum;
    //  double __shared__  Mds[2560];
    //  double __shared__  Nds[2560];
    // int x = threadIdx.x + blockIdx.x * blockDim.x;
	// int y = threadIdx.y + blockIdx.y * blockDim.y;
	// // printf("A[%2d,%2d] %lf \n",x,y,A[x*N+y]);
	// // printf("B[%2d,%2d] %lf \n",x,y,B[x*N+y]);
    // double elem1,elem2,value=0;
    // for(int k = 0; k < N; k++)
	// {
	// 	Mds[k] = A[x * N + k ];			//取M矩阵的一行
	// 	Nds[k] = B[k * N + y ];
	// }
    //  __syncthreads();
    // //  for(int k = 0; k < N; k++)
	// // {

	// // 	elem1 = Mds[k];			//取M矩阵的一行
	// // 	elem2 = Nds[k];			//取N矩阵的一列
		
	// // 	value += elem1 * elem2;	//求和
	// // }
	
	// for(int k = 0; k < N; k++)
	// {
	// 	value +=  Mds[k]* Nds[k];	//求和
	// }
    // __syncthreads();
	// C[x * N + y] = value;
}


int main()
{
	// const int N = pow(2,14);
	//const int N = 2560;
    const int N = 1280;
    const int M = 32;
    double *A,*B,*C;
    printf("MatrixSize N:[%d]\n",N);

	printf("\r分配CPU空间..");
    //矩阵A的内存空间分配
	A=(double*)malloc(sizeof(double*) * N * N);		//分配二维数组

    //向量b的内存空间分配
	B=(double*)malloc(sizeof(double*) * N * N);		//分配二维数组

    //矩阵C的内存空间分配
	C=(double*)malloc(sizeof(double*) * N * N);		//分配二维数组

	printf("\r初始化矩阵..");
    //矩阵A的初始化
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
        {
            A[i*N+j] = i-0.1*j+1;
            B[i*N+j] = j-0.1*i+1;
        }
    }

	printf("\r分配GPU空间..");
    double *Dev_A,*Dev_B,*Dev_C;	//定义GPU内存指针
	//设备端内存分配
	cudaMalloc((void**)&Dev_A, N * N * sizeof(double));
	cudaMalloc((void**)&Dev_B, N * N * sizeof(double));	
    cudaMalloc((void**)&Dev_C, N * N * sizeof(double));


	printf("\r内存拷贝..");
	//数据拷贝，主机到设备
	cudaMemcpy(Dev_A, A, N * N * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_B, B, N * N * sizeof(double),cudaMemcpyHostToDevice);


	printf("\r开始计算..\r");

	
    float esp_time_gpu;
	clock_t start_gpu, stop_gpu;
    start_gpu = clock();// start timing
    
    dim3 block(M,M);
	dim3 grid(N/M,N/M);
	MatMul<<<grid,block>>>(Dev_A,Dev_B,Dev_C,N,M);//调用核函数
	cudaMemcpy(C,Dev_C,N * N * sizeof(double),cudaMemcpyDeviceToHost);
	
	// cudaError_t cudaError = cudaGetLastError();
    // printf("CUDA error: %s\n", cudaGetErrorString(cudaError));

	cudaDeviceSynchronize(); // synchronzie
    stop_gpu = clock();// end timing
	esp_time_gpu = (float)(stop_gpu - start_gpu) / CLOCKS_PER_SEC * 1000;

	printf("GPU计算完成..  ");
	printf("运行时间:%f(ms)\n",esp_time_gpu);

	double *H_C=(double*)malloc(sizeof(double) * N * N);
	// for (int i = 0; i < N; ++i){
    //     for (int k = 0; k < N; ++k){
    //         H_C[i]+=A[i][k]*B[k];
    //     }
    // }

    float esp_time_cpu;
	clock_t start_cpu, stop_cpu;
    start_cpu = clock();// start timing

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            double value = 0.0;
            for (int k = 0; k < N; k++)
                value = value + A[i * N + k ] * B[k * N + j];
            H_C[i * N + j] = value;
        }
    
	stop_cpu = clock();// end timing
	esp_time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC * 1000;
	printf("CPU计算完成..  ");
	printf("运行时间:%f(ms)\n",esp_time_cpu);

    // printf("\nGPU 矩阵:\n");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%9lf  ", C[i * N + j]);
    //     }
    //     printf("\n");
    // }


    // printf("\nCPU 矩阵:\n");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%9lf  ", H_C[i * N + j]);
    //     }
    //     printf("\n");
    // }



	bool flag = true;
    for (int i = 0; i < N; ++i){
        double a=H_C[i];
        double b=C[i];
        if (fabs(a-b)>eps)
        {
            flag = false;
            printf("[%lf %lf]%d  ",a,b,i);
			//break;
        }
    }
    if (flag == true)
        printf("Result Correct\n");
    else{
        printf("Resul Wrong\n");
    }

	//释放GPU内存
	cudaFree(Dev_A);
	cudaFree(Dev_B);
	cudaFree(Dev_C);
	
	//释放CPU内存
    free(A);
    free(B);
	free(C);
	free(H_C);
	return 0;
}
