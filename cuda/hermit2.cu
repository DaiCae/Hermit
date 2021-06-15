#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double eps = 1e-8;

int QR_improve(double *A, int N)
{
    double q = (A[N * (N - 2) + N - 2] - A[N * N - 1]) / 2;
    double p;
    if (q >= eps)
    {
        p = A[N * N - 1] + q - sqrtl(q * q + A[N * (N - 1) - 1] * A[N * (N - 1) - 1]);
    }
    else
        p = A[N * N - 1] + q + sqrtl(q * q + A[N * (N - 1) - 1] * A[N * (N - 1) - 1]);
    
    double *Q = new double[N * N];
    double *R = new double[N * N];
    for (int k = 0; k < (N - 1)/2; k++)
    {
        if (fabs(A[k * N + (k + 1)]) <= eps)
        {
            continue;
        }
        double elem1 = (A[k * N + k] - p) * (A[k * N + k] - p);
        double elem2 = A[(k + 1) * N + k] * A[(k + 1) * N + k];
        double r = sqrt(elem1 + elem2);

        //printf("%9lf  %9lf  %9lf\n", A[k*N+k] ,A[(k+1)*N+k],r);
        double Cos = (A[k * N + k] - p) / r;
        double Sin = A[(k + 1) * N + k] / r;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                R[i * N + j] = 0;
                if (i == j)
                    R[i * N + j] = 1;
            }
        }

        R[k * N + k] = Cos;
        R[k * N + (k + 1)] = Sin;
        R[(k + 1) * N + k] = Sin * -1;
        R[(k + 1) * N + (k + 1)] = Cos;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                Q[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    Q[i * N + j] += R[i * N + k] * A[k * N + j];
            }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    A[i * N + j] += Q[i * N + k] * R[k + j * N];
            }
    }

    delete[] Q;
    delete[] R;

    // return flag;
    return 1;
}

__global__ void MatMul(double *A, double *B, double *C, int n,int ch)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	//显式声明共享内存a，b子矩阵块
	__shared__ double shareA[32][32];
	__shared__ double shareB[32][32];
	double  y = 0;

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
			y += shareA[threadIdx.y][i] * shareB[i][threadIdx.x];

		}
		// sync to wait for all threads in one block to finish compute
		__syncthreads();
	}
	if (row < n && col < n)
		C[row * n + col] = y;
	// int x = threadIdx.x + blockIdx.x * blockDim.x;
	// int y = threadIdx.y + blockIdx.y * blockDim.y;
	// // printf("A[%2d,%2d] %lf \n",x,y,A[x*N+y]);
	// // printf("B[%2d,%2d] %lf \n",x,y,B[x*N+y]);
	// double elem1,elem2,value=0;
	// for(int k = 0; k < N; k++)
	// {
	// 	elem1 = A[x * N + k ];			//取M矩阵的一行
	// 	elem2 = B[k * N + y ];			//取N矩阵的一列	
	// 	value += elem1 * elem2;	//求和
	// }
	// C[x * N + y] = value;
}

extern __shared__ double share_alpha[];
__global__ void Householder_gpu(double *A, double *B, double * alpha, int N, int k)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	//if(x>=N || y>=N) return;
	
	//初始化alpha向量
	if(y==0) 
	{
		alpha[x]=0;
		if(x < k) alpha[x] = A[k * N + x];
	}
	__syncthreads();

	double mol,sum=0;
	if(x==0 && y==0){
		//求出向量mol
		for(int i=0;i<k;i++){
			sum+=alpha[i] * alpha[i];
		}
		mol=sqrt(sum);
		//printf("[%10lf]\n",mol);
		if(alpha[k-1] > 0) mol=-mol;
		sum -= alpha[k-1] * mol;
		alpha[k-1]=alpha[k-1] -mol;
		//printf("[%10lf]\n",sum);
	}	
	__syncthreads();

	// B[x * N + y] = -1 * alpha[x] * alpha[y] / sum;
	if(x==0 && y==0){
		for (int i = 0; i <= N - 1; i++)
		{
			for (int j = 0; j <= N - 1; j++)
				B[i * N + j] = -1 * alpha[i] * alpha[j] / sum;
			B[i*N+i]+=1;
		}
	}
}

int Householder_cpu(int N, double * Dev_A, double * Dev_B, double * Dev_C )
{	
	printf("开始计算..");
    //使用event计算时间
	cudaEvent_t start,stop;
	float elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	//=====================================================================
	double *Dev_alpha;
	cudaMalloc((void**)&Dev_alpha, N * sizeof(double));

	for (int k = N - 1; k > 1; k--)
    {
		dim3 block(32,32);
		dim3 grid( N / block.x , N / block.y );
		//dim3 grid(N/block.x);

		//计算H矩阵
		//Householder_gpu<<<grid,block,N * sizeof(double)>>>(Dev_A,Dev_B,N,k);//调用核函数
		Householder_gpu<<<grid,block>>>(Dev_A,Dev_B,Dev_alpha,N,k);//调用核函数

		//A左乘B保存到C
		MatMul<<<grid,block>>>(Dev_A,Dev_B,Dev_C,N,32);//调用核函数
		MatMul<<<grid,block>>>(Dev_B,Dev_C,Dev_A,N,32);//调用核函数

		// Householder<<<grid,block>>>(Dev_A,Dev_B,Dev_C,N);//调用核函数
		//break;
	}

	//=====================================================================
	// cudaError_t cudaError = cudaGetLastError();
    // printf("CUDA Errors: %s\n", cudaGetErrorString(cudaError));

	cudaEventRecord(stop,0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);    
	cudaEventDestroy(stop);
	
	printf("\r运行时间:%f(ms)\n",elapsedTime);
	return 1;
}

//进行一半的QR检查
int check(double *A, int N){
    // for (int x = 0; x < N/2-1; x++){
    //     for (int y = 0; y < N/2-1; y++){
    //         if(x==y) continue;
    //         if (fabs(A[x * N + y]) > eps)
    //             return 1;
    //     }
    // }
    // return 0;
    for (int x = 0; x < (N - 1)/2; x++){
        if (fabs(A[(x + 1) * N + x]) > eps)
            return 1;
    }
    return 0;
}

//打印结果矩阵
void show(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%12.8lf,", A[i * N + j]);
        }
        printf("\n");
    }
}


void symmat(double H[], int N)
{
    int n=N/2;
    double *A = new double[n * n];
    double *B = new double[n * n];

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        for(int j=0 ;j< n; j++){
            if(j >= i){
                A[i * n + j] = rand() % 20 -10;
                B[i * n + j] = rand() % 20 -10;
            }
            else{
                A[i * n + j] = A[j * n + i];
                B[i * n + j] = -1 * B[j * n + i];
            }
        }
        B[i*n+i]=0;
    }

    // show(A,n);
    // printf("\n");
    // show(B,n);
    // printf("\n");

    for (int i = 0; i < N; i++)
    {
        for(int j = 0;j <N ;j++)
        {
            if((i>=n&&j>=n)||(i<n && j<n)) 
                H[i*N+j]= A[i%n * n + j%n];
            else if(i<n)
                H[i*N+j]= -1 * B[i%n * n + j%n];
            else
                H[i*N+j]= B[i%n * n + j%n];
        }
    }
}

int main()
{
	printf("分配CPU内存空间..\n");
	int N =32;
	double *B = new double[N * N];
    double *C = new double[N * N];
    double *A = new double[N * N];
    symmat(A,N);
    // printf("\nGPU 矩阵:\n");
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < N; j++)
	// 		printf("%11lf  ", A[i * N + j]);
	// 	printf("\n");
    // }
	printf("分配GPU内存空间..\n");
	//定义GPU内存指针
	double *Dev_A,*Dev_B,*Dev_C;
	//设备端内存分配
	cudaMalloc((void**)&Dev_A, N * N * sizeof(double));
	cudaMalloc((void**)&Dev_B, N * N * sizeof(double));
	cudaMalloc((void**)&Dev_C, N * N * sizeof(double));
	printf("内存拷贝..\n");
	//数据拷贝，主机到设备
	cudaMemcpy(Dev_A, A, N * N * sizeof(double),cudaMemcpyHostToDevice);   
	printf("\n========================GPU=======================\n");
	Householder_cpu(N,Dev_A,Dev_B,Dev_C);
	double *A0 = new double[N * N];
	double *B0 = new double[N * N];
	//内存拷回
	cudaMemcpy(A0, Dev_A, N * N * sizeof(double),cudaMemcpyDeviceToHost);
	// printf("\nGPU 矩阵:\n");
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < N; j++)
	// 		printf("%11lf  ", A0[i * N + j]);
	// 	printf("\n");
    printf("\n=============================================================================\n");
    printf("QR 矩阵\n");
    int num = 0;
    while (check(A0, N)){
        QR_improve(A0, N);
        printf("\rNo.%d", ++num);
    }
    printf("\n");
    show(A0, N);
    printf("\n=============================================================================\n");
    double b[N];
    for (int i = 0; i < N; i++)
    {
        b[i] = A0[i * N + i];
    }
    //InsertSort(b,10);
    for (int i = 0; i < N; i++)
    {
        printf("b[%d]:%12.10lf \n", i, b[i]);
    }

	//释放GPU内存
	cudaFree(Dev_A);
	cudaFree(Dev_B);
	cudaFree(Dev_C);	
	//释放CPU内存
    // free(A);
    //free(B);
	free(C);

	return 0;
}