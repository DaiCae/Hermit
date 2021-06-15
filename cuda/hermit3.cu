#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double eps = 1e-8;

//检查QR检查
int check(double *A, int N){
    for (int x = 0; x < N - 1; x++){
        if (fabs(A[(x + 1) * N + x]) > eps)
            return 1;
    }
    return 0;
}

//通过QR(Givens rotation)方法求出全部特征值 A为三对角矩阵 N为矩阵阶数
int QR(double *A, int N)
{

    double *Q = new double[N * N];
    double *R = new double[N * N];

    for (int k = 0; k < N - 1; k++)
    {
        if (fabs(A[k * N + (k + 1)]) <= eps)
        {
            continue;
        }

        double elem1 = A[k * N + k] * A[k * N + k];
        double elem2 = A[(k + 1) * N + k] * A[(k + 1) * N + k];
        double r = sqrt(elem1 + elem2);

        //printf("%9lf  %9lf  %9lf\n", A[k*N+k] ,A[(k+1)*N+k],r);
        double Cos = A[k * N + k] / r;
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

    // printf("\nA 矩阵:\n");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%9lf  ", A[i * N + j]);
    //     }
    //     printf("\n");
    // }

    delete[] Q;
    delete[] R;

    // return flag;
    return 1;
}

//改进的QR 
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


int Householder(int n, double A[])
{
	printf("\r开始计算..");
    //使用event计算时间
	cudaEvent_t start,stop;
	float elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	//=====================================================================
    double mol, q, value;
    double *alpha = new double[n];
    double *H = new double[n * n];
    double *B = new double[n * n];

    for (int i = n - 1; i > 1; i--)
    {
        q = 0.0;
        //初始化alpha向量为0
        for (int j = 0; j < n; j++)
            alpha[j] = 0.0;

        for (int j = 0; j < i; j++){
            alpha[j] = A[i * n + j];
            q += alpha[j] * alpha[j];
        }
        mol = sqrt(q);

        if (alpha[i - 1] > 0.0)
            mol = -mol;


        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;

        for (int j = 0; j <= n - 1; j++){
            for (int k = 0; k <= n - 1; k++)
                H[j * n + k] = -alpha[j] * alpha[k] / q;
            H[n * j + j] = H[n * j + j] + 1.0;
        }

        for (int j = 0; j < i+1; j++)
            for (int k = 0; k < i+1; k++)
            {
                value = 0.0;
                for (int u = 0; u < i+1; u++)
                    value = value + H[u + j * n] * A[n * u + k];
                B[k + j * n] = value;
            }

        for (int j = 0; j <  i+1; j++)
            for (int k = 0; k < i+1; k++)
            {
                value = 0.0;
                for (int u = 0; u < i+1; u++)
                    value = value + B[u + j * n] * H[n * u + k];
                A[k + j * n] = value;
            }

    }
    

    delete[] alpha;
    delete[] H;
    delete[] B;

	//=====================================================================
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


__global__ void MatMul(double *A, double *B, double *C, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// printf("A[%2d,%2d] %lf \n",x,y,A[x*N+y]);
	// printf("B[%2d,%2d] %lf \n",x,y,B[x*N+y]);

	double elem1,elem2,value=0;
	for(int k = 0; k < N; k++)
	{

		elem1 = A[x * N + k ];			//取M矩阵的一行
		elem2 = B[k * N + y ];			//取N矩阵的一列
		
		value += elem1 * elem2;	//求和
	}
	C[x * N + y] = value;
}

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

	if(x==0 && y==0){
		double mol,sum=0;
		//求出向量mol
		for(int i=0;i<k;i++){
			sum += alpha[i] * alpha[i];
		}
		mol=sqrt(sum);
		//printf("[%10lf]\n",mol);
		if(alpha[k-1] > 0) mol=-mol;
		sum -= alpha[k-1] * mol;
		alpha[k-1]=alpha[k-1] -mol;
		//printf("[%10lf]\n",sum);
		B[0]=sum;
	}	
	
	__syncthreads();
	
	double sum=B[0];
	//计算H矩阵
	B[x * N + y] = -1 * alpha[x] * alpha[y] /sum;
	//对角线元素加一
	if(x==y) B[x * N + y]+=1;

	// if(y==0) printf("%10lf  ",alpha[x]);
	// if(y==0 && x==0) printf("\n");
	// if(y==0 && x==0){
	// 	printf("\nH 矩阵:\n");
    //     for (int i = 0; i < N; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             printf("%10lf  ", B[i * N + j]);
    //         }
    //         printf("\n");
    //     }
	// }
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
		dim3 block(10,10);
		dim3 grid( N / block.x , N / block.y );
		//dim3 grid(N/block.x);

		//计算H矩阵
		//Householder_gpu<<<grid,block,N * sizeof(double)>>>(Dev_A,Dev_B,N,k);//调用核函数
		Householder_gpu<<<grid,block>>>(Dev_A,Dev_B,Dev_alpha,N,k);//调用核函数

		//A左乘B保存到C
		MatMul<<<grid,block>>>(Dev_A,Dev_B,Dev_C,N);//调用核函数
		MatMul<<<grid,block>>>(Dev_B,Dev_C,Dev_A,N);//调用核函数

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

__global__ void QR_gpu(double *A, double *B, double * alpha, int N, int k)
{
	//判断是否为需要的线程块
	int id = blockIdx.x;
	if (id % 4 != k ) return;
	
	//取得块矩阵索引
	int x = threadIdx.x;
	int y = threadIdx.y;

	__shared__ double Sub_A[4][4];

	double Cos =A[id * N + id];
	double Sin =A[ (id + 1) * N +id];
	double mol=sqrt(Cos * Cos + Sin * Sin);
	Cos = Cos / mol;
	Sin = Sin / mol;
	//[k , k]:Cos [k , k+1]:Sin
	//[k+1,k]:Cos [k+1,k+1]:Sin

	

	__syncthreads();


}

int QR_cpu(int N, double * Dev_A, double * Dev_B, double * Dev_C )
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

	dim3 block(4,4);
	dim3 grid(N-1); 

	for (int k = 0; k < 4; k++)
    {
		//计算H矩阵
		QR_gpu<<<grid,block>>>(Dev_A,Dev_B,Dev_alpha,N,k);//调用核函数
		// Householder<<<grid,block>>>(Dev_A,Dev_B,Dev_C,N);//调用核函数
		//break;
	}

	//=====================================================================
	cudaError_t cudaError = cudaGetLastError();
    printf("CUDA Errors: %s\n", cudaGetErrorString(cudaError));

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

int main()
{
	printf("分配CPU内存空间..\n");
    //double *A,*B,*C;


	// int N = 4;
    // double A[16] = {
	// 	4.0, 1.0, -2.0, 2.0, 
	// 	1.0, 2.0, 0.0, 1.0,
	// 	-2.0, 0.0, 3.0, -2.0,
	// 	2.0, 1.0, -2.0, -1.0
	// };

	int N=10;
    double A[100] = {1.0, 2.0, 3.0, 4.0, 5.0, 0.0, -1.0, -1.0, -1.0, -1.0,
                     2.0, 2.0, 3.0, 4.0, 6.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                     3.0, 3.0, 3.0, 1.0, 5.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                     4.0, 4.0, 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
                     5.0, 6.0, 5.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                     0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                     -1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0, 6.0,
                     -1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 5.0,
                     -1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 1.0, 3.0, 1.0,
                     -1.0, 0.0, 0.0, -1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 3.0};



	
	double *B = new double[N * N];
    double *C = new double[N * N];
	
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
    
	//QR计数
	int num = 0;

	printf("\n========================GPU=======================\n");
	Householder_cpu(N,Dev_A,Dev_B,Dev_C);
	

	double *A0 = new double[N * N];
	double *B0 = new double[N * N];
	//内存拷回
	cudaMemcpy(A0, Dev_A, N * N * sizeof(double),cudaMemcpyDeviceToHost);
	
	QR_cpu(N,Dev_A,Dev_B,Dev_C);

	
	printf("\nGPU 矩阵:\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			printf("%11lf  ", A0[i * N + j]);
		printf("\n");
	}

    while (check(A0, N)){
        QR(A0, N);
        printf("\rNo.%d", ++num);
    }
    printf("\rTotal loop(QR):%d \n", num);

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			printf("%11lf  ", A0[i * N + j]);
		printf("\n");
	}

	printf("\n========================CPU=======================\n");
	Householder(N,A);
	printf("\nCPU 矩阵:\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			printf("%11lf  ", A[i * N + j]);
		printf("\n");
	}

	num=0;
    while (check(A, N)){
        QR(A, N);
        printf("\rNo.%d", ++num);
    }
    printf("\rTotal loop(QR):%d \n", num);

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			printf("%11lf  ", A[i * N + j]);
		printf("\n");
	}








	



	//释放GPU内存
	cudaFree(Dev_A);
	cudaFree(Dev_B);
	cudaFree(Dev_C);


	
	//释放CPU内存
    // free(A);
    // free(B);
	free(C);

	return 0;
}


