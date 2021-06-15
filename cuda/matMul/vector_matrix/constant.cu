#include <stdio.h>
#include <stdlib.h>
#include <math.h>


//__constant__ double const_b[16384];
__constant__ double const_b[8192];

__global__ void MatMul(double *A,double *C,const int N,int i)
{
	int x=threadIdx.x+blockDim.x*blockIdx.x;

	double elem1,elem2,value=0;
	for(int k=i*N/2;k<(i+1)*N/2;k++){

		elem1=A[x*N+k];
		elem2=const_b[k%(N/2)];

		value += elem1 * elem2;	//求和
	}
	C[x]+=value;
}

int main()
{
	const int N = pow(2,14);
	printf("分配CPU内存空间..\n");

    //矩阵A的内存空间分配
	double *A =(double *)malloc(N * N * sizeof(double));

    // double **A=(double**)malloc(sizeof(double*) * N);		//分配二维数组
    // A[0]=(double*)malloc(sizeof(double) * N * N);		//分配一维数组
    // for(int i=1;i<N;i++) A[i]=A[i-1]+N;

    //向量b的内存空间分配
    // double *b=(double*)malloc(sizeof(double) * N);
    double b[N];

    //矩阵C的内存空间分配
	double *C=(double*)malloc(sizeof(double) * N);

	printf("初始化矩阵..\n");
    //初始化
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
        {
            A[i*N+j]=i-0.1*j+1;
        }
        b[i]=log(sqrt(i*i-i+2));
    }
	
	printf("分配GPU内存空间..\n");
	//定义GPU内存指针
	double *Dev_A,*Dev_C;
	//设备端内存分配
	cudaMalloc((void**)&Dev_A, N * N * sizeof(double));
	cudaMalloc((void**)&Dev_C, N * sizeof(double));


	//使用event计算时间
	cudaEvent_t start,stop;
	float elapsedTime = 0;
	//创建Event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	printf("内存拷贝..\n");
	//数据拷贝，主机到设备
	cudaMemcpy(Dev_A, A, N * N * sizeof(double),cudaMemcpyHostToDevice);
	// cudaMemcpy(Dev_b, b, N * 1 * sizeof(double),cudaMemcpyHostToDevice);
	
	printf("开始计算..\n");

	dim3 block(16);
	dim3 grid(N/block.x);

	for(int i=0;i<2;i++){
		cudaMemcpyToSymbol(const_b, &b[i*N/2], sizeof(double) * N/2);
		MatMul<<<grid,block>>>(Dev_A,Dev_C,N,i);//调用核函数
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
			test_c[i]+=A[i*N+k]*b[k];
         }
    }

	bool flag = true;
    for (int i = 0; i < N; ++i){
        float a=test_c[i];
        float b=C[i];
        if (abs(a-b)>0.1)
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
	// cudaFree(Dev_b);
	cudaFree(Dev_C);
	
	//释放CPU内存
    free(A);
    //free(b);
	free(C);

	return 0;
}