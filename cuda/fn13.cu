#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "time.h"

// #include <cucomplex.h>

double eps = 1e-8;


// typedef enum solverEigMode_ {
// 	solverEigMode_vector,
// 	solverEigMode_novector
// }solverEigMode;

// typedef enum solverFillMode_ {
// 	solverFillMode_lower,
// 	solverFillMode_upper
// }solverFillMode;

// Hermite矩阵特征值求解函数调用声明
// void solverDnZheevd(solverEigMode jobz,
//                     solverFillMode uplo,
//                     int n,
//                     cuDoubleComplex *d_A,
//                     int lda,
//                     double *d_W, 
//                     int *devInfo);


////////////////////////////////////////////////////////////////
// 串行代码开始
////////////////////////////////////////////////////////////////
void symmat(double H[], int N)
{
    int n = N / 2;
    double *A = new double[n * n];
    double *B = new double[n * n];

    for (int i = 0; i < n; i++)
    {
        srand((unsigned)time(NULL));
        for (int j = 0; j < n; j++)
        {
            if (j >= i)
            {
                A[i * n + j] = rand() % 20 - 10;
                B[i * n + j] = rand() % 20 - 10;
            }
            else
            {
                A[i * n + j] = A[j * n + i];
                B[i * n + j] = -1 * B[j * n + i];
            }
        }
        B[i * n + i] = 0;
    }

    // show(A,n);
    // printf("\n");
    // show(B,n);
    // printf("\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if ((i >= n && j >= n) || (i < n && j < n))
                H[i * N + j] = A[i % n * n + j % n];
            else if (i < n)
                H[i * N + j] = -1 * B[i % n * n + j % n];
            else
                H[i * N + j] = B[i % n * n + j % n];
        }
    }
}

void show(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%15.10lf,", A[i * N + j]);
        }
        printf("\n");
    }
}

void sort(double A[],int N)
{
    double temp = 0.0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N - i - 1; j++)
        {
            if (A[j] > A[j + 1])
            {
                temp = A[j];
                A[j] = A[j + 1];
                A[j + 1] = temp;
            }
        }
    }
}

//Householder变换
int Householder_cpu(int n, double A[], double b[], double c[])
{
    double mol, q, value;
    double *V = new double[n];
    double *S = new double[n];
    double *Q = new double[n];

    for (int i = n - 1; i > 1; i--)
    {
        q = 0.0;
        //初始化V向量为0
        V[i] = 0.0;

        for (int j = 0; j < i; j++)
        {
            V[j] = A[i * n + j];
            q += V[j] * V[j];
            // printf("%10.8lf, ",V[j] * V[j]);
        }
        if( (q-eps)<0.0) continue;
        mol = sqrt(q);

        // printf("\n\n");
        // printf("%lf\n",q);
        // printf("%lf\n",mol);
        //合并规约
    //=======================================================
        if (V[i - 1] > 0.0)
            mol = -mol;
        q -= V[i - 1] * mol;
        V[i - 1] = V[i - 1] - mol;

        // printf("q: %20.18lf [%d]\n",q,i);
        // for(int k=0;k<n;k++)
        //     printf("V[%d]:%10lf \n",k,V[k]);
        // printf("\n");

        // 求S向量
        for (int j = 0; j < i + 1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i + 1; k++)
            {
                value += A[j * n + k] * V[k] / q;
            }
            S[j] = value;
        }

        // for(int k=0;k<n;k++)
        //     printf("S[%d]:%10lf \n",k,S[k]);
        // printf("\n");


        // 求K的值
        double K = 0;
        for (int j = 0; j < i + 1; j++)
        {
            K +=V[j] * S[j] / (2 * q);
        }
    //=======================================================
        // 求Q向量
        for (int j = 0; j < i + 1; j++)
        {
            Q[j] = S[j] - K * V[j];
        }

        // printf("K: %20.18lf %d\n",K,i);
        // for(int k=0;k<n;k++)
        //     printf("Q[%d]:%10lf \n",k,Q[k]);
        // printf("\n\n");

    //=======================================================
        // 求A(i+1)矩阵
        for (int j = 0; j < i + 1; j++)
        {
            for (int k = 0; k < i + 1; k++)
            {
                A[j * n + k] = A[j * n + k] - V[j] * Q[k] - Q[j] * V[k];
                // Q[j*n+k] = Q[j*n+k]-B[j*n+k]-C[j*n+k];
            }
        }
        
        // show(A,n);
        // printf("\n");
        // break;
    }
    //=======================================================
    for(int i=0;i<n-1;i++){
        b[i] = A[i*n+i];
        c[i] = A[i*n+i+1];
    }
    b[n-1]=A[n*n-1];
    c[n-1]=0;

    // for(int k=0;k<n;k++)
    //     printf("| %20.14lf  |  %20.14lf |\n",b[k],c[k]);

    delete[] V;
    delete[] S;
    delete[] Q;
    return 1;
}

//QR
int QR_cpu(int N, double *b, double *c, double eps)
{
    // N 为矩阵阶数
    // b[] 为主对角线元素
    // c[] 为次对角线元素
    
    double h, p, r;
    double f = 0.0;
    for (int k = 0; k <= N - 1; k++){
        do{
            double g = b[k];
            // (b[k+1] - b[k]) / (2 * c[k])
            p = (b[k + 1] - g) / (2.0 * c[k]);
            r = sqrt(p * p + 1.0);
            if (p >= 0.0)
                b[k] = c[k] / (p + r);
            else
                b[k] = c[k] / (p - r);
            h = g - b[k];

            for (int i = k + 1; i <= N - 1; i++)
                b[i] = b[i] - h;
            f = f + h;

            p = b[N-1];

            double e = 1.0;
            double s = 0.0;
            for (int i = N - 2; i >= k; i--)
            {
                g = e * c[i];
                h = e * p;
                if (fabs(p) >= fabs(c[i]))
                {
                    e = c[i] / p;
                    r = sqrt(e * e + 1.0);
                    c[i + 1] = s * p * r;
                    s = e / r;
                    e = 1.0 / r;
                }
                else
                {
                    e = p / c[i];
                    r = sqrt(e * e + 1.0);
                    c[i + 1] = s * c[i] * r;
                    s = 1.0 / r;
                    e = e / r;
                }
                p = e * b[i] - s * g;
                b[i + 1] = h + s * (e * g + s * b[i]);
            }
            c[k] = s * p;
            b[k] = e * p;
        } while (fabs(c[k]) > eps);
        
        b[k] = b[k] + f;
    }
    return (1);
}

////////////////////////////////////////////////////////////////
// 串行代码结束
////////////////////////////////////////////////////////////////

__global__ void Householder_gpu(double *A, double *V, double *S ,double *Q,int N, int i)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= N) return;

    //初始化向量
    if(idx<i) 
        V[idx] = A[i * N + idx];
    else 
        V[idx] = 0.0;

    //将V向量元素求平方保存到S向量
    S[idx] = V[idx] * V[idx];
    //对V向量中的元素求和

    if(idx == 0 && blockIdx.x == 0 ){
        double q=0.0;
        //对V向量中的元素求和后求mol
        for(int j=0; j<i; j++)
            q += S[j];
        double mol=sqrt(q);
        //判断正负更新mol的符号
        if (V[i-1] > 0.0) mol = -mol;
        q -=  V[i-1] * mol;
        V[i-1] -= mol;
        S[0]=q;
    }
    //在所有线程中同步q的值
    double q=S[0];
    __syncthreads();

    //=============================================
        __syncthreads();
        if(idx==32){
            printf("q: %20.18lf %d\n",q,i);
            printf("V: ");
            for(int k=0;k<N;k++)
                printf("%10lf ",V[k]);
            printf("\n");
        }
        __syncthreads();
    //=============================================

    // 求S向量
    double sum=0.0;
    for (int k = 0; k < i + 1; k++){
        sum += A[idx * N + k] * V[k] / q;
    }
        S[idx] = sum;
    if(idx>i)
    {
        S[idx] = 1;
    }

    //=============================================
        __syncthreads();
        if(idx==0){
            for(int k=0;k<N;k++)
                printf("S[%d]:%10lf\n",k,S[k]);
            printf("\n");
        }
        __syncthreads();
    //=============================================

    //计算K的值
    Q[idx] = V[idx] * S[idx] / (2 * q);
    __syncthreads();
    if(idx == 0 && blockIdx.x == 0 ){
        double K = 0;
        for (int j = 0; j < i + 1; j++){
            K +=  Q[j];
            printf("%20.15lf \n",Q[j]);
        }
        Q[0]=K;
        printf("K: %20.18lf %d\n",K,i);
    }
    double K =Q[0];
    __syncthreads();
    // 求Q向量
    Q[idx] = S[idx] - K * V[idx];



    //=============================================
        __syncthreads();
        if(idx==0){
            printf("Q: ");
            for(int k=0;k<N;k++)
                printf("%10lf ",Q[k]);
            printf("\n\n");
        }
        __syncthreads();
    //=============================================



    for (int k = 0; k < i + 1; k++)
    {
        A[idx * N + k] = A[idx * N + k] - V[idx] * Q[k] - Q[idx] * V[k];
    }

    if(i==2){
        S[idx]=A[idx*N+idx];
        Q[idx]=A[idx*N+idx+1];
    
        if(idx==N-1) Q[idx]=0.0;
        // __syncthreads();
        // if(idx==0){
        //     for(int k=0;k<N;k++)
        //         printf("%10lf  ",Q[k]);
        //     printf("\n\n");
        // }
    }
}

__global__ void Householder_step_0(double *A, double *V, double *Q , int N, int i){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	// unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= N) return;

    //初始化向量
    if(idx<i) 
        V[idx] = A[i * N + idx];
    else 
        V[idx] = 0.0;

    //将V向量元素求平方保存到Q向量
    Q[idx] = V[idx] * V[idx];

    // if(idx==0)
    //     for(int k=0;k<N;k++)
    //         printf("%10.8lf, ",Q[k]);
    
    // //对S向量中的元素求和
    // unsigned int tid = threadIdx.x;
    // //计算起始指针位置
    // double *point_s = Q + blockIdx.x * blockDim.x;
    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    // {
    //     if (tid < stride)
    //     {
    //         point_s[tid] += point_s[tid + stride];
    //     }
    //     __syncthreads();
    // }
    // //写回结果到Global Memory
    // if (tid == 0)
    //     Q[blockIdx.x] = point_s[0];
}

__global__ void Householder_step_1(double *A, double *V, double *Q ,double *b, double q, double mol, int N, int i){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    //关闭多余的线程
    if (idx >= N) return;

    if (V[i - 1] > 0.0)
        mol = -mol;
    q -= V[i - 1] * mol;

    if(idx==0)
        V[i - 1] = V[i - 1] - mol;

    __syncthreads();

    // 求S向量 并保存到Q中
    double value = 0.0;
    for (int k = 0; k < i + 1; k++)
    {
        value += A[idx * N + k] * V[k] / q;
    }
    Q[idx] = value;
    // if(idx >= i) Q[idx]=1;

    //=============== 输出V 和 S向量 ==================
        // __syncthreads();
        // if(idx==0){
        //     printf("q: %20.18lf [%d]\n",q,i);
        //     for(int k=0;k<N;k++)
        //     {
        //         printf("V[%d]:%10lf\n",k,V[k]);
        //     }
        //     printf("\n");
        //     for(int k=0;k<N;k++)
        //     {
        //         printf("S[%d]:%10lf\n",k,Q[k]);
        //     }
        //     printf("\n");
        // }
        // __syncthreads();
    //=============================================

    b[idx]= V[idx] * Q[idx] * 0.5 / q;

    // __syncthreads();
    // //对c向量中的元素求和
    // unsigned int tid = threadIdx.x;
    // //计算起始指针位置
    // double *point_s = b + blockIdx.x * blockDim.x;
    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    // {
    //     if (tid < stride)
    //     {
    //         point_s[tid] += point_s[tid + stride];
    //     }
    //     __syncthreads();
    // }
    // //写回结果到Global Memory
    // if (tid == 0)
    //     b[blockIdx.x] = point_s[0];

}

__global__ void Householder_step_2(double *V, double *Q , double K, int N, int i){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    //关闭多余的线程
    if (idx >= N) return;
    Q[idx] = Q[idx] - K * V[idx];

    //=============== 输出Q向量 ==================
        // __syncthreads();
        // if(idx==0){
        //     printf("K: %20.18lf %d\n",K,i);
        //     for(int k=0;k<N;k++)
        //     {
        //         printf("Q[%d]:%10lf\n",k,Q[k]);
        //     }
        //     printf("\n");
        // }
        // __syncthreads();
    //=============================================
}

__global__ void Householder_step_3(double *A, double *V, double *Q , double *b, double *c, int N, int i){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;

    //关闭多余的线程
    if (idx >= N) return;
    if (idy >= N) return;

    //更新A矩阵的值
    A[idx * N + idy] -=  V[idx] * Q[idy] + Q[idx] * V[idy];

    // if(idx==0 && idy ==0){
    //     for(int x=0; x<N ;x++)
    //     {
    //         for(int y=0; y<N ;y++)
    //         {
    //             printf("%15.10lf,",A[x*N+y]);
    //         }    
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

__global__ void Householder_step_4(double *A, double *b, double *c, int N){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    //关闭多余的线程
    if (idx >= N) return;


    b[idx] = A[idx * N + idx];
    if(idx != N-1)
        c[idx] = A[idx * N + idx + 1];
    
    
}


int solverDnZheevd(int N, double * D_A,double * Dev_W,double * H_W)
{	
    //使用event计算时间
	cudaEvent_t start,stop;
	float elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	//=====================================================================
	double *Dev_V, *Dev_Q;
    double *Dev_b, *Dev_c;
    
    // cudaMalloc((void**)&D_S, N * sizeof(double));
	// cudaMalloc((void**)&D_Q, N * sizeof(double));
    cudaMalloc((void**)&Dev_V, N * sizeof(double));
    cudaMalloc((void**)&Dev_Q, N * sizeof(double));
    cudaMalloc((void**)&Dev_b, N * sizeof(double));
	cudaMalloc((void**)&Dev_c, N * sizeof(double));

    //定义N维向量的grid
    dim3 block(1024);
    dim3 grid((N-1)/block.x + 1);

    //计算需要的最小块数的长度取整
    int length = (N-1)/32 + 1;

    //定义N维矩阵的grid
    dim3 Block(32,32);
    dim3 Grid(length, length);

    //在CPU上申请求和用的缓存数组
    double *H_Sum = new double[N];
    double sum;
    //循环分步调用核函数
    for (int i = N- 1; i > 1; i--)
    {
        // if(i==N/2) continue;
        //=========================================
        //  初始化V向量 合并规约求mol
        //=========================================
        Householder_step_0<<<grid,block>>>(D_A, Dev_V, Dev_Q, N, i);//调用核函数
        cudaDeviceSynchronize();
        // cudaMemcpy(H_Sum, Dev_Q, grid.x * sizeof(double), cudaMemcpyDeviceToHost);
        // double sum = 0;
        // for (int k = 0; k < grid.x; k++){
        //     // printf("%10.8lf,",H_Sum[k]);
        //     sum += H_Sum[k];    //sum 为 q
        // }
        cudaMemcpy(H_Sum, Dev_Q, N * sizeof(double), cudaMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 q
        }
        if( (sum-eps)<0.0) continue;

        double mol =sqrtl(sum);
        // printf("\n\n");
        // printf("%lf\n",sum);
        // printf("%lf\n",mol);


        //=========================================
        //  求出 V 和 S 向量 合并规约求出 K的值 （S向量保存在Q中）
        //=========================================
        Householder_step_1<<<grid,block>>>(D_A, Dev_V, Dev_Q, Dev_b, sum, mol ,N, i);//调用核函数
        cudaDeviceSynchronize();
        // cudaMemcpy(H_Sum, Dev_b, grid.x * sizeof(double), cudaMemcpyDeviceToHost);
        // sum = 0;
        // for (int k = 0; k < grid.x; k++){
        //     sum += H_Sum[k];    //sum 为 K
        // }
        cudaMemcpy(H_Sum, Dev_b, N * sizeof(double), cudaMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 K
        }

        //=========================================
        //  通过 V，S向量和 K 求出Q向量
        //=========================================
        Householder_step_2<<<grid,block>>>(Dev_V, Dev_Q, sum ,N, i);//调用核函数
        cudaDeviceSynchronize();

        //=========================================
        //  通过 V，Q向量 更新原矩阵A 若为最后一次循环则抽出主次对角线元素
        //=========================================
        Householder_step_3<<<Grid,Block>>>(D_A, Dev_V, Dev_Q, Dev_b, Dev_c, N, i);//调用核函数
        cudaDeviceSynchronize();
        // break;
    }
    Householder_step_4<<<grid,block>>>(D_A, Dev_b, Dev_c, N);//调用核函数

    double *c = new double[N];
	cudaMemcpy(H_W, Dev_b, N * sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(c, Dev_c, N * sizeof(double),cudaMemcpyDeviceToHost);

    // c[N-1]=0;

    // for(int k=0;k<N;k++)
    //     printf("| %20.14lf  |  %20.14lf |\n",H_W[k],c[k]);
    // printf("\n\n");

    QR_cpu(N, H_W, c, eps);
    sort(H_W, N);	

    // for(int k=0;k<N;k++)
    //     printf("%17.14lf\n",H_W[k]);
    // printf("\n\n");

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
	
	printf("\r运行时间:%f (ms)\n",elapsedTime);
	return 1;
}

int main()
{
	printf("分配CPU内存空间..\n");

    int N = 2560;
    double *A = new double[N * N];
    symmat(A, N);

    // double A[100] = {
    //     1.0, 2.0, 3.0, 4.0, 5.0, 0.0, -1.0, -1.0, -1.0, -1.0,
    //     2.0, 2.0, 3.0, 4.0, 6.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    //     3.0, 3.0, 3.0, 1.0, 5.0, 1.0, -1.0, 0.0, 0.0, 0.0,
    //     4.0, 4.0, 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    //     5.0, 6.0, 5.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    //     0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    //     -1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0, 6.0,
    //     -1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 5.0,
    //     -1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 1.0, 3.0, 1.0,
    //     -1.0, 0.0, 0.0, -1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 3.0
    // };

    double *b = new double[N];
    double *c = new double[N];
    time_t start, end;

	
	printf("分配GPU内存空间..\n");
	//定义GPU内存指针
	double *Dev_A,*Dev_W;
	//设备端内存分配
	cudaMalloc((void**)&Dev_A, N * N * sizeof(double));
	cudaMalloc((void**)&Dev_W, N * sizeof(double));


	printf("内存拷贝..\n\n");
	//数据拷贝，主机到设备
	cudaMemcpy(Dev_A, A, N * N * sizeof(double),cudaMemcpyHostToDevice);
    

    printf("========================================================\n");
    printf("GPU is calculating\n");	double *w = new double[N];	
	double *H_W= new double[N];	
    solverDnZheevd(N,Dev_A,Dev_W,H_W);
	// cudaMemcpy(H_W, Dev_W, N * N * sizeof(double),cudaMemcpyDeviceToHost);

    printf("========================================================\n");
    printf("CPU Householder is calculating\n");
    start = clock();
    Householder_cpu(N, A, b, c);
    end = clock();
    // for(int k=0;k<N;k++)
    //     printf("| %20.14lf  |  %20.14lf |\n",b[k],c[k]);
    // printf("\n\n");
    printf("CPU HS time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("========================================================\n");
    printf("CPU QR is calculating\n");
    start = clock();
    QR_cpu(N, b, c, eps);
    sort(b, N);
    end = clock();
    printf("CPU QR time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("========================================================\n");
    printf("Result checking\n");
    for(int k=0;k<N;k++){
        double diff=fabs(H_W[k]-b[k]);
        if (diff<=eps)
            printf("No[%2d]: %17.8lf || %17.8lf || %12.7lf \n", k, H_W[k], b[k],diff);
        else
            printf("No[%2d]: %17.8lf || %17.8lf || %12.7lf || E!\n", k, H_W[k], b[k],diff);
    }


	//释放GPU内存
	cudaFree(Dev_A);
	
	//释放CPU内存
    // free(A);
    // free(B);
	// free(C);

	return 0;
}


