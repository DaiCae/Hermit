#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_complex.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "time.h"

__global__ void transform_gpu(hipDoubleComplex *d_A, double *dev_A, int N, bool lower)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= N)
        return;
    if (idy >= N)
        return;

    //将 d_A转换成完整矩阵
    int n = N / 2;
    if (idx < n && idy < n)
    {
        if (lower)
        {
            if (idx < idy)
            {
                d_A[idx * n + idy].x = d_A[idy * n + idx].x;
                d_A[idx * n + idy].y = d_A[idy * n + idx].y * -1;
            }
        }
        else
        {
            if (idx > idy)
            {
                d_A[idx * n + idy].x = d_A[idy * n + idx].x;
                d_A[idx * n + idy].y = d_A[idy * n + idx].y * -1;
            }
        }
    }
    __syncthreads();

    //将复矩阵转换为实矩阵
    if (idx < n && idy < n)
    {
        dev_A[idx * N + idy] = d_A[idx * n + idy].x;
    }
    if (idx < n && idy >= n)
    {
        dev_A[idx * N + idy] = d_A[idx * n + (idy - n)].y;
    }
    if (idx >= n && idy < n)
    {
        dev_A[idx * N + idy] = d_A[(idx - n) * n + idy].y * -1;
    }
    if (idx >= n && idy >= n)
    {
        dev_A[idx * N + idy] = d_A[(idx - n) * n + (idy - n)].x;
    }
}

void transform(hipDoubleComplex *d_A, double *dev_A, int N, bool lower)
{
    dim3 grid((N - 1) / 32 + 1, (N - 1) / 32 + 1);
    dim3 block(32, 32);
    hipLaunchKernelGGL(transform_gpu, grid, block, 0, 0, d_A, dev_A, N, lower);
}

// 对特征值和特征向量排序
void sort_vector(double A[], int N, double *q)
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
                for (int k = 0; k < N; k++)
                {
                    temp = q[j * N + k];
                    q[j * N + k] = q[(j + 1) * N + k];
                    q[(j + 1) * N + k] = temp;
                }
            }
        }
    }
}

int Householder(int n, double A[], double b[], double c[], double *Q)
{
    double mol, q, value, K;
    double *alpha = new double[n];
    // 初始化Q为I矩阵
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Q[i * n + j] = 0.0;
        }
        Q[i * n + i] = 1;
    }

    for (int i = n - 1; i > 1; i--)
    {
        q = 0.0;
        //初始化alpha向量为0
        alpha[i] = 0.0;
        for (int j = 0; j < i; j++)
        {
            alpha[j] = A[i * n + j];
            q += alpha[j] * alpha[j];
        }
        // 判断q的值是否小于精度
        if (q + 1.0 == 1.0)
        {
            continue;
        }
        mol = sqrt(q);
        if (alpha[i - 1] > 0.0)
            mol = -mol;
        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;

        // 求Q(i+1)矩阵（矩阵减矩阵）
        // for (int j = 0; j < n - 1; j++)
        // {
        //     value = 0.0;
        //     for (int k = 0; k < i; k++)
        //         value += Q[j * n + k] * alpha[k] / q;
        //     b[j] = value;
        // }
        // for (int j = 0; j < n - 1; j++)
        // {
        //     for (int k = 0; k < i; k++)
        //     {
        //         Q[j * n + k] = Q[j * n + k] - b[j] * alpha[k];
        //     }
        // }

        // 求Q(i+1)矩阵（矩阵减矩阵）转置
        for (int j = 0; j < n - 1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i; k++)
                value += Q[k * n + j] * alpha[k] / q;
            b[j] = value;
        }
        for (int j = 0; j < n - 1; j++)
        {
            for (int k = 0; k < i; k++)
            {
                Q[k * n + j] = Q[k * n + j] - b[j] * alpha[k];
            }
        }


        // 求A(i+1)矩阵
        for (int j = 0; j < i + 1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i + 1; k++)
                value += A[j * n + k] * alpha[k] / q;
            b[j] = value;
        }
        K = 0;
        for (int j = 0; j < i + 1; j++)
        {
            K += alpha[j] * b[j] / (2 * q);
        }
        for (int j = 0; j < i + 1; j++)
        {
            c[j] = b[j] - K * alpha[j];
        }
        for (int j = 0; j < i + 1; j++)
        {
            for (int k = 0; k < i + 1; k++)
            {
                A[j * n + k] = A[j * n + k] - alpha[j] * c[k] - c[j] * alpha[k];
            }
        }
    }

    // 抽出主、次对角线元素
    for (int i = 0; i < n - 1; i++)
    {
        b[i] = A[i * n + i];
        c[i] = A[i * n + i + 1];
    }
    b[n - 1] = A[n * n - 1];
    c[n - 1] = 0;
    delete[] alpha;
    return 1;
}

//QR（带特征向量）
int QR_vector(int N, double *b, double *c, double *q, double eps){

    // N 为矩阵阶数
    // b[] 为主对角线元素
    // c[] 为次对角线元素
    
    double h, p, r;
    double f = 0.0;
    c[N - 1] = 0.0;
    for (int k = 0; k <= N - 1; k++){
        do
        {
            double g = b[k];
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

                //=======================
                // #pragma omp simd
                // #pragma omp parallel for 
                for (int j = 0; j <= N - 1; j++)
                {
                    int u = (i + 1) * N + j;
                    int v = (i)*N + j;
                    h = q[u];
                    q[u] = s * q[v] + e * h;
                    q[v] = e * q[v] - s * h;
                }
                //=======================
            }
            c[k] = s * p;
            b[k] = e * p;
        } while (fabs(c[k]) > eps);
        
        b[k] = b[k] + f;
    }
    return (1);
}

__global__ void Householder_step_0(double *Q , int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    //关闭多余线程
    if (idx >= N) return;
    // if (idy >= N) return;

    //初始化Q矩阵为 I 矩阵
    for (int j = 0; j < N; j++)
        Q[j * N + idx] = 0.0;
    Q[idx * N + idx] = 1;

}

__global__ void Householder_step_1(double *A ,double *alpha, double *beta, int N, int i)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N) return;

    if (idx < i)
    {
        alpha[idx] = A[i * N + idx];
        beta[idx] = alpha[idx] * alpha[idx];
    }else{
        alpha[idx] = 0.0;
        beta[idx] = 0.0;
    }

}

__global__ void Householder_step_Q(double *Q ,double *alpha, double *b, double q, int N, int i)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >N-1) return;
    
    // // 求Q(i+1)矩阵（矩阵减矩阵）
    // double value = 0.0;
    // for (int k = 0; k < i; k++)
    //     value += Q[idx * N + k] * alpha[k] / q;
    // b[idx] = value;
        

    // for (int k = 0; k < i; k++)
    // {
    //     Q[idx * N + k] = Q[idx * N + k] - b[idx] * alpha[k];
    // }

    //Q矩阵转置的求法
    double value = 0.0;
    for (int k = 0; k < i; k++)
        value += Q[k * N + idx] * alpha[k] / q;
    b[idx] = value;
        

    for (int k = 0; k < i; k++)
    {
        Q[k * N + idx] = Q[k * N + idx] - b[idx] * alpha[k];
    }
    
}

__global__ void Householder_step_2(double *A, double *alpha, double *beta, double *b, double q, int N, int i)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //idx < i+1
    if(idx >= i + 1) return;

    //======================================
    // 求A(i+1)矩阵
    // double value = 0.0;
    // for (int k = 0; k < i + 1; k++)
    //     value += A[idx * N + k] * alpha[k] / q;
    // b[idx] = value;

    double value = 0.0;
    for (int k = 0; k < i + 1; k++)
        value += A[k * N + idx] * alpha[k] / q;
    b[idx] = value;

    // 求K
    beta[idx] = alpha[idx] * value / (2 * q);
}

__global__ void Householder_step_3(double *alpha, double *b, double *c, double K, int N, int i)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    //idx < i+1
    if(idx >= i + 1) return;

    //======================================
    // 求A(i+1)矩阵
    c[idx] = b[idx] - K * alpha[idx];
    __syncthreads();
}

__global__ void Householder_step_4(double *A ,double *alpha, double *c, int N, int i)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    //idx < i+1
    if(idx >= i + 1) return;

    for (int k = 0; k < i + 1; k++)
    {
        A[k * N + idx] = A[k * N + idx] - alpha[idx] * c[k] - c[idx] * alpha[k];
    }
}

__global__ void Householder_step_5(double *A ,double *b, double *c, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N-1) return;

    // 抽出主、次对角线元素
    b[idx] = A[idx * N + idx];
    c[idx] = A[idx * N + idx + 1];

    if(idx==0)
    {
        b[N-1]=A[N * N - 1];
        c[N-1]=0.0;
    }
}

int mysolver_vector(int N, double *Dev_A, double *Dev_W, hipDoubleComplex *d_A)
{

    // int omp_set_num_threads(8);
    // int T_num = omp_get_num_threads();
    // printf("Threads num: %d\n",T_num);

    time_t start, end;
    start = clock();
    //主次对角线 b c 向量指针
    double *Dev_b;
    double *Dev_c;

    //临时向量
    double *Dev_alpha;
    double *Dev_beta;

    //乘积矩阵 在设备端指针
    double *Dev_Q;

    //分配空间 主次对角线
    hipMalloc((void **)&Dev_b, N * sizeof(double));
    hipMalloc((void **)&Dev_c, N * sizeof(double));

    hipMalloc((void **)&Dev_alpha, N * sizeof(double));
    hipMalloc((void **)&Dev_beta, N * sizeof(double));

    //分配空间 乘积矩阵
    hipMalloc((void **)&Dev_Q, N * N * sizeof(double));

    //定义N维向量的grid
    dim3 block(64);
    dim3 grid((N-1)/block.x + 1,1);
    
    printf("block size: %d | grid size: %d\n",block.x,grid.x);
    
    //计算需要的最小块数的长度取整
    int length = (N-1)/32 + 1;

    // //定义N维矩阵的grid
    // dim3 Block(16,16);
    // dim3 Grid(length, length);
    // printf("Block size: (%d,%d) | Grid size: (%d,%d)\n",Block.x,Block.y,Grid.x,Grid.y);

    //在CPU上申请求和用的缓存数组
    double *H_Sum = new double[N];
    double sum,eps=1e-12;
    
    time_t start0, end0;
    double t0=0.0;
    double t1=0.0;
    double t2=0.0;
    double t3=0.0;
    double t4=0.0;

    //Householder 循环分步调用核函数
    // hipLaunchKernelGGL(Householder_step_0,grid,block,0,0, Dev_Q, N);
    Householder_step_0<<<grid,block>>>(Dev_Q, N);//调用核函数
    for(int i = N-1; i>1 ;i--)
    {
        //=========================================
        //  初始化alpha向量 合并规约求alpha 的 mol
        //=========================================
        // hipLaunchKernelGGL(Householder_step_1,grid,block,0,0, Dev_A, Dev_alpha, Dev_beta, N, i);
        start0=clock();
        Householder_step_1<<<grid,block>>>(Dev_A, Dev_alpha, Dev_beta, N, i);//调用核函数
        hipDeviceSynchronize();
        end0 =clock();
        t0 += (double)(end0 - start0) / CLOCKS_PER_SEC * 1000;

        start0=clock();
        hipMemcpy(H_Sum, Dev_beta, N * sizeof(double), hipMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 q
        }
        //检查是否为0
        if (sum + 1.0 == 1.0) 
        {
            continue;
            printf("continue %d\n",i);
        }

        double mol =sqrtl(sum);
        
        //将alpha[i-1]拷贝到cpu端
        hipMemcpy(H_Sum, Dev_alpha + i - 1, 1 * sizeof(double), hipMemcpyDeviceToHost);
        if(H_Sum[0] >0.0)
            mol = -mol;
        sum -= H_Sum[0] * mol;
        H_Sum[0] = H_Sum[0] - mol;
        //写回alpha[i-1]
        hipMemcpy(Dev_alpha + i - 1, H_Sum, 1 * sizeof(double), hipMemcpyHostToDevice);
        end0 =clock();
        t1 +=(double)(end0 - start0) / CLOCKS_PER_SEC * 1000;


        //=========================================
        // 求Q(i+1)矩阵（矩阵减矩阵）
        // 求出K
        //=========================================
        // hipLaunchKernelGGL(Householder_step_2,grid,block,0,0, Dev_Q, Dev_A, Dev_alpha, Dev_beta, Dev_b, sum, N, i);
        // Householder_step_2_0<<<grid,block>>>(Dev_Q ,Dev_alpha, Dev_b, sum, N, i);//调用核函数
        // Householder_step_2_1<<<grid,block>>>(Dev_Q ,Dev_alpha, Dev_b, sum, N, i);//调用核函数
        start0=clock();
        Householder_step_Q<<<grid,block>>>(Dev_Q ,Dev_alpha, Dev_b, sum, N, i);//调用核函数
        hipDeviceSynchronize();
        end0 =clock();
        t2 +=(double)(end0 - start0) / CLOCKS_PER_SEC * 1000;
        
        start0=clock();
        Householder_step_2<<<grid,block>>>(Dev_A, Dev_alpha, Dev_beta, Dev_b, sum, N, i);//调用核函数
        hipDeviceSynchronize();
        hipMemcpy(H_Sum, Dev_beta, N * sizeof(double), hipMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++)
            sum += H_Sum[k];    //sum 为 K
        end0 =clock();
        t3 +=(double)(end0 - start0) / CLOCKS_PER_SEC * 1000;

        //=========================================
        // 求A(i+1)矩阵
        //=========================================
        // hipLaunchKernelGGL(Householder_step_3,grid,block,0,0, Dev_A, Dev_alpha, Dev_b, Dev_c, sum, N, i);
        start0=clock();
        Householder_step_3<<<grid,block>>>(Dev_alpha, Dev_b, Dev_c, sum, N, i);//调用核函数
        Householder_step_4<<<grid,block>>>(Dev_A, Dev_alpha, Dev_c, N, i);//调用核函数
        hipDeviceSynchronize();
        end0 =clock();
        t4 +=(double)(end0 - start0) / CLOCKS_PER_SEC * 1000;

    }
    printf("t0 = %lf,\nt1 = %lf,\nt2 = %lf,\nt3 = %lf,\nt4 = %lf\n\n", t0,t1,t2,t3,t4);


    //=========================================
    // 抽出主、次对角线元素
    //=========================================
    // hipLaunchKernelGGL(Householder_step_4,grid,block,0,0, Dev_A, Dev_b, Dev_c, N);
    Householder_step_5<<<grid,block>>>(Dev_A, Dev_b, Dev_c, N);//调用核函数
    hipDeviceSynchronize();

    hipFree(Dev_alpha);
    hipFree(Dev_beta);

    double *H_c = new double[N];
    double *H_Q = new double[N * N];

    hipMemcpy(H_Sum, Dev_b, N * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(H_c, Dev_c, N * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(H_Q, Dev_Q, N * N * sizeof(double), hipMemcpyDeviceToHost);
    
    end = clock();
    printf("HS GPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // for (int i = 0; i < N * N; i++)
    // {    
    //     Q[i] = H_Q[i];
    // }

    start = clock();
    QR_vector(N, H_Sum, H_c, H_Q, eps);
    end = clock();
    printf("QR CPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // start = clock();
    // Matrix_transpose<<<Grid,Block>>>(Dev_Q,N);
    // hipDeviceSynchronize();
    // end = clock();
    // printf("QR Trans time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    start = clock();
    sort_vector(H_Sum, N, H_Q);
    end = clock();
    printf("Sort time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    double *H_A = new double[N * N];
    hipDoubleComplex *A_yl = (hipDoubleComplex *)malloc(N / 2 * N / 2 * sizeof(hipDoubleComplex));
    
<<<<<<< HEAD
=======
//     // 转置特征矩阵
//     for (int i = 0; i < N; i++)
//     {
//         for (int j = 0; j < N; j++)
//         {
//             H_A[i * N + j] = H_Q[j * N + i];
//         }
//     }
//     // 抽一半特征向量
//     for (int i = 0; i < N / 2; i++)
//     {
//         for (int j = 0; j < N; j++)
//         {
//             H_Q[j * N + i] = H_A[j * N + i * 2];
//         }
//     }
//     // 将实特征向量转换乘复特征向量
//     for (int i = 0; i < N / 2; i++)
//     {
//         for (int j = 0; j < N / 2; j++)
//         {
//             A_yl[i * N / 2 + j].y = H_Q[i * N + j];
//             A_yl[i * N / 2 + j].x = H_Q[(N * N / 2) + i * N + j];
//         }
//     }

>>>>>>> 8a264712984ac4c0e4d23b393656d35b3b0469a3
    // 将实特征向量(按行) 转换成复特征向量(按列)
    for (int i = 0; i < N / 2; i++)
    {
        for (int j = 0; j < N / 2; j++)
        {
            A_yl[j * N / 2 + i].y = H_Q[i * 2 * N + j];
            A_yl[j * N / 2 + i].x = H_Q[i * 2 * N + N/2 + j];
        }
    }

    for (int i = 0; i < N / 2; i++)
        H_c[i] = H_Sum[i * 2];

    //结果拷回
    hipMemcpy(Dev_W, H_c, N / 2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_A, A_yl, N * N / 4 * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

    hipFree(Dev_b);
    hipFree(Dev_c);
    hipFree(Dev_Q);
    return 1;
}

