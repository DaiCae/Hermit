#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_complex.h>
#include <math.h>
#include <sys/time.h>

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
                    temp = q[k * N + j];
                    q[k * N + j] = q[k * N + j + 1];
                    q[k * N + j + 1] = temp;
                }
            }
        }
    }
}

// Householder（带特征向量）
int Householder_vector(int n, double A[], double b[], double c[], double *Q)
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
        for (int j = 0; j <= n-1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i; k++)
                value += Q[j * n + k] * alpha[k] / q;
            b[j] = value;
        }
        for (int j = 0; j <= n-1; j++)
        {
            for (int k = 0; k < i; k++)
            {
                Q[j * n + k] = Q[j * n + k] - b[j] * alpha[k];
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
int QR_vector(int N, double *b, double *c, double *q, double eps, int limit){
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
                    //=======================
                    for (int j = 0; j <= N - 1; j++)
                    {
                        int u = j * N + i + 1;
                        int v = u - 1;
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
    //=======================
    // for (int i = 0; i <= N - 1; i++)
    // {
    //     int k = i;
    //     p = b[i];
    //     if (i + 1 <= N - 1)
    //     {
    //         int j = i + 1;
    //         while ((j <= N - 1) && (b[j] <= p))
    //         {
    //             k = j;
    //             p = b[j];
    //             j++;
    //         }
    //     }
    //     if (k != i)
    //     {
    //         b[k] = b[i];
    //         b[i] = p;
    //         for (int j = 0; j <= N - 1; j++)
    //         {
    //             int u = j * N + i;
    //             int v = j * N + k;
    //             p = q[j * N + i];
    //             q[j * N + i] = q[j * N + k];
    //             q[j * N + k] = p;
    //         }
    //     }
    // }
    //=======================
    return (1);
}

//求解特征值和特征向量
int mysolver_cpu_vector(int N, double *Dev_A, double *Dev_W, hipDoubleComplex *d_A)
{
    int l = 1000;
    //原矩阵
    double *a = new double[N * N];
    double *Qalpha = new double[N * N];
    //主次对角线
    double *b = new double[N];
    double *c = new double[N];
    //
    double *y_l = new double[N * N];
    double eps = 1e-12;
    double Eps = 1e-8;
    hipDoubleComplex *A_yl = (hipDoubleComplex *)malloc(N / 2 * N / 2 * sizeof(hipDoubleComplex));
    hipMemcpy(a, Dev_A, N * N * sizeof(double), hipMemcpyDeviceToHost);

    Householder_vector(N, a, b, c, Qalpha);
    QR_vector(N, b, c, Qalpha, eps, l);

    sort_vector(b, N, Qalpha);


    //抽出一半特征值
    for (int i = 0; i < N / 2; i++)
        c[i] = b[i * 2];

    //抽出一半特征向量
    // for (int i = 0; i < N / 2; i++)
    // {
    //     if (fabs(a[N * (N - 1) + i * 2]) >= Eps)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             y_l[j * N + i] = Qalpha[j * N + i * 2];
    //         }
    //     }
    //     else
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             y_l[j * N + i] = Qalpha[j * N + i * 2 + 1];
    //         }
    //     }
    // }
    for (int i = 0; i < N / 2; i++)
    {    for(int j=0; j<N;j++){
            y_l[j * N + i] = Qalpha[j * N + i * 2];
        }
    }


    //实部虚部合并为复数
    for (int i = 0; i < N / 2; i++)
    {
        for (int j = 0; j < N / 2; j++)
        {
            A_yl[i * N / 2 + j].y = y_l[i * N + j];
            A_yl[i * N / 2 + j].x = y_l[(N * N / 2) + i * N + j];
        }
    }


    //结果拷回
    hipMemcpy(Dev_W, c, N / 2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_A, A_yl, N * N / 2 * sizeof(double), hipMemcpyHostToDevice);

    free(a);
    free(b);
    free(c);
    free(y_l);
    free(A_yl);
    return 1;
}



__global__ void Householder_step_0(double *Q , int N){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N) return;

    //初始化Q矩阵为 I 矩阵
    Q[idx * N + idx] = 1;

}

__global__ void Householder_step_1(double *A ,double *alpha, double *beta, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N) return;

    if (idx < i)
    {
        alpha[idx] = A[i * N + idx];
        beta[idx] = alpha[idx] * alpha[idx];
    }else{
        alpha[idx] = 0.0;
    }

}

__global__ void Householder_step_2(double *Q ,double *A, double *alpha, double *beta, double *b, double q, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N) return;
    //======================================
    // 求Q(i+1)矩阵（矩阵减矩阵）
    double value =0.0;
    for (int k = 0; k < i; k++)
        value += Q[idx * N + k] * alpha[k] /q;
    b[idx]=value;
    __syncthreads();

    for (int k = 0; k < i; k++)
        Q[idx * N + k] = Q[idx * N + k] - b[idx] * alpha[k];
    __syncthreads();

    //idx < i+1
    if(idx >= i + 1) return;

    //======================================
    // 求A(i+1)矩阵
    value = 0.0;
    for (int k = 0; k < i + 1; k++)
        value += A[idx * N + k] * alpha[k] / q;
    b[idx] = value;

    // 求K
    beta[idx] = alpha[idx] * b[idx] / (2 * q);
}

__global__ void Householder_step_3(double *A ,double *alpha, double *b, double *c, double K, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    //idx < i+1
    if(idx >= i + 1) return;

    //======================================
    // 求A(i+1)矩阵
    c[idx] = b[idx] - K * alpha[idx];
    __syncthreads();
    for (int k = 0; k < i + 1; k++)
    {
        A[idx * N + k] = A[idx * N + k] - alpha[idx] * c[k] - c[idx] * alpha[k];
    }
}

__global__ void Householder_step_4(double *A ,double *b, double *c, int N){
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
    dim3 block(1024);
    dim3 grid((N-1)/block.x + 1);

    // //计算需要的最小块数的长度取整
    // int length = (N-1)/32 + 1;

    // //定义N维矩阵的grid
    // dim3 Block(32,32);
    // dim3 Grid(length, length);

    //在CPU上申请求和用的缓存数组
    double *H_Sum = new double[N];
    double sum,eps=1e-12;

    //Householder 循环分步调用核函数
    hipLaunchKernelGGL(Householder_step_0,grid,block,0,0, Dev_Q, N);
    for(int i = N-1; i>1 ;i--)
    {
        //=========================================
        //  初始化alpha向量 合并规约求alpha 的 mol
        //=========================================
        hipLaunchKernelGGL(Householder_step_1,grid,block,0,0, Dev_A, Dev_alpha, Dev_beta, N, i);
        hipDeviceSynchronize();
        hipMemcpy(H_Sum, Dev_beta, N * sizeof(double), hipMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 q
        }
        //检查是否为0
        if( (sum-eps)<0.0) continue;
        double mol =sqrtl(sum);
        
        //将alpha[i-1]拷贝到cpu端
        hipMemcpy(H_Sum, Dev_alpha + i - 1, 1 * sizeof(double), hipMemcpyDeviceToHost);
        if(H_Sum[0] >0.0)
            mol = -mol;
        sum -= H_Sum[0] * mol;
        H_Sum[0] = H_Sum[0] - mol;
        //写回alpha[i-1]
        hipMemcpy(Dev_alpha + i - 1, H_Sum, 1 * sizeof(double), hipMemcpyHostToDevice);
        
        //=========================================
        // 求Q(i+1)矩阵（矩阵减矩阵）
        // 求出K
        //=========================================
        hipLaunchKernelGGL(Householder_step_2,grid,block,0,0, Dev_Q, Dev_A, Dev_alpha, Dev_beta, Dev_b, sum, N, i);
        hipDeviceSynchronize();
        hipMemcpy(H_Sum, Dev_beta, N * sizeof(double), hipMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++)
            sum += H_Sum[k];    //sum 为 K

        //=========================================
        // 求A(i+1)矩阵
        //=========================================
        hipLaunchKernelGGL(Householder_step_3,grid,block,0,0, Dev_A, Dev_alpha, Dev_b, Dev_c, sum, N, i);
        hipDeviceSynchronize();
    }

    //=========================================
    // 抽出主、次对角线元素
    //=========================================
    hipLaunchKernelGGL(Householder_step_4,grid,block,0,0, Dev_A, Dev_b, Dev_c, N);
    hipDeviceSynchronize();

    hipFree(Dev_alpha);
    hipFree(Dev_beta);

    double *H_c = new double[N];
    double *H_Q = new double[N * N];

    hipMemcpy(H_Sum, Dev_b, N * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(H_c, Dev_c, N * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(H_Q, Dev_Q, N * N * sizeof(double), hipMemcpyDeviceToHost);

    QR_vector(N, H_Sum, H_c, H_Q, eps, 0);
    sort_vector(H_Sum, N, H_Q);

    //抽出一半特征值
    for (int i = 0; i < N / 2; i++)
        H_c[i] = H_Sum[i * 2];

    double *y_l = new double[N * N];

    //抽出一半特征向量
    for (int i = 0; i < N / 2; i++)
    {    for(int j=0; j<N;j++){
            y_l[j * N + i] = H_Q[j * N + i * 2];
        }
    }

    hipDoubleComplex *A_yl = (hipDoubleComplex *)malloc(N / 2 * N / 2 * sizeof(hipDoubleComplex));

    //实部虚部合并为复数
    for (int i = 0; i < N / 2; i++)
    {
        for (int j = 0; j < N / 2; j++)
        {
            A_yl[i * N / 2 + j].y = y_l[i * N + j];
            A_yl[i * N / 2 + j].x = y_l[(N * N / 2) + i * N + j];
        }
    }


    //结果拷回
    hipMemcpy(Dev_W, H_c, N / 2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_A, A_yl, N * N / 2 * sizeof(double), hipMemcpyHostToDevice);

    hipFree(Dev_b);
    hipFree(Dev_c);
    printf("OK!");
    return 1;
}










