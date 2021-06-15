#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_complex.h>
#include <math.h>
#include <sys/time.h>

double eps = 1e-15;

__global__ void transform_gpu(hipDoubleComplex *d_A, double *dev_A, int N, bool lower){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(idx >= N) return;
    if(idy >= N) return;
    

    //将 d_A转换成完整矩阵
    int n =N/2;
    if(idx < n && idy < n ){
        if(lower){
            if(idx < idy){
                d_A[idx*n+idy].x = d_A[idy*n+idx].x;
                d_A[idx*n+idy].y = d_A[idy*n+idx].y * -1;
            }
        }else{
            if(idx > idy){
                d_A[idx*n+idy].x = d_A[idy*n+idx].x;
                d_A[idx*n+idy].y = d_A[idy*n+idx].y * -1;
            }
        }
    }
    __syncthreads();

    //将复矩阵转换为实矩阵
    if(idx < n && idy < n ){
        dev_A[idx * N + idy] = d_A[idx * n + idy].x;
    }
    if(idx < n && idy >= n ){
        dev_A[idx * N + idy] = d_A[idx * n + (idy-n)].y ;
    }
    if(idx >= n && idy < n ){
        dev_A[idx * N + idy] = d_A[(idx-n) * n + idy].y * -1;
    }
    if(idx >= n && idy >= n ){
        dev_A[idx * N + idy] = d_A[ (idx-n) * n + (idy-n)].x;
    }
}

void transform(hipDoubleComplex *d_A, double *dev_A, int N, bool lower){
    dim3 grid( (N-1) / 32 + 1, (N-1) / 32 + 1);
    dim3 block(32,32);
    hipLaunchKernelGGL(transform_gpu,grid,block,0,0, d_A, dev_A, N, lower);
}

void sort(double A[],int N){
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
//QR
int QR_cpu(int N, double *b, double *c, double eps){
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

__global__ void Householder_step_0(double *A, double *V, double *Q , int N, int k){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	// int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= N) return;

    //初始化向量
    if(idx < k) 
        V[idx] = A[k * N + idx];
    else 
        V[idx] = 0.0;

    //将V向量元素求平方保存到Q向量
    Q[idx] = V[idx] * V[idx];

}

__global__ void Householder_step_1(double *A, double *V, double *Q ,double *b, double q, double mol, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

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
    b[idx]= V[idx] * Q[idx] * 0.5 / q;
}

__global__ void Householder_step_2(double *V, double *Q , double K, int N, int i){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    //关闭多余的线程
    if (idx >= N) return;
    Q[idx] = Q[idx] - K * V[idx];
}

__global__ void Householder_step_3(double *A, double *V, double *Q , double *b, double *c, int N, int i){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;

    //关闭多余的线程
    if (idx >= N) return;
    if (idy >= N) return;

    //更新A矩阵的值
    A[idx * N + idy] -=  V[idx] * Q[idy] + Q[idx] * V[idy];
}

__global__ void Householder_step_4(double *A, double *b, double *c, int N){
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    //关闭多余的线程
    if (idx >= N) return;


    b[idx] = A[idx * N + idx];
    if(idx != N-1)
        c[idx] = A[idx * N + idx + 1];
    
    
}


int mysolver_novector(int N, double * Dev_A,double * Dev_W)
{	
	double *Dev_V, *Dev_Q;
    double *Dev_b, *Dev_c;

    hipMalloc(&Dev_V, N * sizeof(double));
    hipMalloc(&Dev_Q, N * sizeof(double));
    hipMalloc(&Dev_b, N * sizeof(double));
	hipMalloc(&Dev_c, N * sizeof(double));

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
    for(int i = N- 1; i > 1; i--)
    {
        // if(i==N/2) continue;
        //=========================================
        //  初始化V向量 合并规约求mol
        //=========================================
        // Householder_step_0<<<grid,block>>>(Dev_A, Dev_V, Dev_Q, N, i);//调用核函数
        hipLaunchKernelGGL(Householder_step_0,grid,block,0,0, Dev_A, Dev_V, Dev_Q, N, i);
        hipDeviceSynchronize();
        hipMemcpy(H_Sum, Dev_Q, N * sizeof(double), hipMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 q
        }

        //检查是否为0
        if( (sum-eps)<0.0) continue;

        double mol =sqrtl(sum);

        //=========================================
        //  求出 V 和 S 向量 合并规约求出 K的值 （S向量保存在Q中）
        //=========================================
        // Householder_step_1<<<grid,block>>>(Dev_A, Dev_V, Dev_Q, Dev_b, sum, mol ,N, i);//调用核函数
        hipLaunchKernelGGL(Householder_step_1,grid,block,0,0, Dev_A, Dev_V, Dev_Q, Dev_b, sum, mol ,N, i);
        hipDeviceSynchronize();
        hipMemcpy(H_Sum, Dev_b, N * sizeof(double), hipMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 K
        }

        //=========================================
        //  通过 V，S向量和 K 求出Q向量
        //=========================================
        // Householder_step_2<<<grid,block>>>(Dev_V, Dev_Q, sum ,N, i);//调用核函数
        hipLaunchKernelGGL(Householder_step_2,grid,block,0,0, Dev_V, Dev_Q, sum ,N, i);
        hipDeviceSynchronize();

        //=========================================
        //  通过 V，Q向量 更新原矩阵A 若为最后一次循环则抽出主次对角线元素
        //=========================================
        // Householder_step_3<<<Grid,Block>>>(Dev_A, Dev_V, Dev_Q, Dev_b, Dev_c, N, i);//调用核函数
        hipLaunchKernelGGL(Householder_step_3,Grid,Block,0,0, Dev_A, Dev_V, Dev_Q, Dev_b, Dev_c, N, i);
        hipDeviceSynchronize();
        // break;
    }
    // Householder_step_4<<<grid,block>>>(Dev_A, Dev_b, Dev_c, N);//调用核函数
    hipLaunchKernelGGL(Householder_step_4,grid,block,0,0, Dev_A, Dev_b, Dev_c, N);

    double *b = new double[N];
    double *c = new double[N];
	hipMemcpy(b, Dev_b, N * sizeof(double),hipMemcpyDeviceToHost);
	hipMemcpy(c, Dev_c, N * sizeof(double),hipMemcpyDeviceToHost);

    QR_cpu(N, b, c, eps);
    sort(b, N);	
	
    //取出一半特征值
    for(int i=0;i<N/2;i++)
        c[i]=b[i*2];

    //结果拷回
    hipMemcpy(Dev_W, c, N/2 * sizeof(double),hipMemcpyHostToDevice);

    hipFree(Dev_V);
    hipFree(Dev_Q);
    hipFree(Dev_b);
    hipFree(Dev_c);

    free(H_Sum);
    free(b);
    free(c);

	return 1;
}

//Householder
void Householder_old(double *a,int n, double* q, double* b, double* c)
{
    int i, j, k, u, v;
    
    //将a拷贝到q
    for (i = 0; i < n * n ; i++)
        q[i] = a[i];

    for (i = n - 1; i >= 1; i--)
    {
        double h = 0.0;
        if (i > 1)
            for (k = 0; k <= i - 1; k++)
            {
                u = i * n + k;
                h = h + q[u] * q[u];
            }
        
        // 检查是否为0
        if (h + 1.0 == 1.0)
        {
            c[i] = 0.0;
            if (i == 1)
                c[i] = q[i * n + i - 1];
            b[i] = 0.0;
        }
        else
        {
            c[i] = sqrt(h);
            u = i * n + i - 1;
            if (q[u] > 0.0)
                c[i] = -c[i];
            h = h - q[u] * c[i];
            q[u] = q[u] - c[i];
            
            double f = 0.0;
            double g = 0.0;

            for (j = 0; j <= i - 1; j++)
            {
                q[j * n + i] = q[i * n + j] / h;
                g = 0.0;
                for (k = 0; k <= j; k++)
                    g = g + q[j * n + k] * q[i * n + k];
                if (j + 1 <= i - 1)
                    for (k = j + 1; k <= i - 1; k++)
                        g = g + q[k * n + j] * q[i * n + k];
                c[j] = g / h;
                f = f + g * q[j * n + i];
            }

            double h2 = f / (h + h);
            for (j = 0; j <= i - 1; j++)
            {
                f = q[i * n + j];
                g = c[j] - h2 * f;
                c[j] = g;
                for (k = 0; k <= j; k++)
                {
                    u = j * n + k;
                    q[u] = q[u] - f * c[k] - g * q[i * n + k];
                }
            }
            b[i] = h;
        }
    }
    for (i = 0; i <= n - 2; i++)
        c[i] = c[i + 1];
    c[n - 1] = 0.0;

    b[0] = 0.0;
    for (i = 0; i <= n - 1; i++)
    {
        if ((b[i] != 0.0) && (i - 1 >= 0))
            for (j = 0; j <= i - 1; j++)
            {
                double g = 0.0;
                for (k = 0; k <= i - 1; k++)
                    g = g + q[i * n + k] * q[k * n + j];
                for (k = 0; k <= i - 1; k++)
                {
                    u = k * n + j;
                    q[u] = q[u] - g * q[k * n + i];
                }
            }
        u = i * n + i;
        b[i] = q[u];
        q[u] = 1.0;
        if (i - 1 >= 0)
            for (j = 0; j <= i - 1; j++)
            {
                q[i * n + j] = 0.0;
                q[j * n + i] = 0.0;
            }
    }
    return;
}

void Householder_0(double *a, int n, double* q, double* b, double* c)
{
    int k, j, i, u, v;
    for (k = n - 1; k >= 1; k--)
    {
        double h = 0.0;
        if (k > 1)
            for (i = 0; i <= k - 1; i++)
            {
                u = k * n + i;
                h = h + a[u] * a[u];
            }
        if (h + 1.0 == 1.0)
        {
            c[k] = 0.0;
            if (k == 1)
                c[k] = a[k * n + k - 1];
            b[k] = 0.0;
        }
        else
        {
            c[k] = sqrt(h);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                c[k] = -c[k];
            h = h - a[u] * c[k];
            a[u] = a[u] - c[k];

            double f = 0.0;
            double g = 0.0;
            for (j = 0; j <= k - 1; j++)
            {
                a[j * n + k] = a[k * n + j] / h;
                g = 0.0;
                for (int l = 0; l <= j; l++)
                    g = g + a[j * n + l] * a[k * n + l];
                if (j + 1 <= k - 1)
                    for (int l = j + 1; l <= k - 1; l++)
                        g = g + a[l * n + j] * a[k * n + l];
                c[j] = g / h;
                f = f + g * a[j * n + k];
            }
            double h2 = f / (h + h);
            for (j = 0; j <= k - 1; j++)
            {
                f = a[k * n + j];
                g = c[j] - h2 * f;
                c[j] = g;
                for (int l = 0; l <= j; l++)
                {
                    u = j * n + l;
                    a[u] = a[u] - f * c[l] - g * a[k * n + l];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    for (k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}


//Householder4
void Householder(double *a, int n, double* q, double* b, double* c)
{
    int u, v;
    for (int k = n - 1; k > 1; k--)
    {
        double sum = 0.0;

        for (int i = 0; i <= k - 1; i++)
        {
            u = k * n + i;
            sum = sum + a[u] * a[u];
        }

        if (sum == 0.0)
        {
            continue;
            // c[k] = 0.0;
            // b[k] = 0.0;
        }
        else
        {
            double mol = sqrt(sum);
            // c[k] = sqrt(sum);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                mol = -mol;
            c[k] = mol;
            sum = sum - a[u] * mol;
            a[u] = a[u] - mol;

            double f = 0.0;
            double g = 0.0;
            
            for (int i = 0; i <= k - 1; i++)
                a[i * n + k] = a[k * n + i] / sum;

            for (int i = 0; i <= k - 1; i++)
            {
                // a[i * n + k] = a[k * n + i] / sum;
                g = 0.0;
                // for (int j = 0; j <= i; j++)
                //     g = g + a[i * n + j] * a[k * n + j];
                // for (int j = i + 1; j <= k - 1; j++)
                //     g = g + a[j * n + i] * a[k * n + j];
                for (int j = 0; j <= k - 1; j++)
                    if(j<=i) g = g + a[i * n + j] * a[k * n + j];
                    else g = g + a[j * n + i] * a[k * n + j];

                c[i] = g / sum;
                f = f + g * a[i * n + k];
            }
            
            double h2 = f / (sum + sum);
            for (int i = 0; i <= k - 1; i++)
            {
                double elem = a[k * n + i];
                g = c[i] - h2 * elem;
                c[i] = g;
                for (int j = 0; j <= i; j++)
                {
                    u = i * n + j;
                    a[u] = a[u] - elem * c[j] - g * a[k * n + j];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    c[1] = a[n];
    for (int k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}




//QR
int QR(int N, double *b, double *c, double *q, double eps, int limit)
{
    int k, m, it, u, v;
    double d, f, h, g, p, r, e, s;
    c[N - 1] = 0.0;
    d = 0.0;
    f = 0.0;
    for (int j = 0; j <= N - 1; j++)
    {
        it = 0;
        h = eps * (fabs(b[j]) + fabs(c[j]));
        if (h > d)
            d = h;
        m = j;
        while ((m <= N - 1) && (fabs(c[m]) > d))
            m = m + 1;
        if (m != j)
        {
            do
            {
                if (it == limit)
                {
                    printf("fail\n");
                    return (-1);
                }
                it = it + 1;
                g = b[j];
                p = (b[j + 1] - g) / (2.0 * c[j]);
                r = sqrt(p * p + 1.0);
                if (p >= 0.0)
                    b[j] = c[j] / (p + r);
                else
                    b[j] = c[j] / (p - r);
                h = g - b[j];
                for (int i = j + 1; i <= N - 1; i++)
                    b[i] = b[i] - h;
                f = f + h;
                p = b[m];
                e = 1.0;
                s = 0.0;
                for (int i = m - 1; i >= j; i--)
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

                    // for (k = 0; k <= N - 1; k++)
                    // {
                    //     u = k * N + i + 1;
                    //     v = u - 1;
                    //     h = q[u];
                    //     q[u] = s * q[v] + e * h;
                    //     q[v] = e * q[v] - s * h;
                    // }
                }
                c[j] = s * p;
                b[j] = e * p;
            } while (fabs(c[j]) > d);
        }
        b[j] = b[j] + f;
    }
    // for (int i = 0; i <= N - 1; i++)
    // {
    //     k = i;
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
    //             u = j * N + i;
    //             v = j * N + k;
    //             p = q[j * N + i];
    //             q[j * N + i] = q[j * N + k];
    //             q[j * N + k] = p;
    //         }
    //     }
    // }
    return (1);
}

int mysolver_cpu(int N, double * Dev_A,double * Dev_W)
{
    int l = 1000000;

    double *a = new double[N * N];
    double *q = new double[N * N];
    double *b = new double[N];
    double *c = new double[N];

    double eps = 1e-7;

    hipMemcpy(a, Dev_A,N * N * sizeof(double), hipMemcpyDeviceToHost);

    Householder(a, N, q, b, c);
    QR(N, b, c, q, eps, l);
    sort(b,N);

    for(int i=0;i<N/2;i++)
        c[i]=b[i*2];

    //结果拷回
    hipMemcpy(Dev_W, c, N/2 * sizeof(double),hipMemcpyHostToDevice);

    free(a);
    free(q);
    free(b);
    free(c);

    return 1;
}

