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
int Householder_vector(int n, double A[], double B0[], double C0[], double *Qalpha)
{
    double mol, q, value, K;
    double *alpha = new double[n];
    double *B = new double[n * n];
    double *C = new double[n * n];
    double *S = new double[n];
    double *Q = new double[n];
    double *W = new double[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Qalpha[i * n + j] = 0.0;
        }
        Qalpha[i * n + i] = 1;
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
        // 求W向量
        for (int j = 0; j < n; j++)
        {
            value = 0.0;
            for (int k = 0; k < n; k++)
                value += Qalpha[j * n + k] * alpha[k] / q;
            W[j] = value;
        }
        // 求Qalpha(i+1)矩阵
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                Qalpha[j * n + k] = Qalpha[j * n + k] - W[j] * alpha[k];
            }
        }
        // 求S向量
        for (int j = 0; j < i + 1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i + 1; k++)
                value += A[j * n + k] * alpha[k] / q;
            S[j] = value;
        }
        // 求K的值
        K = 0;
        for (int j = 0; j < i + 1; j++)
        {
            K += alpha[j] * S[j] / (2 * q);
        }
        // 求Q向量
        for (int j = 0; j < i + 1; j++)
        {
            Q[j] = S[j] - K * alpha[j];
        }
        // 求A(i+1)矩阵
        for (int j = 0; j < i + 1; j++)
        {
            for (int k = 0; k < i + 1; k++)
            {
                A[j * n + k] = A[j * n + k] - alpha[j] * Q[k] - Q[j] * alpha[k];
            }
        }
    }
    // 抽出主、次对角线元素
    for (int i = 0; i < n - 1; i++)
    {
        B0[i] = A[i * n + i];
        C0[i] = A[i * n + i + 1];
    }
    B0[n - 1] = A[n * n - 1];
    C0[n - 1] = 0;
    delete[] alpha;
    delete[] B;
    delete[] C;
    delete[] S;
    return 1;
}

//QR（带特征向量）
int QR_vector_old(int N, double *b, double *c, double *q, double eps, int limit)
{
    int k, m, u, v;
    double d, f, h, g, p, r, e, s;
    d = 0.0;
    f = 0.0;
    for (int j = 0; j <= N - 1; j++)
    {
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
                g = b[j];
                p = (b[j + 1] - g) / (2.0 * c[j]);
                r = sqrt(p * p + 1.0);
                if (p >= 0.0)
                    b[j] = c[j] / (p + r);
                else
                    b[j] = c[j] / (p - r);
                double h = g - b[j];
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

                    for (k = 0; k <= N - 1; k++)
                    {
                        u = k * N + i + 1;
                        v = u - 1;
                        h = q[u];
                        q[u] = s * q[v] + e * h;
                        q[v] = e * q[v] - s * h;
                    }
                }
                c[j] = s * p;
                b[j] = e * p;
            } while (fabs(c[j]) > d);
        }
        b[j] = b[j] + f;
    }
    for (int i = 0; i <= N - 1; i++)
    {
        k = i;
        p = b[i];
        if (i + 1 <= N - 1)
        {
            int j = i + 1;
            while ((j <= N - 1) && (b[j] <= p))
            {
                k = j;
                p = b[j];
                j++;
            }
        }
        if (k != i)
        {
            b[k] = b[i];
            b[i] = p;
            for (int j = 0; j <= N - 1; j++)
            {
                u = j * N + i;
                v = j * N + k;
                p = q[j * N + i];
                q[j * N + i] = q[j * N + k];
                q[j * N + k] = p;
            }
        }
    }
    return (1);
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
    QR_vector_old(N, b, c, Qalpha, eps, l);

    sort_vector(b, N, Qalpha);


    //抽出一半特征值
    for (int i = 0; i < N / 2; i++)
        c[i] = b[i * 2];

    //抽出一半特征向量
    for (int i = 0; i < N / 2; i++)
    {
        if (fabs(a[N * (N - 1) + i * 2]) >= Eps)
        {
            for (int j = 0; j < N; j++)
            {
                y_l[j * N + i] = Qalpha[j * N + i * 2];
            }
        }
        else
        {
            for (int j = 0; j < N; j++)
            {
                y_l[j * N + i] = Qalpha[j * N + i * 2 + 1];
            }
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