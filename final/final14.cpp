#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "time.h"
#include "algorithm"
using namespace std;

double eps = 1e-8;

void show(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("% 5.3lf,  ", A[i * N + j]);
        }
        printf("\n");
    }
}

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

//Householder变换
int Householder(int n, double A[], double b[], double c[])
{
    double *alpha = new double[n];
    // double *B = new double[n * n];
    // double *C = new double[n * n];
    double *S = new double[n];
    double *Q = new double[n];
    b[n - 1] = A[n * n - 1];

    for (int i = n - 1; i > 1; i--)
    {
        double q = 0.0;
        //初始化alpha向量为0
        for (int j = 0; j < i + 1; j++)
            alpha[j] = 0.0;

        for (int j = 0; j < i; j++)
        {
            alpha[j] = A[i * n + j];
            q += alpha[j] * alpha[j];
        }
        // 判断q的值是否小于精度
        // if (q + 1.0 == 1.0)
        // {
        //     b[i - 1] = A[(i - 1) * n + i - 1];
        //     continue;
        // }
        double mol = sqrtl(q);

        if (alpha[i - 1] > 0.0)
            mol = -mol;
        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;

        // 求S向量
        for (int j = 0; j < i + 1; j++)
        {
            double value = 0.0;
            for (int k = 0; k < i + 1; k++)
                value += A[j * n + k] * alpha[k] / q;
            S[j] = value;
        }
        // 求K的值
        double K = 0;
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
                //================================================================
                // B[j * n + k] = alpha[j] * Q[k];
                // C[j * n + k] = Q[j] * alpha[k];
                // A[j * n + k] = A[j * n + k] - B[j * n + k] - C[j * n + k];
                //================================================================
                A[j * n + k] = A[j * n + k] - alpha[j] * Q[k] - Q[j] * alpha[k];
                // Q[j*n+k] = Q[j*n+k]-B[j*n+k]-C[j*n+k];
            }
        }
        b[i - 1] = A[n * (i - 1) + i - 1];
        c[i] = A[n * i + i - 1];
    }
    b[0] = A[0];
    c[1] = A[n];

    for (int i = 0; i < n - 1; i++)
    {
        c[i] = c[i + 1];
    }
    c[n - 1] = 0;

    delete[] alpha;
    // delete[] B;
    // delete[] C;
    delete[] S;
    delete[] Q;
    return 1;
}

int Householder_cpu(int n, double A[], double b[], double c[])
{
    double mol, q, value;
    double *alpha = new double[n];
    double *S = new double[n];
    double *Q = new double[n];

    b[n - 1] = A[n * n - 1];
    for (int i = n - 1; i > 1; i--)
    {
        q = 0.0;
        //初始化alpha向量为0
        for (int j = 0; j < i + 1; j++)
            alpha[j] = 0.0;

        for (int j = 0; j < i; j++)
        {
            alpha[j] = A[i * n + j];
            q += alpha[j] * alpha[j];
        }
        
        // for(int k=0;k<n;k++)
        //     printf("%10lf  ",alpha[k]);
        // printf("%10lf\n",q);
        // printf("\n\n");

        mol = sqrt(q);

        if (alpha[i - 1] > 0.0)
            mol = -mol;
        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;
        
        
        // printf("%10lf %10lf %10lf\n",alpha[i-1],q ,mol);
        
        
        // 求S向量
        for (int j = 0; j < i + 1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i + 1; k++)
                value += A[j * n + k] * alpha[k] / q;
            S[j] = value;
        }

        // for(int k=0;k<n;k++)
        //     printf("%10lf  ",S[k]);
        // printf("\n\n");



        // 求K的值
        double K = 0;
        for (int j = 0; j < i + 1; j++)
        {
            K += alpha[j] * S[j] / (2 * q);
        }
        // 求Q向量
        for (int j = 0; j < i + 1; j++)
        {
            Q[j] = S[j] - K * alpha[j];
        }

        // for(int k=0;k<n;k++)
        //     printf("%10lf  ",Q[k]);
        // printf("\n\n");


        // 求A(i+1)矩阵
        for (int j = 0; j < i + 1; j++)
        {
            for (int k = 0; k < i + 1; k++)
            {
                A[j * n + k] = A[j * n + k] - alpha[j] * Q[k] - Q[j] * alpha[k];
                // Q[j*n+k] = Q[j*n+k]-B[j*n+k]-C[j*n+k];
            }
        }
    }
    for(int i=0;i<n-1;i++){
        b[i] = A[i*n+i];
        c[i] = A[i*n+i+1];
    }
    b[n-1]=A[n*n-1];
    c[n-1]=0;

    for(int k=0;k<n;k++)
        printf("%10lf  ",c[k]);
    printf("\n\n");


    delete[] alpha;
    delete[] S;
    delete[] Q;
    return 1;
}

//QR
int QR(int N, double *b, double *c, double eps)
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


int main()
{
    int N = 10;
    // double *A = new double[N * N];
    // symmat(A, N);

    double A[100] = {
        1.0, 2.0, 3.0, 4.0, 5.0, 0.0, -1.0, -1.0, -1.0, -1.0,
        2.0, 2.0, 3.0, 4.0, 6.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        3.0, 3.0, 3.0, 1.0, 5.0, 1.0, -1.0, 0.0, 0.0, 0.0,
        4.0, 4.0, 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
        5.0, 6.0, 5.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0, 6.0,
        -1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 5.0,
        -1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 1.0, 3.0, 1.0,
        -1.0, 0.0, 0.0, -1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 3.0
    };


    double *b = new double[N];
    double *c = new double[N];


    time_t start, end;

    printf("========================================================\n");
    printf("Householder\n");
    start = clock();
    Householder_cpu(N, A, b, c);
    end = clock();
    printf("HS time=%lfs\n", (double)(end - start) / CLOCKS_PER_SEC);

    printf("========================================================\n");
    start = clock();
    QR(N, b, c, eps);
    end = clock();
    sort(b, b + N);
    printf("QR time=%lfs\n", (double)(end - start) / CLOCKS_PER_SEC);

    printf("========================================================\n");
    int error=0;
    for (int i = 0; i < N; i = i + 2)
    {
        double k = b[i + 1] - b[i];
        if (k >= eps)
            error += 1;
        // else
        //     continue;
        printf("b[%d]: %17.14lf || %17.14lf || %10.8lf\n", i, b[i], b[i+1],k);
    }
    if(error==0) printf("No error\n");

    // delete[] A;
    delete[] b;
    delete[] c;

    return 0;
}
