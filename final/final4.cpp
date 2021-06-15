#include "math.h"
#include "stdio.h"
#include "time.h"
#include "stdlib.h"
double eps = 1.0e-7;
int Householder(int n, double *A)
{
    int i, j, k, u;
    double h, q, value;
    double *B = new double[n];
    double *C = new double[n * n];
    double *D = new double[n * n];

    for (i = n - 1; i > 1; i--)
    {
        q = 0.0;
        for (j = 0; j < i + 1; j++)
        {
            B[j] = 0.0;
        }
        for (k = 0; k < i; k++)
        {
            B[k] = A[i * n + k];
            q += B[k] * B[k];
        }
        h = sqrtl(q);
        if (B[i - 1] > 0.0)
            h = -h;
        q = q - B[i - 1] * h;
        B[i - 1] = B[i - 1] - h;
        for (j = 0; j < i + 1; j++)
        {
            for (k = 0; k < i + 1; k++)
                C[j * n + k] = -B[j] * B[k] / q;
            C[n * j + j] = C[n * j + j] + 1.0;
        }
        for (j = 0; j < i + 1; j++)
            for (k = 0; k <= i + 1; k++)
            {
                value = 0.0;
                for (u = 0; u < i + 1; u++)
                    value = value + C[u + j * n] * A[n * u + k];
                D[k + j * n] = value;
            }
        for (j = 0; j < i + 1; j++)
            for (k = 0; k < i + 1; k++)
            {
                value = 0.0;
                for (u = 0; u < i + 1; u++)
                    value = value + D[u + j * n] * C[n * u + k];
                A[k + j * n] = value;
            }
    }

    delete[] B;
    delete[] C;
    delete[] D;

    return 1;
}
// int QR(double *A, int N)
// {
//     double *Q = new double[N * N];
//     double *R = new double[N * N];
//     for (int k = 0; k < N - 1; k++)
//     {
//         if (fabs(A[k * N + (k + 1)]) <= eps)
//         {
//             continue;
//         }
//         double q= (A[k * N + k]-A[(k+1) * N + (k+1)])/2;
//         double p;
//         if (q>=eps)
//         {
//             p= A[(k +1)* N + (k+1)]-A[k * N + k+1]*A[k * N + k+1]/(q+sqrtl(q*q+A[k * N + k+1]*A[k * N + k+1]));
//         }
//         else
//         {
//             p= A[(k +1)* N + (k+1)]-A[k * N + k+1]*A[k * N + k+1]/(q-sqrtl(q*q+A[k * N + k+1]*A[k * N + k+1]));
//         }
//         double elem1 = (A[k * N + k]- p) * (A[k * N + k]-p);
//         double elem2 = A[(k + 1) * N + k] * A[(k + 1) * N + k];
//         double r = sqrtl(elem1 + elem2);
//         //printf("%9lf  %9lf  %9lf\n", A[k*N+k] ,A[(k+1)*N+k],r);
//         double Cos = (A[k * N + k]-p) / r;
//         double Sin = A[(k + 1) * N + k] / r;
//         for (int i = 0; i < N; i++)
//         {
//             for (int j = 0; j < N; j++)
//             {
//                 R[i * N + j] = 0;
//                 if (i == j)
//                     R[i * N + j] = 1;
//             }
//         }
//         R[k * N + k] = Cos;
//         R[k * N + (k + 1)] = Sin;
//         R[(k + 1) * N + k] = Sin * -1;
//         R[(k + 1) * N + (k + 1)] = Cos;
//         for (int i = 0; i < N; i++)
//             for (int j = 0; j < N; j++)
//             {
//                 Q[i * N + j] = 0.0;
//                 for (int k = 0; k < N; k++)
//                     Q[i * N + j] += R[i * N + k] * A[k * N + j];
//             }
//         for (int i = 0; i < N; i++)
//             for (int j = 0; j < N; j++)
//             {
//                 A[i * N + j] = 0.0;
//                 for (int k = 0; k < N; k++)
//                     A[i * N + j] += Q[i * N + k] * R[k + j * N];
//             }
//     }
//     // printf("\nA 矩阵:\n");
//     // for (int i = 0; i < N; i++)
//     // {
//     //     for (int j = 0; j < N; j++)
//     //     {
//     //         printf("%9lf  ", A[i * N + j]);
//     //     }
//     //     printf("\n");
//     // }
//     delete[] Q;
//     delete[] R;
//     // return flag;
//     return 1;
// }
int qr(int n, double *A)
{
    int i, j, k, u;
    double h, q, value;
    double *B = new double[n];
    double *C = new double[n * n];
    double *D = new double[n * n];

    for (i = n - 1; i > 0; i--)
    {
        q = 0.0;
        for (j = 0; j < n; j++)
        {
            B[j] = 0.0;
        }
        for (k = 0; k < i + 1; k++)
        {
            B[k] = A[i * n + k];
            q += B[k] * B[k];
        }
        h = sqrtl(q);
        if (B[i] > 0.0)
            h = -h;
        q = q - B[i] * h;
        B[i] = B[i] - h;
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
                C[j * n + k] = -B[j] * B[k] / q;
            C[n * j + j] = C[n * j + j] + 1.0;
        }
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
            {
                value = 0.0;
                for (u = 0; u < n; u++)
                    value = value + C[u + j * n] * A[n * u + k];
                D[k + j * n] = value;
            }
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
            {
                value = 0.0;
                for (u = 0; u < n; u++)
                    value = value + D[u + j * n] * C[n * u + k];
                A[k + j * n] = value;
            }
    }

    delete[] B;
    delete[] C;
    delete[] D;

    return 1;
}
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
int QRimprove(double *A, int N)
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
    for (int k = 0; k < N - 1; k++)
    {
        // if (fabs(A[k * N + (k + 1)]) <= eps)
        // {
        //    continue;
        // }
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

// 随机生成10阶对称矩阵
void symmat(double B[][100], double A[], int n)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (j > i)
                B[i][j] = rand() % 10 + 1;
            else if (i > j)
                B[i][j] = B[j][i];
            else
                B[i][j] = rand() % 10 + 1;
        }
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            A[i * n + j] = B[i][j];
    }
}
//打印结果矩阵
void show(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%11.8lf ", A[i * N + j]);
        }
        printf("\n");
    }
}

// //检查精度 type==0 为Householder检查 type==1为QR检查
int check(double *A, int N, int type)
{
    if (type == 0)
    {
        for (int x = 0; x < N - 2; x++)
        {
            if (fabs(A[(x + 2) * N + x]) > eps)
            {
                return 1;
            }
        }
    }
    else
    {
        for (int x = 0; x < N * N - 1; x++)
        {
            if (x % (N + 1) == 0)
            {
                continue;
            }
            if (fabs(A[x]) > eps)
            {
                return 1;
            }
        }
    }
    return 0;
}

//检查精度 type==0 为Householder检查 type==1为QR检查
// int check(double *A, int N, int type)
// {
//     if (type == 0)
//     {
//         for (int x = 0; x < N - 2; x++)
//         {
//             if (fabs(A[(x + 2) * N + x]) > eps)
//             {
//                 return 1;
//             }
//         }
//     }
//     else
//     {
//         for (int x = 0; x < N - 1; x++)
//         {
//             if (fabs(A[(x + 1) * N + x]) > eps)
//             {
//                 return 1;
//             }
//         }
//     }
//     return 0;
// }

int main()
{
    int n = 10;
    // double A[16] = {4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0};
    //double B[100][100];
    //double *A = new double[n * n];
    //symmat(B, A, n); // 随机生成10阶对称矩阵

    double A[100] = {1.0, 2.0,  3.0,  4.0, 5.0, 0.0, -1.0, -1.0, -1.0, -1.0,
                    2.0, 2.0,  3.0,  4.0, 6.0, 1.0,  0.0,  1.0,  0.0,  0.0,
                    3.0, 3.0,  3.0,  1.0, 5.0, 1.0, -1.0,  0.0,  0.0,  0.0,
                    4.0, 4.0,  1.0,  3.0, 1.0, 1.0,  0.0,  0.0,  0.0, -1.0,
                    5.0, 6.0,  5.0,  1.0, 3.0, 1.0,  0.0,  0.0,  1.0,  0.0,
                    0.0, 1.0,  1.0,  1.0, 1.0, 1.0,  2.0,  3.0,  4.0,  5.0,
                    -1.0, 0.0, -1.0,  0.0, 0.0, 2.0,  2.0,  3.0,  4.0,  6.0,
                    -1.0, 1.0,  0.0,  0.0, 0.0, 3.0,  3.0,  3.0,  1.0,  5.0,
                    -1.0, 0.0,  0.0,  0.0, 1.0, 4.0,  4.0,  1.0,  3.0,  1.0,
                    -1.0, 0.0,  0.0, -1.0, 0.0, 5.0,  6.0,  5.0,  1.0,  3.0};

    // for (int i = 0; i < n*n; i++)
    // {
    //        C[i]=A[i];
    // }
    // for (int i = 0; i < n * n; i++) //打印三对角矩阵
    // {
    //     printf("%13.8e ", A[i]);
    //     if ((i + 1) % n == 0)
    //         printf("\n");
    // }
    Householder(n, A);
    printf("\n=============================================================================\n");
    printf("Householder 矩阵\n");
    for (int i = 0; i < n * n; i++) //打印三对角矩阵
    {
        printf("%10.7lf ", A[i]);
        if ((i + 1) % n == 0)
            printf("\n");
    }
    printf("\n=============================================================================\n");
    // printf("QR 矩阵\n");
    // int num = 0;
    // while (check(C, n, 1))
    // {
    //     QR(C, n);
    //     num++;
    //     // printf("\rNo.%d",num);
    // }
    // printf("\rTotal loop(QR):%d \n", num);
    // show(C, n);
    //  printf("\n=============================================================================\n");
    // printf("QRimprove 矩阵\n");
    // for (int i = 0; i < 8; i++)
    // {
    //    QRimprove(A,n);
    //    printf("No.%d ",i);
    // }
    // printf("\n");
    // show(A, n);

    int num1 = 0;
    while (check(A, n, 1))
    {
        // QR(A,n);
        qr(n,A);
        // QRimprove(A,n);
        num1++;
        printf("\rNo.%d",num1);
    }
    printf("\rTotal loop(QRimprove):%d \n", num1);
    show(A, n);
    printf("\n");
    // printf("    %13.7lf \n", eps);
    printf("\n=============================================================================\n");
    // double b[n];
    // for (int i = 0; i < n; i++)
    // {
    //     b[i] = A[i * n + i];
    // }
    // //InsertSort(b,10);
    // for (int i = 0; i < n; i++)
    // {
    //     printf("b[%d]:%13.8e \n", i, b[i]);
    //     // printf("   %13.7lf \n", eps);
    // }
    //delete[] A;
}