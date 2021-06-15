#include "math.h"
#include "stdio.h"
#include "time.h"
double eps = 1e-8;

int Householder(int n, double A[])
{
    int i, j, k, u;
    double h, q, value;
    double *B = new double[n];
    double *C = new double[n*n];
    double *D = new double[n*n];

    for (i = n - 1; i > 1; i--)
    {
        q = 0.0;
        for (j = 0; j < n; j++)
        {
            B[j] = 0.0;
        }
        for (k = 0; k < i; k++)
        {
            B[k] = A[i * n + k];
            q += B[k] * B[k];
        }
        h = sqrt(q);
        if (B[i - 1] > 0.0)
            h = -h;
        q = q - B[i - 1] * h;
        B[i - 1] = B[i - 1] - h;

        for (j = 0; j <= n - 1; j++)
        {
            for (k = 0; k <= n - 1; k++)
                C[j * n + k] = -B[j] * B[k] / q;
            C[n * j+ j] = C[n * j + j] + 1.0;
        }


        printf("\nH 矩阵:\n");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%11lf  ", C[i * n + j]);
            }
            printf("\n");
        }


        for (j = 0; j <= n - 1; j++)
            for (k = 0; k <= n - 1; k++){
                value = 0.0;
                for (u = 0; u <= n - 1; u++)
                    value = value + C[u + j * n] * A[n * u + k];
                D[k+j*n] = value;
            }

        for (j = 0; j <= n - 1; j++)
            for (k = 0; k <= n - 1; k++){
                value = 0.0;
                for (u = 0; u <= n - 1; u++)
                    value = value + D[u+j*n] * C[n * u + k];
                A[k+j*n] = value;
            }

    }

    delete[] B;
    delete[] C;
    delete[] D;
    
    return 1;
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

//检查精度 type==0 为Householder检查 type==1为QR检查
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
        for (int x = 0; x < N - 1; x++)
        {
            if (fabs(A[(x + 1) * N + x]) > eps)
            {
                return 1;
            }
        }
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
            printf("%11lf  ", A[i * N + j]);
        }
        printf("\n");
    }
}

int main()
{

    //double A[16] = {4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0};
    
    
    // static double A[25] = {
    //     10.0, 1.0, 2.0, 3.0, 4.0,
    //     1.0, 9.0, -1.0, 2.0, -3.0,
    //     2.0, -1.0, 7.0, 3.0, -5.0,
    //     3.0, 2.0, 3.0, 12.0, -1.0,
    //     4.0, -3.0, -5.0, -1.0, 15.0
    //     };

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
    int n=10;

    // time_t start,end;
    // start = clock();
    Householder(n, A);
    // end = clock();
    // printf("time=%f\n",(double)(end-start)/CLK_TCK);

    printf("\n=============================================================================\n");
    printf("Householder 矩阵\n");
    for (int i = 0; i < n*n; i++)
    {
        printf("%12lf  ", A[i]);
        if ( (i +1) % n ==0)
            printf("\n");
    }

    printf("\n=============================================================================\n");
    printf("QR 矩阵\n");
    int num = 0;
    while (check(A, n, 1))
    {
        QR(A, n);
        num++;
        printf("\rNo.%d",num);
    }
    printf("\rTotal loop(QR):%d \n", num);
    show(A, n);
    
    printf("\n=============================================================================\n");
    double b[10];
    for(int i=0;i<n;i++){
        b[i]=A[i*n+i];
    }
    //InsertSort(b,10);
    for(int i=0;i<n;i++){
        printf("b[%d]:%12lf \n", i,b[i]);
    }

}
