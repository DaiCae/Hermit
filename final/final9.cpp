#include "math.h"
#include "stdio.h"
#include <stdlib.h>
#include "time.h"
double eps = 1e-8;

int Householder(int n, double A[])
{
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

        // for (int j = 0; j < n; j++)
        // {
        //     printf("%11lf ", alpha[j]);
        // }
        // printf("\nmol:%11lf\n", mol);

        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;
        
        // printf("\nmol:%11lf\n", q);
        
        for (int j = 0; j <= n - 1; j++){
            for (int k = 0; k <= n - 1; k++)
                H[j * n + k] = -1 * alpha[j] * alpha[k] / q;
            H[n * j + j] = H[n * j + j] + 1.0;
        }

        // printf("\nH 矩阵:\n");
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         printf("%11lf  ", H[i * n + j]);
        //     }
        //     printf("\n");
        // }

        for (int j = 0; j < i+1; j++)
            for (int k = 0; k < i+1; k++)
            {
                value = 0.0;
                for (int u = 0; u < i+1; u++)
                    value = value + H[u + j * n] * A[n * u + k];
                B[k + j * n] = value;
            }
        
        // printf("\nB 矩阵:\n");
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         printf("%11lf  ", B[i * n + j]);
        //     }
        //     printf("\n");
        // }

        for (int j = 0; j <  i+1; j++)
            for (int k = 0; k < i+1; k++)
            {
                value = 0.0;
                for (int u = 0; u < i+1; u++)
                    value = value + B[u + j * n] * H[n * u + k];
                A[k + j * n] = value;
            }

        // printf("\nA 矩阵:\n");
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         printf("%11lf  ", A[i * n + j]);
        //     }
        //     printf("\n");
        // }
        
    }
    

    delete[] alpha;
    delete[] H;
    delete[] B;

    return 1;
}

//通过QR(Givens rotation)方法求出全部特征值 A为三对角矩阵 N为矩阵阶数
int QR_OLD(double *A, int N)
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

//循环数目减少一倍
int QR(double *A, int N)
{

    double *Q = new double[N * N];
    double *R = new double[N * N];

    for (int k = 0; k < (N - 1)/2; k++)
    {
        // if (fabs(A[k * N + (k + 1)]) <= eps)
        // {
        //     continue;
        // }

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

        // printf("\nA 矩阵:\n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%9lf  ", A[i * N + j]);
        //     }
        //     printf("\n");
        // }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                Q[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    Q[i * N + j] += R[i * N + k] * A[k * N + j];
            }

        // printf("\nA 矩阵:\n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%9lf  ", Q[i * N + j]);
        //     }
        //     printf("\n");
        // }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    A[i * N + j] += Q[i * N + k] * R[k + j * N];
            }
            
        //printf("\n=================================================================\n");
    
    }
    
    delete[] Q;
    delete[] R;
    // return flag;
    return 1;
}

//改进的QR
int QR_improve_old(double *A, int N)
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

//对角线从上到下逼近
int QR_close(double *A, int N)
{

    double *Q = new double[N * N];
    double *R = new double[N * N];
    
    int k=0,num=0;
    while (k < N-1)
    {
        if (fabs(A[k * N + (k + 1)]) - eps <= 0 )
        {
            printf("////////////////////////////////////////////////////////////////\n");
            printf("No.%d is ok! Times:%d\n",k,num);
            printf("////////////////////////////////////////////////////////////////\n");
            printf("%lf %lf %lf\n",fabs(A[k * N + (k + 1)]),eps,fabs(A[k * N + (k + 1)])-eps);
            k++;
            num=0;

            // printf("\nA 矩阵:\n");
            // for (int i = 0; i < N; i++)
            // {
            //     for (int j = 0; j < N; j++)
            //     {
            //         printf("%9lf  ", Q[i * N + j]);
            //     }
            //     printf("\n");
            // }
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

        // printf("\nA 矩阵:\n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%9lf  ", A[i * N + j]);
        //     }
        //     printf("\n");
        // }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                Q[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    Q[i * N + j] += R[i * N + k] * A[k * N + j];
            }

        // printf("\nA 矩阵:\n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%9lf  ", Q[i * N + j]);
        //     }
        //     printf("\n");
        // }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    A[i * N + j] += Q[i * N + k] * R[k + j * N];
            }
            

        num++;
        //printf("\n=================================================================\n");
    }
    delete[] Q;
    delete[] R;
    // return flag;
    return 1;
}

//将矩阵分为4组后分组QR计算
int QR_close_group(double *A, int N)
{
    double *Q = new double[N * N];
    double *R = new double[N * N];

    for(int id=0;id<4;id++){
        for (int k = 0; k < (N - 1)/2; k++)
        {
            if(k % 4 != id){
                continue;
            }

            //printf("\n=================================================================\n");
            
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

            // printf("\nA 矩阵:\n");
            // for (int i = 0; i < N; i++)
            // {
            //     for (int j = 0; j < N; j++)
            //     {
            //         printf("%9lf  ", A[i * N + j]);
            //     }
            //     printf("\n");
            // }

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    Q[i * N + j] = 0.0;
                    for (int k = 0; k < N; k++)
                        Q[i * N + j] += R[i * N + k] * A[k * N + j];
                }

            // printf("\nA 矩阵:\n");
            // for (int i = 0; i < N; i++)
            // {
            //     for (int j = 0; j < N; j++)
            //     {
            //         printf("%9lf  ", Q[i * N + j]);
            //     }
            //     printf("\n");
            // }

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    A[i * N + j] = 0.0;
                    for (int k = 0; k < N; k++)
                        A[i * N + j] += Q[i * N + k] * R[k + j * N];
                }
                
            //printf("\n=================================================================\n");
        

        }
    
    }

    delete[] Q;
    delete[] R;
    // return flag;
    return 1;
}

//进行一半的QR检查
int check(double *A, int N){
    // for (int x = 0; x < N/2-1; x++){
    //     for (int y = 0; y < N/2-1; y++){
    //         if(x==y) continue;
    //         if (fabs(A[x * N + y]) > eps)
    //             return 1;
    //     }
    // }
    // return 0;
    for (int x = 0; x < (N - 1)/2; x++){
        if (fabs(A[(x + 1) * N + x]) > eps)
            return 1;
    }
    return 0;
}

int check_old(double *A, int N){
    for (int x = 0; x < N - 1; x++){
        if (fabs(A[(x + 1) * N + x]) > eps)
            return 1;
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
            printf("%11.7lf,", A[i * N + j]);
        }
        printf("\n");
    }
}

// 随机生成10阶对称矩阵
void symmat(double H[], int N)
{
    int n=N/2;
    double *A = new double[n * n];
    double *B = new double[n * n];

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
    {
        for(int j=0 ;j< n; j++){
            if(j >= i){
                A[i * n + j] = rand() % 20 -10;
                B[i * n + j] = rand() % 20 -10;
            }
            else{
                A[i * n + j] = A[j * n + i];
                B[i * n + j] = -1 * B[j * n + i];
            }
        }
        B[i*n+i]=0;
    }

    // show(A,n);
    // printf("\n");
    // show(B,n);
    // printf("\n");

    for (int i = 0; i < N; i++)
    {
        for(int j = 0;j <N ;j++)
        {
            if((i>=n&&j>=n)||(i<n && j<n)) 
                H[i*N+j]= A[i%n * n + j%n];
            else if(i<n)
                H[i*N+j]= -1 * B[i%n * n + j%n];
            else
                H[i*N+j]= B[i%n * n + j%n];
        }
    }
}

int main()
{
    int n =10;
    // double *A = new double[n * n];
    //symmat(A,n);

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




    show(A, n);

    time_t start,end;
    printf("\n=============================================================================\n");
    printf("Householder 矩阵\n");
    start = clock();
    Householder(n, A);
    end = clock();
    printf("time=%f\n",(double)(end-start));
    show(A, n);

    printf("\n=============================================================================\n");
    printf("QR 矩阵\n");
    int num = 0;
    start = clock();
    while (check(A, n)){
        QR_improve(A, n);
        printf("\rNo.%d", ++num);
        if(num > 10000) break;
    }
    end = clock();
    printf("\rTotal loop(QR):%d \n", num);
    printf("time=%lf\n",(double)(end-start));
    show(A, n);

    // printf("\n=============================================================================\n");
    // double b[10];
    // for (int i = 0; i < n; i++)
    // {
    //     b[i] = A[i * n + i];
    // }
    // //InsertSort(b,10);
    // for (int i = 0; i < n; i++)
    // {
    //     printf("b[%d]:%12lf \n", i, b[i]);
    // }
}
