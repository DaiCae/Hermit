#include "stdio.h"
#include "math.h"
#include "time.h"
#include "stdlib.h"


//Householder变换
int Householder_old(int n, double A[],double *b,double* c)
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

        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;
        
        // printf("\nmol:%11lf\n", q);
        
        for (int j = 0; j <= n - 1; j++){
            for (int k = 0; k <= n - 1; k++)
                H[j * n + k] = -1 * alpha[j] * alpha[k] / q;
            H[n * j + j] = H[n * j + j] + 1.0;
        }

        for (int j = 0; j < i+1; j++)
            for (int k = 0; k < i+1; k++)
            {
                value = 0.0;
                for (int u = 0; u < i+1; u++)
                    value = value + H[u + j * n] * A[n * u + k];
                B[k + j * n] = value;
            }

        for (int j = 0; j <  i+1; j++)
            for (int k = 0; k < i+1; k++)
            {
                value = 0.0;
                for (int u = 0; u < i+1; u++)
                    value = value + B[u + j * n] * H[n * u + k];
                A[k + j * n] = value;
            }
        
        for(int i=0;i<n;i++){
            b[i]=A[i*n+i];
        }
        for(int i=0;i<n-1;i++){
            c[i]=A[(i+1)*n+i];
        }
    }    

    delete[] alpha;
    delete[] H;
    delete[] B;

    return 1;
}


//Householder
void Householder(double *a,int n, double* q, double* b, double* c)
{
    int i, j, k, u, v;
    double h, f, g, h2;
    for (i = 0; i < n*n; i++)
        q[i] = a[i];
    
    for (i = n - 1; i >= 1; i--)
    {
        //printf("No.%d\r",i);
        h = 0.0;
        if (i > 1)
            for (k = 0; k <= i - 1; k++)
            {
                u = i * n + k;
                h = h + q[u] * q[u];
            }
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
            f = 0.0;
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
            h2 = f / (h + h);
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
                g = 0.0;
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

//QR
int QR(int N, double *b, double *c, double *q, double eps)
{
    int k, m, u, v;
    double d, f, h, g, p, r, e, s;
    c[N - 1] = 0.0;
    d = 0.0;
    f = 0.0;
    // double h1 = eps * (fabs(b[0]) + fabs(c[0]));
    // printf("%.15lf\n",h1);
    for (int j = 0; j <= N - 1; j++)
    {
        //d=eps;

        //////////////////////////////////////////////////////
        //动态精度值调整
        h = eps * (fabs(b[j]) + fabs(c[j]));
        if (h > d)
            d = h;
        //////////////////////////////////////////////////////


        //////////////////////////////////////////////////////
        //子矩阵抽取
        m = j;
        while ((m <= N - 1) && (fabs(c[m]) > d))
            m = m + 1;
        if (m != j)        
        //////////////////////////////////////////////////////
        // m = N-1;
        // if (1)
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
            // printf("[%d] ",m);
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
//QR
int QR1(int N, double *b, double *c, double *q, double eps)
{
    int k, m, u, v;
    double d, f, h, g, p, r, e, s;
    c[N - 1] = 0.0;
    d = 0.0;
    f = 0.0;
    // double h1 = eps * (fabs(b[0]) + fabs(c[0]));
    // printf("%.15lf\n",h1);
    for (int j = 0; j <= N - 1; j++)
    {
        //d=eps;

        //////////////////////////////////////////////////////
        //动态精度值调整
        h = eps * (fabs(b[j]) + fabs(c[j]));
        if (h > d)
            d = h;
        //////////////////////////////////////////////////////


        //////////////////////////////////////////////////////
        //子矩阵抽取
        // m = j;
        // while ((m <= N - 1) && (fabs(c[m]) > d))
        //     m = m + 1;
        // if (m != j)        
        //////////////////////////////////////////////////////
        m = N-1;
        do
        {
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


//排序
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

//随机Her矩阵生成
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
                A[i * n + j] = rand()%100 / double((101)) *10;
                B[i * n + j] = rand()%100 / double((101)) *10;
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

int cycle(int N)
{
    // N=10;
    // double a[100] = {
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

    double *a = new double[N * N];
    symmat(a,N);

    int i, j, k, l = 1000000;
    double *q = new double[N * N];
    double *b = new double[N];
    double *c = new double[N];
    // double q[N*N], b[N], c[N];

    double *b1 = new double[N];
    double *c1 = new double[N];





    printf("\n====================================================\n");
    
    double eps = 1e-7;
    time_t start,end;

    start=clock();
    Householder(a, N, q, b, c);
    // Householder_old(N,a, b, c);
    // Householder_old(N,a, b, c);
    end=clock();
    printf("HS Time=%lf(ms)\n",(double)(end-start)/CLOCKS_PER_SEC*1000);
    
    
    
    
    // printf("Householder:\n");
    // for (i = 0; i <N; i++)
    // {
    //     for (j = 0; j <N; j++){
    //         if(i==j)
    //             printf("%10.7lf ", b[i]);
    //         else if(i-j==1)
    //             printf("%10.7lf ", c[i-1]);
    //         else if(i-j==-1)
    //             printf("%10.7lf ", c[i]);
    //         else
    //             printf("%10.7lf ", 0.0000000000);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    for(int i=0;i<N;i++)
    {
        b1[i]=b[i];
        c1[i]=c[i];
    }
    
    start=clock();
    k = QR(N, b, c, q, eps);
    end=clock();
    printf("QR Time=%lf(ms)\n",(double)(end-start)/CLOCKS_PER_SEC*1000);


    start=clock();
    k = QR1(N, b1, c1, q, eps);
    end=clock();
    printf("QR1 Time=%lf(ms)\n",(double)(end-start)/CLOCKS_PER_SEC*1000);

    // for (i = 0; i <N; i++){
    //         printf("|%20.16lf  |\n", b[i]);
    //     }

    int err=0;
    if (k > 0)
    {
        sort(b,N);
        sort(b1,N);
        printf("|-----------------|-----------------|--------------|\n");
        printf("|      Eige       |       Eige      |     Diff     |\n");
        printf("|-----------------|-----------------|--------------|\n");
        for (i = 0; i <N; i+=2){
            // printf("%10.7lf \n", b[i]);
            double diff=fabs(b1[i+1]-b1[i]);
            // if(diff<=1e-7) continue;

            printf("|%15.8lf  |", b1[i]);
            printf("%15.8lf  |", b[i+1]);
            printf("%12.15lf  |\n",diff);
            if(diff>1e-7) 
            {
                err++;
            }
        }
        printf("                                              Err:%2d",err);
    
    // printf("====================================================\n\n\n");
        // printf("MAT Q IS:\n");
        // for (i = 0; i <N; i++)
        // {
        //     for (j = 0; j <N; j++)
        //         printf("%10.7lf ", q[i*N+j]);
        //     printf("\n");
        // }
        // printf("\n");
    }

    if(err>0) err=1;
    return err;
}

int main()
{
    int N=64;
    int eee=0;
    for(int i=0;i<1;i++){
        eee+=cycle(N);
    }
    printf("\nTotal Error: %d\n",eee);
    printf("\nN: %d\n",N);
}
