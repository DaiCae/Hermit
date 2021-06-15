#include "math.h"
#include "stdio.h"
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
        //��ʼ��alpha����Ϊ0
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

        for (int j = 0; j <= n - 1; j++){
            for (int k = 0; k <= n - 1; k++)
                H[j * n + k] = -alpha[j] * alpha[k] / q;
            H[n * j + j] = H[n * j + j] + 1.0;
        }

        // printf("\nH ����:\n");
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         printf("%11lf  ", H[i * n + j]);
        //     }
        //     printf("\n");
        // }

        // for (int j = 0; j < n; j++)
        //     for (int k = 0; k < n; k++)
        //     {
        //         value = 0.0;
        //         for (int u = 0; u < n; u++)
        //             value = value + H[u + j * n] * A[n * u + k];
        //         B[k + j * n] = value;
        //     }

        // for (int j = 0; j < n; j++)
        //     for (int k = 0; k < n; k++)
        //     {
        //         value = 0.0;
        //         for (int u = 0; u < n; u++)
        //             value = value + B[u + j * n] * H[n * u + k];
        //         A[k + j * n] = value;
        //     }

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

        // printf("\nA ����:\n");
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

//ͨ��QR(Givens rotation)�������ȫ������ֵ AΪ���ԽǾ��� NΪ�������
int QR(double *A, int N)
{

    double *P = new double[N * N];
    double *R = new double[N * N];

    for (int r = 0; r < N - 1; r++)
    {
        double elem1 = A[r * N + r];
        double elem2 = A[r * N + (r + 1)];
        double elem3 = A[(r + 1) * N + (r + 1)];
        double elem4 = A[(r+1)*N+(r+2)];

        double g=(elem3-elem1)*0.5 *elem2;
        double sgn=1;
        if(g<0) sgn=-1;
        double ks=elem1-elem2/(g + sgn * sqrt(1 + g * g));

        // double x = sqrt(elem4* elem4 +(elem1-ks)*(elem1-ks));
        // double Cos = (elem3 - ks)/x;
        // double Sin = (-1 * elem2)/x;

        double x = 1;
        double Cos = 1;
        double Sin = 1;

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                R[i * N + j] = 0;
                if (i == j)
                    R[i * N + j] = 1;
            }
        }

        R[r * N + r] = Cos;
        R[r * N + (r + 1)] = Sin;
        R[(r + 1) * N + r] = Sin * -1;
        R[(r + 1) * N + (r + 1)] = Cos;

        
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                P[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    P[i * N + j] += A[i * N + k] * R[k * N + j];
            }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 0.0;
                for (int k = 0; k < N; k++)
                    A[i * N + j] += R[i + k * N] * P[k + j * N];
            }
    
        printf("\nA ����:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%9lf ", A[i * N + j]);
            }
            printf("\n");
        }
        break;
    }


    delete[] P;
    delete[] R;

    // return flag;
    return 1;
}

//���QR���
int check(double *A, int N){
    for (int x = 0; x < N - 1; x++){
        if (fabs(A[(x + 1) * N + x]) > eps)
            return 1;
    }
    return 0;
}

//��ӡ�������
void show(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%9lf ", A[i * N + j]);
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
    //int N=1000;

    int n=10;
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

    // int n = 100;
    // double *A = new double[n * n];
    // for(int i=0;i<n;i++)
    // {
    //     for(int j=0;j<n;j++)
    //     {
    //         if(i>j)
    //             A[i*n+j]=i-0.1*j+1;
    //         else
    //             A[i*n+j]=j-0.1*i+1;
    //     }
    // }
    //show(A, n);

    time_t start,end;
    printf("\n=============================================================================\n");
    printf("Householder ����\n");

    start = clock();
    Householder(n, A);
    end = clock();

    printf("time=%f\n",(double)(end-start));


    show(A, n);

    printf("\n=============================================================================\n");
    printf("QR ����\n");
    int num = 0;
    while (check(A, n)){
        QR(A, n);
        printf("\rNo.%d", ++num);
    }
    printf("\rTotal loop(QR):%d \n", num);
    //show(A, n);

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
