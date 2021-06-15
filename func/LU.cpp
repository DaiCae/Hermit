#include<stdio.h>
#include<math.h>

void LU(double *A, double*b, double *X ,int N)
{
    //中间项
    double *Y =new double[N];
    //选取列主元的比较器
    double *S =new double[N];

    int i, j, k;            //计数器
    for (k = 0; k < N; k++)
    {
        //选列主元
        int index = k;
        for (i = k; i < N; i++)
        {
            double temp = 0;
            for (int m = 0; m < k; m++)
            {
                temp = temp + A[i*N+m] * A[m*N+k];
            }
            S[i] = A[i*N+k] - temp;
            if (S[index] < S[i])
            {
                index = i;
            }
        }
        //交换行
        double temp;
        for (i = k; i < N; i++)
        {
            temp = A[index*N+i];
            A[index*N+i] = A[k*N+i];
            A[k*N+i] = temp;
        }
        temp = b[index];
        b[index] = b[k];
        b[k] = temp;
        // 构造L、U矩阵
        for (j = k; j < N; j++)
        {
            double temp = 0;
            for (int m = 0; m < k; m++)
            {
                temp = temp + A[k*N+m] * A[m*N+j];
            }
            A[k*N+j] = A[k*N+j] - temp;   //先构造U一行的向量
        }
        for (i = k + 1; i < N; i++)
        {
            double temp = 0;
            for (int m = 0; m < k; m++)
            {
                temp = temp + A[i*N+m] * A[m*N+k];
            }
            A[i*N+k] = (A[i*N+k] - temp) / A[k*N+k];  //再构造L一列的向量
        }
    }
    //求解LY = B
    Y[0] = b[0];
    for (i = 1; i < N; i++)
    {
        double temp = 0;
        for (int j = 0; j < i; j++)
        {
            temp = temp + A[i*N+j] * Y[j];
        }
        Y[i] = b[i] - temp;
    }
    //求解UX = Y
    X[N - 1] = Y[N - 1] / A[(N - 1)*N+N - 1];
    for (i = N - 2; i >= 0; i--)
    {
        double  temp = 0;
        for (int j = i + 1; j < N; j++)
        {
            temp = temp + A[i*N+j] * X[j];
        }
        X[i] = (Y[i] - temp) / A[i*N+i];
    }
    //打印X
    printf("结果X=\n");
    for(i=0;i<N;i++)
        printf("%8.4lf\n",X[i]);
}

int main(){
    int N=10;
    //系数矩阵
    double A[N * N]={1.0, 2.0,  3.0,  4.0, 5.0, 0.0, -1.0, -1.0, -1.0, -1.0,
                    2.0, 2.0,  3.0,  4.0, 6.0, 1.0,  0.0,  1.0,  0.0,  0.0,
                    3.0, 3.0,  3.0,  1.0, 5.0, 1.0, -1.0,  0.0,  0.0,  0.0,
                    4.0, 4.0,  1.0,  3.0, 1.0, 1.0,  0.0,  0.0,  0.0, -1.0,
                    5.0, 6.0,  5.0,  1.0, 3.0, 1.0,  0.0,  0.0,  1.0,  0.0,
                    0.0, 1.0,  1.0,  1.0, 1.0, 1.0,  2.0,  3.0,  4.0,  5.0,
                    -1.0, 0.0, -1.0,  0.0, 0.0, 2.0,  2.0,  3.0,  4.0,  6.0,
                    -1.0, 1.0,  0.0,  0.0, 0.0, 3.0,  3.0,  3.0,  1.0,  5.0,
                    -1.0, 0.0,  0.0,  0.0, 1.0, 4.0,  4.0,  1.0,  3.0,  1.0,
                    -1.0, 0.0,  0.0, -1.0, 0.0, 5.0,  6.0,  5.0,  1.0,  3.0};    
    //右端项
    double alpha[N] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};                   
    double beta[N];
    LU(A,alpha,beta,N);

}