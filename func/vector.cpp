#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define N 10
//A为（A-tI）矩阵   b为vector   X为结果矩阵  n为矩阵的阶数
// void gauss(double *A,double*b,double *X,int n)
// {
//     int i,j,k;
//     float mik;      //消元过程所用变量
//     float S;        //回代过程所用变量
//                     //消元

//     for(k=0; k<N-1; k++)
//     {
//         for(i=k+1; i<N; i++)
//         {
//             mik=A[i*N+k]/A[k*N+k];
//             for(j=k; j<N; j++)
//             {
//                 A[i*N+j]=A[i*N+j]-mik*A[k*N+j];
//             }
//             b[i]=b[i]-mik*b[k];
//         }
//     } 
    
    
//     X[N-1]=b[N-1]/A[(N-1)*N+(N-1)];
//     for(k=N-2; k>=0; k--)
//     {
//         S=b[k];
//         for(j=k+1; j<N; j++)
//         {
//             S=S-A[k*N+j]*X[j];
//         }
//         X[k]=S/A[k*N+k];
//     }
//     for (i = 0; i < N; i++){
//             printf("%10lf ",X[i]);
//         }
// }
void LU(double *A, double*b, double *X )
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
    // //打印X
    // printf("结果X=\n");
    // for(i=0;i<N;i++)
    //     printf("%8.4lf\n",X[i]);
}

int main(){
    int i,j,k,l,flag;
    double X[N],v[N];           //定义  结果矩阵X和中间矩阵V
    double mac1,t,mac2;
    double eps=1e-8;
    // double A[N*N]={2,-1,3,4,2,5,2,0,2};
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
    mac1=1;
    mac2=0;
    t=3.3730;
    // max1=0;
    // max2=1;
    double b[N*N];              //定义 b为参与do循环的矩阵 
    void gauss();
    flag=1;

    //初始化数组X
    for (i = 0; i <N; i++)       
        X[i]=1;

    //求出(A-lambda * E)  
    for (k = 0; k <N; k++)         
        A[k*N+k]=A[k*N+k]-t;

    while (fabs(mac2-mac1) >= eps){
        mac2=mac1;
        // printf("111\n");

        //复制初始矩阵A
        for(i=0;i<N;i++)          
        {
            for (j = 0; j <N; j++)
            {
                b[i*N+j]=A[i*N+j];
            }
        }

        // max2=max1;
        // max1=X[0];
        
        //求最大值
        mac1=0;
        for(i=1;i<N;i++)                
            mac1+=(X[i]*X[i]);
        mac1=sqrt(mac1);
        printf("%10lf",mac1);
        // if(flag=0)
        //     mac2=mac1;
        // else
        //     continue;
        //update vector v;
        for (i = 0; i < N; i++){
            v[i]=X[i]/mac1;
            //printf("%10lf ",v[i]);
        }
        // printf("\n");


        LU(b,v,X);

        // for (i = 0; i < N; i++){
        //     printf("%10lf ",X[i]);
        // }
        // printf("\n");
    }

    printf("结果X=\n\n");
    for(i=0;i<N;i++)
        printf("%.8lf\n",X[i]);
}
