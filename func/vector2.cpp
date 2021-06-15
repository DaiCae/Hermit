#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "time.h"

double eps = 1e-8;

//A为（A-tI）矩阵 b为v向量 x为结果特征向量
void gauss(int N,double *A,double*b,double *X)
{
    double m[N];            //选取列主元的比较器
    int i,j,k;
    float mik;//消元过程所用变量
    float S;//回代过程所用变量
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
            m[i] = A[i*N+k] - temp;
            if (m[index] < m[i])
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
    
        //消元过程    double A[9]={2,-1,3,4,2,5,2,0,2};
        for(i=k+1; i<N; i++)
        {
            mik=A[i*N+k]/A[k*N+k];
            for(j=k; j<N; j++)
            {
                A[i*N+j]=A[i*N+j]-mik*A[k*N+j];
            }
            b[i]=b[i]-mik*b[k]; 
        }
    } //消元结束
    //回代
    X[N-1]=b[N-1]/A[(N-1)*N+(N-1)];
    for(k=N-2; k>=0; k--)
    {
        S=b[k];
        for(j=k+1; j<N; j++)
        {
            S=S-A[k*N+j]*X[j];
        }
        X[k]=S/A[k*N+k];
    }


    // printf("结果b=\n\n");
    // for(i=0;i<N;i++)
    //     printf("%10.7lf\n",b[i]);

    printf("结果X=\n\n");
    for(i=0;i<N;i++)
        printf("%10.7lf\n",X[i]);
}



// int main(){
//     double A[9]={2,-1,3,4,2,5,2,0,2};
//     double b[3]={1,4,6};
//     // double b[3]={9,-1,-6};    
//     double x[3];
//     gauss(3,A,b,x);
// }


int main()
{
    int N = 10;

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

    double *B = new double[N*N];
    
    
    double eige = 16.63340;
    
    double *x=new double[N];
    double *v=new double[N];

    //向量初始化
    for(int i=0;i<N;i++)
        x[i]=1;

    //mol 初始化
    double mol_last = 0.0;
    double mol = 0.0;

    //求出（A-tI）
    for(int i=0;i<N;i++)
        A[i*N+i] = A[i*N+i] - eige;
    
    for(int i=0;i<N*N-1 ;i++)
        B[i]=A[i];

    int n=0;
    while(1){
        n++;
        //保存上次的mol
        mol_last = mol;

        //求出mol
        // mol=0.0;
        // for(int i=0;i<N;i++)
        //     mol += x[i] * x[i];
        // mol=sqrt(mol);
        mol=fabs(x[0]);
        for(int i=0;i<N;i++)
            if (mol< fabs(x[i])) mol=fabs(x[i]);

        //求出 v向量
        for(int i=0;i<N;i++)
            v[i] = x[i] / mol ;

        //恢复A矩阵
        for(int i=0;i<N*N-1; i++)
            A[i]=B[i];

        gauss(N,A,v,x);

        printf("\n%8.5lf  %8.5lf No.[%d]\n",mol,mol_last,n);
        if(fabs(mol - mol_last) < eps) break;

    }

}



