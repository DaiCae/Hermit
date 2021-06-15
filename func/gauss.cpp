#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define N 3
//A为（A-tI）矩阵   b为vector   X为结果vector  n为矩阵的阶数
void gauss(double *A,double*b,double *X,int n)
{
    int i,j,k;
    float mik;      //消元过程所用变量
    float S;        //回代过程所用变量
                    //消元

    for(k=0; k<N-1; k++)
    {
        for(i=k+1; i<N; i++)
        {
            mik=A[i*N+k]/A[k*N+k];
            for(j=k; j<N; j++)
            {
                A[i*N+j]=A[i*N+j]-mik*A[k*N+j];
            }
            b[i]=b[i]-mik*b[k];
        }
    } 
    
    
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
}