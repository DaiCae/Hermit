#include "math.h"
#include "time.h"
#include "stdlib.h"

// 随机生成Hermit对称矩阵
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
