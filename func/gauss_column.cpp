void gauss(double *A,double*b,double *X)
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
    
        //消元过程
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
    printf("结果X=\n\n");
    for(i=0;i<N;i++)
        printf("%.4lf\n",X[i]);
}