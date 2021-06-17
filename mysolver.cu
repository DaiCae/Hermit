#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <math.h>
#include <sys/time.h>

void show(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%12.8lf,", A[i * N + j]);
        }
        printf("\n");
    }
}

void symmat(double H[], int N)
{
    int n = N / 2;
    double *A = new double[n * n];
    double *B = new double[n * n];

    for (int i = 0; i < n; i++)
    {
        srand((unsigned)time(NULL));
        for (int j = 0; j < n; j++)
        {
            if (j >= i)
            {
                A[i * n + j] = (rand() % 20 - 10);
                B[i * n + j] = (rand() % 20 - 10);
            }
            else
            {
                A[i * n + j] = A[j * n + i];
                B[i * n + j] = -1 * B[j * n + i];
            }
        }
        B[i * n + i] = 0;
    }

    // show(A,n);
    // printf("\n");
    // show(B,n);
    // printf("\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if ((i >= n && j >= n) || (i < n && j < n))
                H[i * N + j] = A[i % n * n + j % n];
            else if (i < n)
                H[i * N + j] = -1 * B[i % n * n + j % n];
            else
                H[i * N + j] = B[i % n * n + j % n];
        }
    }
}

// 对特征值和特征向量排序
void sort_vector(double A[], int N, double *q)
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
                for (int k = 0; k < N; k++)
                {
                    temp = q[k * N + j];
                    q[k * N + j] = q[k * N + j + 1];
                    q[k * N + j + 1] = temp;
                }
            }
        }
    }
}

// Householder（带特征向量）
int Householder_vector(int n, double A[], double b[], double c[], double *Q)
{
    double mol, q, value, K;
    double *alpha = new double[n];
    // 初始化Q为I矩阵
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Q[i * n + j] = 0.0;
        }
        Q[i * n + i] = 1;
    }

    for (int i = n - 1; i > 1; i--)
    {
        q = 0.0;
        //初始化alpha向量为0
        alpha[i] = 0.0;
        for (int j = 0; j < i; j++)
        {
            alpha[j] = A[i * n + j];
            q += alpha[j] * alpha[j];
        }
        // 判断q的值是否小于精度
        if (q + 1.0 == 1.0)
        {
            continue;
        }
        mol = sqrt(q);
        if (alpha[i - 1] > 0.0)
            mol = -mol;
        q -= alpha[i - 1] * mol;
        alpha[i - 1] = alpha[i - 1] - mol;


        // 求Q(i+1)矩阵（矩阵减矩阵）
        for (int j = 0; j <= n-1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i; k++)
                value += Q[j * n + k] * alpha[k] / q;
            b[j] = value;
        }

        // printf("\n");
        // for(int l=0;l<n;l++)
        //     printf("%12.8lf,", b[l]);
        // printf("\n\n");

        for (int j = 0; j <= n-1; j++)
        {
            for (int k = 0; k < i; k++)
            {
                Q[j * n + k] = Q[j * n + k] - b[j] * alpha[k];
            }
        }




       // 求A(i+1)矩阵
        for (int j = 0; j < i + 1; j++)
        {
            value = 0.0;
            for (int k = 0; k < i + 1; k++)
                value += A[j * n + k] * alpha[k] / q;
            b[j] = value;
        }
        K = 0;
        for (int j = 0; j < i + 1; j++)
        {
            K += alpha[j] * b[j] / (2 * q);
        }

        for (int j = 0; j < i + 1; j++)
        {
            c[j] = b[j] - K * alpha[j];
        }
        for (int j = 0; j < i + 1; j++)
        {
            for (int k = 0; k < i + 1; k++)
            {
                A[j * n + k] = A[j * n + k] - alpha[j] * c[k] - c[j] * alpha[k];
            }
        }

        break;
    }
    
    // 抽出主、次对角线元素
    for (int i = 0; i < n - 1; i++)
    {
        b[i] = A[i * n + i];
        c[i] = A[i * n + i + 1];
    }
    b[n - 1] = A[n * n - 1];
    c[n - 1] = 0;
    delete[] alpha;
    return 1;
}

//QR（带特征向量）
int QR_vector(int N, double *b, double *c, double *q, double eps){
    // N 为矩阵阶数
    // b[] 为主对角线元素
    // c[] 为次对角线元素
    
    double h, p, r;
    double f = 0.0;
    for (int k = 0; k <= N - 1; k++){
        do{
            double g = b[k];
            // (b[k+1] - b[k]) / (2 * c[k])
            p = (b[k + 1] - g) / (2.0 * c[k]);
            r = sqrt(p * p + 1.0);
            if (p >= 0.0)
                b[k] = c[k] / (p + r);
            else
                b[k] = c[k] / (p - r);
            h = g - b[k];

            for (int i = k + 1; i <= N - 1; i++)
                b[i] = b[i] - h;
            f = f + h;

            p = b[N-1];

            double e = 1.0;
            double s = 0.0;
            for (int i = N - 2; i >= k; i--)
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
                    //=======================
                    for (int j = 0; j <= N - 1; j++)
                    {
                        int u = j * N + i + 1;
                        int v = u - 1;
                        h = q[u];
                        q[u] = s * q[v] + e * h;
                        q[v] = e * q[v] - s * h;
                    }
                    //=======================
            }
            c[k] = s * p;
            b[k] = e * p;
        } while (fabs(c[k]) > eps);
        
        b[k] = b[k] + f;
    }
    //=======================
    // for (int i = 0; i <= N - 1; i++)
    // {
    //     int k = i;
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
    //             int u = j * N + i;
    //             int v = j * N + k;
    //             p = q[j * N + i];
    //             q[j * N + i] = q[j * N + k];
    //             q[j * N + k] = p;
    //         }
    //     }
    // }
    //=======================
    return (1);
}

//求解特征值和特征向量
// int mysolver_cpu_vector(int N, double *Dev_A, double *Dev_W, double *d_A)
int mysolver_cpu_vector(int N, double *Dev_A, double *Dev_W, double *Q)
{
    //原矩阵
    double *a = new double[N * N];
    double *Qalpha = new double[N * N];
    //主次对角线
    double *b = new double[N];
    double *c = new double[N];
    //
    double *y_l = new double[N * N];
    double eps = 1e-12;

    cudaMemcpy(a, Dev_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    // printf("\n");
    // show(a,N);

    time_t start, end;
    start = clock(); 
    Householder_vector(N, a, b, c, Qalpha);
    end = clock();
    printf("HS CPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // for (int i = 0; i < N * N; i++)
    // {    
    //     Q[i] = Qalpha[i];
    // }

    start = clock();
    QR_vector(N, b, c, Qalpha, eps);
    end = clock();
    printf("QR CPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    sort_vector(b, N, Qalpha);


    //抽出一半特征值
    for (int i = 0; i < N / 2; i++)
        Dev_W[i] = b[i * 2];

    // for (int i = 0; i < N / 2; i++)
    // {    for(int j=0; j<N;j++){
    //         Q[j * N + i] = Qalpha[j * N + i * 2];
    //     }
    // }

    for (int i = 0; i < N * N; i++)
    {    
        Q[i] = Qalpha[i];
    }

    return 1;
}



__global__ void Householder_step_0(double *Q , int N){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    //关闭多余线程
    if (idx >= N) return;
    // if (idy >= N) return;

    //初始化Q矩阵为 I 矩阵
    for (int j = 0; j < N; j++)
        Q[j * N + idx] = 0.0;
    Q[idx * N + idx] = 1;

}

__global__ void Householder_step_1(double *A ,double *alpha, double *beta, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N) return;

    if (idx < i)
    {
        alpha[idx] = A[i * N + idx];
        beta[idx] = alpha[idx] * alpha[idx];
    }else{
        alpha[idx] = 0.0;
        beta[idx] = 0.0;
    }

}

__global__ void Householder_step_2(double *Q ,double *A, double *alpha, double *beta, double *b, double q, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N) return;
    //======================================
    // 求Q(i+1)矩阵（矩阵减矩阵）
    double value =0.0;
    for (int k = 0; k < i; k++)
        value += Q[idx * N + k] * alpha[k] /q;
    b[idx]=value;
    __syncthreads();

    // if(idx==0){
    //     printf("\n");
    //     for(int l=0;l<N;l++)
    //         printf("%12.8lf,", b[l]);
    //     printf("\n\n");
    // }

    for (int k = 0; k < i; k++)
        Q[idx * N + k] = Q[idx * N + k] - b[idx] * alpha[k];
    __syncthreads();

    //idx < i+1
    if(idx >= i + 1) return;

    //======================================
    // 求A(i+1)矩阵
    value = 0.0;
    for (int k = 0; k < i + 1; k++)
        value += A[idx * N + k] * alpha[k] / q;
    b[idx] = value;

    // 求K
    beta[idx] = alpha[idx] * b[idx] / (2 * q);
}

__global__ void Householder_step_3(double *A ,double *alpha, double *b, double *c, double K, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    //idx < i+1
    if(idx >= i + 1) return;

    //======================================
    // 求A(i+1)矩阵
    c[idx] = b[idx] - K * alpha[idx];
    __syncthreads();
}

__global__ void Householder_step_4(double *A ,double *alpha, double *b, double *c, double K, int N, int i){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    //idx < i+1
    if(idx >= i + 1) return;

    for (int k = 0; k < i + 1; k++)
    {
        A[idx * N + k] = A[idx * N + k] - alpha[idx] * c[k] - c[idx] * alpha[k];
    }
}

__global__ void Householder_step_5(double *A ,double *b, double *c, int N){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //关闭多余线程
    if (idx >= N-1) return;

    // 抽出主、次对角线元素
    b[idx] = A[idx * N + idx];
    c[idx] = A[idx * N + idx + 1];

    if(idx==0)
    {
        b[N-1]=A[N * N - 1];
        c[N-1]=0.0;
    }
}

int mysolver_vector(int N, double *Dev_A, double *Dev_W, double *Q)
{

    time_t start, end;
    start = clock();
    //主次对角线 b c 向量指针
    double *Dev_b;
    double *Dev_c;

    //临时向量
    double *Dev_alpha;
    double *Dev_beta;

    //乘积矩阵 在设备端指针
    double *Dev_Q;

    //分配空间 主次对角线
    cudaMalloc((void **)&Dev_b, N * sizeof(double));
    cudaMalloc((void **)&Dev_c, N * sizeof(double));

    cudaMalloc((void **)&Dev_alpha, N * sizeof(double));
    cudaMalloc((void **)&Dev_beta, N * sizeof(double));

    //分配空间 乘积矩阵
    cudaMalloc((void **)&Dev_Q, N * N * sizeof(double));

    //定义N维向量的grid
    dim3 block(1024);
    dim3 grid((N-1)/block.x + 1,1);
    
    printf("block size: %d | grid size: %d\n",block.x,grid.x);
    
    // //计算需要的最小块数的长度取整
    // int length = (N-1)/32 + 1;

    // //定义N维矩阵的grid
    // dim3 Block(32,32);
    // dim3 Grid(length, length);

    //在CPU上申请求和用的缓存数组
    double *H_Sum = new double[N];
    double sum,eps=1e-12;

    //Householder 循环分步调用核函数
    // hipLaunchKernelGGL(Householder_step_0,grid,block,0,0, Dev_Q, N);
    Householder_step_0<<<grid,block>>>(Dev_Q, N);//调用核函数
    for(int i = N-1; i>1 ;i--)
    {
        //=========================================
        //  初始化alpha向量 合并规约求alpha 的 mol
        //=========================================
        // hipLaunchKernelGGL(Householder_step_1,grid,block,0,0, Dev_A, Dev_alpha, Dev_beta, N, i);
        Householder_step_1<<<grid,block>>>(Dev_A, Dev_alpha, Dev_beta, N, i);//调用核函数
        cudaDeviceSynchronize();
        cudaMemcpy(H_Sum, Dev_beta, N * sizeof(double), cudaMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++){
            sum += H_Sum[k];    //sum 为 q
        }
        //检查是否为0
        if( (sum-eps)<0.0) continue;
        double mol =sqrtl(sum);
        
        //将alpha[i-1]拷贝到cpu端
        cudaMemcpy(H_Sum, Dev_alpha + i - 1, 1 * sizeof(double), cudaMemcpyDeviceToHost);
        if(H_Sum[0] >0.0)
            mol = -mol;
        sum -= H_Sum[0] * mol;
        H_Sum[0] = H_Sum[0] - mol;
        //写回alpha[i-1]
        cudaMemcpy(Dev_alpha + i - 1, H_Sum, 1 * sizeof(double), cudaMemcpyHostToDevice);
        
        //=========================================
        // 求Q(i+1)矩阵（矩阵减矩阵）
        // 求出K
        //=========================================
        // hipLaunchKernelGGL(Householder_step_2,grid,block,0,0, Dev_Q, Dev_A, Dev_alpha, Dev_beta, Dev_b, sum, N, i);
        Householder_step_2<<<grid,block>>>(Dev_Q, Dev_A, Dev_alpha, Dev_beta, Dev_b, sum, N, i);//调用核函数
        cudaDeviceSynchronize();
        cudaMemcpy(H_Sum, Dev_beta, N * sizeof(double), cudaMemcpyDeviceToHost);
        sum = 0;
        for (int k = 0; k < N; k++)
            sum += H_Sum[k];    //sum 为 K

        //=========================================
        // 求A(i+1)矩阵
        //=========================================
        // hipLaunchKernelGGL(Householder_step_3,grid,block,0,0, Dev_A, Dev_alpha, Dev_b, Dev_c, sum, N, i);
        Householder_step_3<<<grid,block>>>(Dev_A, Dev_alpha, Dev_b, Dev_c, sum, N, i);//调用核函数
        Householder_step_4<<<grid,block>>>(Dev_A, Dev_alpha, Dev_b, Dev_c, sum, N, i);//调用核函数
        cudaDeviceSynchronize();

        break;
    }

    //=========================================
    // 抽出主、次对角线元素
    //=========================================
    // hipLaunchKernelGGL(Householder_step_4,grid,block,0,0, Dev_A, Dev_b, Dev_c, N);
    Householder_step_5<<<grid,block>>>(Dev_A, Dev_b, Dev_c, N);//调用核函数
    cudaDeviceSynchronize();

    cudaFree(Dev_alpha);
    cudaFree(Dev_beta);

    double *H_c = new double[N];
    double *H_Q = new double[N * N];

    cudaMemcpy(H_Sum, Dev_b, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(H_c, Dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(H_Q, Dev_Q, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    end = clock();
    printf("HS GPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // for (int i = 0; i < N * N; i++)
    // {    
    //     Q[i] = H_Q[i];
    // }

    start = clock();
    QR_vector(N, H_Sum, H_c, H_Q, eps);
    end = clock();
    printf("QR CPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);


    sort_vector(H_Sum, N, H_Q);

    //抽出一半特征值
    for (int i = 0; i < N / 2; i++)
        Dev_W[i] = H_Sum[i * 2];

    // double *y_l = new double[N * N];

    //抽出一半特征向量
    // for (int i = 0; i < N / 2; i++)
    // {    for(int j=0; j<N;j++){
    //         Q[j * N + i] = H_Q[j * N + i * 2];
    //     }
    // }

    for (int i = 0; i < N * N; i++)
    {    
        Q[i] = H_Q[i];
    }



    // //结果拷回
    // cudaMemcpy(Dev_W, H_c, N / 2 * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_A, A_yl, N * N / 2 * sizeof(double), cudaMemcpyHostToDevice);

    cudaFree(Dev_b);
    cudaFree(Dev_c);
    return 1;
}

int main()
{
    time_t start, end;
	printf("分配内存空间..\n");

    int N = 2560;
    double *A = new double[N * N];
    symmat(A, N);
    // show(A,N);

    double *Dev_W_0 = new double[N];
    double *Dev_W_1 = new double[N];

    double *Dev_Q_0 = new double[N * N];
    double *Dev_Q_1 = new double[N * N];

    //设备端内存分配
    double *Dev_A = new double[N * N];
	cudaMalloc((void**)&Dev_A, N * N * sizeof(double));



	printf("内存拷贝..\n\n");
	//数据拷贝，主机到设备
	cudaMemcpy(Dev_A, A, N * N * sizeof(double),cudaMemcpyHostToDevice);
    
    printf("========================================================\n");
    printf("CPU is calculating\n");
    start = clock();
    mysolver_cpu_vector(N, Dev_A, Dev_W_0, Dev_Q_0);
    end = clock();
    printf("Total CPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("========================================================\n");
    printf("GPU is calculating\n");
    start = clock();
    mysolver_vector(N, Dev_A, Dev_W_1, Dev_Q_1);
    end = clock();
    printf("Total GPU time=%lf (ms)\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("========================================================\n");
    printf("Checking eigvalue\n"); 
    int err=0;
    for(int k=0;k<N/2;k++){
        double diff=fabs(Dev_W_0[k]-Dev_W_1[k]);
        if (diff<=1e-8){
            continue;
            //printf("No[%2d]: %17.8lf || %17.8lf || %12.7lf \n", k, Dev_W_0[k], Dev_W_1[k],diff);
        }
        else{
            printf("No[%2d]: %17.8lf || %17.8lf || %12.7lf || E!\n", k, Dev_W_0[k], Dev_W_1[k],diff);
            err++;
        }
    }
    if(err==0)
        printf("Checking Pass!\n");
    else
        printf("Checking faled!\n");

    printf("========================================================\n");
    printf("Checking eigvector\n");
    err=0;
    for(int k=0;k<N*N;k++){
        double diff=fabs(Dev_Q_0[k]-Dev_Q_1[k]);
        if (diff<=1e-8){
            continue;
            //printf("No[%2d]: %17.8lf || %17.8lf || %12.7lf \n", k, Dev_W_0[k], Dev_W_1[k],diff);
        }
        else{
            printf("No[%2d]: %17.8lf || %17.8lf || %12.7lf || E!\n", k, Dev_Q_0[k], Dev_Q_1[k],diff);
            err++;
        }
    }
    if(err==0)
        printf("Checking pass!\n");
    else
        printf("Checking faled!\n");
    
    // show(Dev_Q_0,N);
    // printf("========================================================\n");
    // show(Dev_Q_1,N);
    
	//释放GPU内存
	// cudaFree(Dev_A);
	
	//释放CPU内存
    // free(A);
    // free(B);
	// free(C);

	return 0;
}







