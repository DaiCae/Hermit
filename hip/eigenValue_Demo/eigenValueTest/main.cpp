#include <stdio.h>               // printf
#include <stdlib.h>              // EXIT_FAILURE
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_complex.h>
#include <math.h>
#include <sys/time.h>

#define LAPACK_COMPLEX_STRUCTURE
#define HAVE_LAPACK_CONFIG_H
#include "lapacke.h"

#define WARMUP
#define ACCURACY_CHECK

#define EPS (1.0e-7)

// 定义计时器
struct my_timer
{
    struct timeval start_time, end_time;
    double time_use; // us
    void start(){
		gettimeofday(&start_time, NULL);
    }
    void stop(){
		gettimeofday(&end_time, NULL);
		time_use = (end_time.tv_sec-start_time.tv_sec)*1.0e6 + end_time.tv_usec-start_time.tv_usec;
    }	
};

#define HIP_CHECK(stat)                                                    \
{                                                                          \
	if(stat != hipSuccess)                                                 \
	{                                                                      \
		std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
		exit(-1);                                                          \
	}                                                                      \
}		

double rand_double(int min,int max)
{
    return min + (int)max * rand()%100 / double((101));
}

typedef enum solverEigMode_ {
	solverEigMode_vector,
	solverEigMode_novector
}solverEigMode;

typedef enum solverFillMode_ {
	solverFillMode_lower,
	solverFillMode_upper
}solverFillMode;

// Hermite矩阵特征值求解函数调用声明
void solverDnZheevd(solverEigMode jobz,
                    solverFillMode uplo,
                    int n,
                    hipDoubleComplex *d_A,
                    int lda,
                    double *d_W, 
                    int *devInfo);
					
// Hermite矩阵特征值求解函数调用实现
void solverDnZheevd(solverEigMode jobz,
                    solverFillMode uplo,
                    int n,
                    hipDoubleComplex *d_A,
                    int lda,
                    double *d_W, 
                    int *devInfo)
{
	/*
	   function block
	*/
	*devInfo = 0;
	
	return;
}

int main(int argc, char** argv)
{
	if( argc != 5) {
		printf("eigenvalue test, Usage: ./eigValueTest eigMode fillMode matrixSize iter_num \n");
		exit(-1);
	}
	solverEigMode  jobz = solverEigMode(int(atoi(argv[1])));
	solverFillMode uplo = solverFillMode(int(atoi(argv[2])));
	int m   = int(atoi(argv[3]));
	int lda = m;
	int iter_num = int(atoi(argv[4]));
	
	printf("jobz:%d, uplo:%d, m:%d, iter_num:%d\n", jobz, uplo, m, iter_num);

	// 初始化Hermite矩阵A - host端
	hipDoubleComplex *A = (hipDoubleComplex*)malloc(lda*m*sizeof(hipDoubleComplex));
	assert(A != NULL);
	lapack_complex_double *lapack_A = (lapack_complex_double*)malloc(lda*m*sizeof(lapack_complex_double));
	assert(lapack_A != NULL);
	double *lapack_W = (double*)malloc(m * sizeof(double));
	assert(lapack_W != NULL);
	char lapack_jobz;
	char lapack_uplo;
/*		
	lapack_complex_double test[16] = {
	   { 3.40,  0.00}, {-2.36,  1.93}, {-4.68, -9.55}, { 5.37,  1.23},
           { 0.00,  0.00}, { 6.94,  0.00}, { 8.13,  1.47}, { 2.07,  5.78},
           { 0.00,  0.00}, { 0.00,  0.00}, {-2.14,  0.00}, { 4.68, -7.44},
           { 0.00,  0.00}, { 0.00,  0.00}, { 0.00,  0.00}, {-7.42,  0.00}
        };
*/	
	
	if(uplo == solverFillMode_lower){
		for(int i = 0; i < m; i++)	{
			for(int j = 0; j < m; j++){
				if(i > j){
					A[i*lda+j].x = rand_double(0, 10);
					A[i*lda+j].y = rand_double(0, 10);
					//A[i*lda+j].x = test[i*lda+j].real;
					//A[i*lda+j].y = test[i*lda+j].imag;
				}else if(i == j){	
					A[i*lda+j].x = rand_double(0, 10);
					A[i*lda+j].y = 0.0;
					//A[i*lda+j].x = test[i*lda+j].real;
					//A[i*lda+j].y = 0.0;
				}else{
					A[i*lda+j].x = 0.0;
					A[i*lda+j].y = 0.0;	
				}
				lapack_A[i*lda+j].real = A[i*lda+j].x;
				lapack_A[i*lda+j].imag = A[i*lda+j].y;	
			      //	printf("%f, %f    ", A[i*lda+j].x, A[i*lda+j].y);
			}
			//printf("\n");
		}
		lapack_uplo = 'L';
	}else if(uplo == solverFillMode_upper){
		for(int i = 0; i < m; i++)	{
			for(int j = 0; j < m; j++){
				if(i < j){
					A[i*lda+j].x = rand_double(0, 10);
					A[i*lda+j].y = rand_double(0, 10);	
					// A[i*lda+j].x = test[i*lda+j].real;
					// A[i*lda+j].y = test[i*lda+j].imag;
				}else if(i == j){	
					A[i*lda+j].x = rand_double(0, 10);
					A[i*lda+j].y = 0.0;	
					// A[i*lda+j].x = test[i*lda+j].real;
					// A[i*lda+j].y = 0.0;
				}else{
					A[i*lda+j].x = 0.0;
					A[i*lda+j].y = 0.0;
				}
				lapack_A[i*lda+j].real = A[i*lda+j].x;
				lapack_A[i*lda+j].imag = A[i*lda+j].y;
			     //	printf("%f, %f    ", A[i*lda+j].x, A[i*lda+j].y);
			}
			//printf("\n");
		}
		lapack_uplo = 'U';
	}else {
		printf("not valid parameter!\n");
		exit(-1);
	}
	
	
	if(jobz == solverEigMode_vector){
		lapack_jobz = 'V';
	}else if(jobz == solverEigMode_novector){
		lapack_jobz = 'N';
	}else{
		printf("not valid parameter!\n");
		exit(-1);
	}
	
	// 初始化Hermite矩阵A - device端
	hipDoubleComplex *d_A;
	HIP_CHECK( hipMalloc((void**) &d_A, lda * m * sizeof(hipDoubleComplex)) );
	HIP_CHECK( hipMemcpy(d_A, A, lda * m * sizeof(hipDoubleComplex), hipMemcpyHostToDevice) );
	
	// 分配device端内存
	double *d_W;
	HIP_CHECK( hipMalloc((void**) &d_W, m * sizeof(double)) );
	double *h_W = (double*)malloc(m*sizeof(double));
	assert(h_W != NULL);
	// host端指针
	int devInfo;
	
	// Hermite矩阵特征值求解函数调用
	
#ifdef WARMUP
	solverDnZheevd(jobz, uplo, m, d_A, lda, d_W, &devInfo);
	if(devInfo != 0){
		printf("eigvalue calculates failed!\n"); 
		exit(-1);
    }
#endif
	
	double sum_costs = 0.0;
	my_timer timer1;
	for(int index = 0; index < iter_num; index++){
		// 设备端数据恢复
		HIP_CHECK( hipMemcpy(d_A, A, lda * m * sizeof(hipDoubleComplex), hipMemcpyHostToDevice) );
		// 执行函数
		hipDeviceSynchronize();
		timer1.start();
		solverDnZheevd(jobz, uplo, m, d_A, lda, d_W, &devInfo);
		hipDeviceSynchronize();
		timer1.stop();
		// 耗时记录
		sum_costs += timer1.time_use;
	}	
	
	if(devInfo != 0){
		printf("eigvalue calculates failed!\n"); 
		exit(-1);
    }
	
	// Copy eigvalue计算结果 - device-to-host
	printf("eigvalue exec averages costs(ms): %.6f ms\n", sum_costs/iter_num/1000.0);
	HIP_CHECK( hipMemcpy(h_W, d_W, m * sizeof(double), hipMemcpyDeviceToHost) );
		
#ifdef ACCURACY_CHECK
	my_timer timer2;
        timer2.start();
	int info = LAPACKE_zheevd( LAPACK_ROW_MAJOR, lapack_jobz, lapack_uplo, m, lapack_A, lda, lapack_W);
	timer2.stop();
	printf("lapack: eigvalue exec averages costs(ms): %.6f ms\n", timer2.time_use/1000.0);
	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm failed to compute eigenvalues.\n" );
		exit( 1 );
	}
	// Accuracy check
	for(int index = 0; index < m; index++)
	{
		double abs_diff = fabs(lapack_W[index] - h_W[index]);
		
		// printf("index:%d, lapack_W:%.6f, h_W:%.6f\n", index, lapack_W[index], lapack_W[index]);
		
		if(abs_diff > EPS)
		{
		    printf("index:%d, lapack_W:%.6f, h_W:%.6f\n", index, lapack_W[index], h_W[index]);
                    printf("Failed verification,please check your code\n");
                    exit(-1);
		}
	}
	printf("check passed!\n");
#endif	

	// 存储资源释放
	free(A);
	free(lapack_A);
	free(h_W);
	free(lapack_W);
	
	hipFree(d_A);
	hipFree(d_W);
	
	return 0;
}

