__global__ void MatrixMulkernel(double *A,double *H,double*B,int n)
{
    int  tx=threadIdx.x + blockIdx.x * blockDim.x;
    int  ty=threadIdx.y + blockIdx.y * blockDim.y;
    double pvalue=0.0;
    double value=0.0;
    for(int k=0;k<n;k++)
    {
        double Md=H[ty*n+k];
        double Nd=A[k*n+tx];
        pvalue+=Md*Nd;
    }
    B[ty*n+tx]=pvalue;
    for(int j=0;j<n;j++)
    {
        double Mds=B[ty*n+j];
        double Nds=H[j*n+tx];
        value+=Mds*Nds;
    }
    A[ty*n+tx]=value;
}