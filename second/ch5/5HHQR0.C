#include "stdio.h"
#include "math.h"
#include "time.h"

int hhqr(double *a, int n, double *u, double *v, double eps, int jt)
// double a[], u[], v[], eps;
{
    int m, it, i, j, k, l, ii, jj, kk, ll;
    double b, c, w, g, xy, p, q, r, x, s, e, f, z, y;
    it = 0;
    m = n;
    while (m != 0)
    {
        l = m - 1;
        while ((l > 0) && (fabs(a[l * n + l - 1]) > eps * (fabs(a[(l - 1) * n + l - 1]) + fabs(a[l * n + l]))))
            l = l - 1;
        ii = (m - 1) * n + m - 1;
        jj = (m - 1) * n + m - 2;
        kk = (m - 2) * n + m - 1;
        ll = (m - 2) * n + m - 2;
        if (l == m - 1)
        {
            u[m - 1] = a[(m - 1) * n + m - 1];
            v[m - 1] = 0.0;
            m = m - 1;
            it = 0;
        }
        else if (l == m - 2)
        {
            b = -(a[ii] + a[ll]);
            c = a[ii] * a[ll] - a[jj] * a[kk];
            w = b * b - 4.0 * c;
            y = sqrt(fabs(w));
            if (w > 0.0)
            {
                xy = 1.0;
                if (b < 0.0)
                    xy = -1.0;
                u[m - 1] = (-b - xy * y) / 2.0;
                u[m - 2] = c / u[m - 1];
                v[m - 1] = 0.0;
                v[m - 2] = 0.0;
            }
            else
            {
                u[m - 1] = -b / 2.0;
                u[m - 2] = u[m - 1];
                v[m - 1] = y / 2.0;
                v[m - 2] = -v[m - 1];
            }
            m = m - 2;
            it = 0;
        }
        else
        {
            if (it >= jt)
            {
                printf("fail\n");
                return (-1);
            }
            it = it + 1;
            for (j = l + 2; j <= m - 1; j++)
                a[j * n + j - 2] = 0.0;
            for (j = l + 3; j <= m - 1; j++)
                a[j * n + j - 3] = 0.0;
            for (k = l; k <= m - 2; k++)
            {
                if (k != l)
                {
                    p = a[k * n + k - 1];
                    q = a[(k + 1) * n + k - 1];
                    r = 0.0;
                    if (k != m - 2)
                        r = a[(k + 2) * n + k - 1];
                }
                else
                {
                    x = a[ii] + a[ll];
                    y = a[ll] * a[ii] - a[kk] * a[jj];
                    ii = l * n + l;
                    jj = l * n + l + 1;
                    kk = (l + 1) * n + l;
                    ll = (l + 1) * n + l + 1;
                    p = a[ii] * (a[ii] - x) + a[jj] * a[kk] + y;
                    q = a[kk] * (a[ii] + a[ll] - x);
                    r = a[kk] * a[(l + 2) * n + l + 1];
                }
                if ((fabs(p) + fabs(q) + fabs(r)) != 0.0)
                {
                    xy = 1.0;
                    if (p < 0.0)
                        xy = -1.0;
                    s = xy * sqrt(p * p + q * q + r * r);
                    if (k != l)
                        a[k * n + k - 1] = -s;
                    e = -q / s;
                    f = -r / s;
                    x = -p / s;
                    y = -x - f * r / (p + s);
                    g = e * r / (p + s);
                    z = -x - e * q / (p + s);
                    for (j = k; j <= m - 1; j++)
                    {
                        ii = k * n + j;
                        jj = (k + 1) * n + j;
                        p = x * a[ii] + e * a[jj];
                        q = e * a[ii] + y * a[jj];
                        r = f * a[ii] + g * a[jj];
                        if (k != m - 2)
                        {
                            kk = (k + 2) * n + j;
                            p = p + f * a[kk];
                            q = q + g * a[kk];
                            r = r + z * a[kk];
                            a[kk] = r;
                        }
                        a[jj] = q;
                        a[ii] = p;
                    }
                    j = k + 3;
                    if (j >= m - 1)
                        j = m - 1;
                    for (i = l; i <= j; i++)
                    {
                        ii = i * n + k;
                        jj = i * n + k + 1;
                        p = x * a[ii] + e * a[jj];
                        q = e * a[ii] + y * a[jj];
                        r = f * a[ii] + g * a[jj];
                        if (k != m - 2)
                        {
                            kk = i * n + k + 2;
                            p = p + f * a[kk];
                            q = q + g * a[kk];
                            r = r + z * a[kk];
                            a[kk] = r;
                        }
                        a[jj] = q;
                        a[ii] = p;
                    }
                }
            }
        }
    }
    return (1);
}

void hhbg(double *a, int n)
{
    int i, j, k, u, v;
    double d, t;
    for (k = 1; k <= n - 2; k++)
    {
        d = 0.0;
        for (j = k; j <= n - 1; j++)
        {
            u = j * n + k - 1;
            t = a[u];
            if (fabs(t) > fabs(d))
            {
                d = t;
                i = j;
            }
        }
        if (fabs(d) + 1.0 != 1.0)
        {
            if (i != k)
            {
                for (j = k - 1; j <= n - 1; j++)
                {
                    u = i * n + j;
                    v = k * n + j;
                    t = a[u];
                    a[u] = a[v];
                    a[v] = t;
                }
                for (j = 0; j <= n - 1; j++)
                {
                    u = j * n + i;
                    v = j * n + k;
                    t = a[u];
                    a[u] = a[v];
                    a[v] = t;
                }
            }
            for (i = k + 1; i <= n - 1; i++)
            {
                u = i * n + k - 1;
                t = a[u] / d;
                a[u] = 0.0;
                for (j = k; j <= n - 1; j++)
                {
                    v = i * n + j;
                    a[v] = a[v] - t * a[k * n + j];
                }
                for (j = 0; j <= n - 1; j++)
                {
                    v = j * n + k;
                    a[v] = a[v] + t * a[j * n + i];
                }
            }
        }
    }
    return;
}

void sort(double A[],int N)
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
            }
        }
    }
}

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
                A[i * n + j] = rand()%100 / double((101)) *10;
                B[i * n + j] = rand()%100 / double((101)) *10;
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

int cycle(){
    
    int i, j, jt = 100000000;
    int N = 512;
    double *a = new double[N * N];
    symmat(a,N);
    double eps = 1e-13;
    double u[N], v[N];
    // double a[5*5] = {1.0, 6.0, -3.0, -1.0, 7.0,
    //                         8.0, -15.0, 18.0, 5.0, 4.0,
    //                         -2.0, 11.0, 9.0, 15.0, 20.0,
    //                         -13.0, 2.0, 21.0, 30.0, -6.0,
    //                         17.0, 22.0, -5.0, 3.0, 6.0};

    // double a[100] = {
    //     1.0, 2.0, 3.0, 4.0, 5.0, 0.0, -1.0, -1.0, -1.0, -1.0,
    //     2.0, 2.0, 3.0, 4.0, 6.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    //     3.0, 3.0, 3.0, 1.0, 5.0, 1.0, -1.0, 0.0, 0.0, 0.0,
    //     4.0, 4.0, 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
    //     5.0, 6.0, 5.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    //     0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    //     -1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0, 6.0,
    //     -1.0, 1.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 5.0,
    //     -1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 1.0, 3.0, 1.0,
    //     -1.0, 0.0, 0.0, -1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 3.0
    // };


    hhbg(a, N);
    // printf("MAT H IS:\n");
    // for (i = 0; i < N; i++)
    // {
    //     for (j = 0; j < N; j++)
    //         printf("%13.7lf ", a[i * N + j]);
    //     printf("\n");
    // }
    printf("\n");
    i = hhqr(a, N, u, v, eps, jt);
    sort(u,N);
    int err=0;
    if (i > 0)
        for (i = 0; i < N; i+=2)
        {
            printf("%13.8lf  |", u[i]);
            printf("%13.8lf  |", u[i+1]);
            double diff=fabs(u[i+1]-u[i]);
            printf("  EPS:%10.7lf \n",fabs(u[i+1]-u[i]));
            if(diff>1e-7) err++;
        }
    printf("\n");
    printf("Err:%d\n",err);
    if(err>0) err=1;
    return err;
}

int main()
{
    int eee=0;
    for(int i=0;i<100;i++){
        eee+=cycle();
    }

    printf("\nTotal Err:%d\n",eee);
}



