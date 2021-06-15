#include "stdio.h"
#include "math.h"
#include "time.h"
#include "stdlib.h"

//Householder
void Householder_old(double *a,int n, double* q, double* b, double* c)
{
    int i, j, k, u, v;
    double h, f, g, h2;
    
    //将a拷贝到q
    for (i = 0; i < n * n ; i++)
        q[i] = a[i];

    for (i = n - 1; i >= 1; i--)
    {
        double h = 0.0;
        if (i > 1)
            for (k = 0; k <= i - 1; k++)
            {
                u = i * n + k;
                h = h + q[u] * q[u];
            }
        if (h + 1.0 == 1.0)
        {
            c[i] = 0.0;
            if (i == 1)
                c[i] = q[i * n + i - 1];
            b[i] = 0.0;
        }
        else
        {
            c[i] = sqrt(h);
            u = i * n + i - 1;
            if (q[u] > 0.0)
                c[i] = -c[i];
            h = h - q[u] * c[i];
            q[u] = q[u] - c[i];
            f = 0.0;
            for (j = 0; j <= i - 1; j++)
            {
                q[j * n + i] = q[i * n + j] / h;
                g = 0.0;
                for (k = 0; k <= j; k++)
                    g = g + q[j * n + k] * q[i * n + k];
                if (j + 1 <= i - 1)
                    for (k = j + 1; k <= i - 1; k++)
                        g = g + q[k * n + j] * q[i * n + k];
                c[j] = g / h;
                f = f + g * q[j * n + i];
            }
            h2 = f / (h + h);
            for (j = 0; j <= i - 1; j++)
            {
                f = q[i * n + j];
                g = c[j] - h2 * f;
                c[j] = g;
                for (k = 0; k <= j; k++)
                {
                    u = j * n + k;
                    q[u] = q[u] - f * c[k] - g * q[i * n + k];
                }
            }
            b[i] = h;
        }
    }
    for (i = 0; i <= n - 2; i++)
        c[i] = c[i + 1];
    c[n - 1] = 0.0;
    b[0] = 0.0;
    for (i = 0; i <= n - 1; i++)
    {
        if ((b[i] != 0.0) && (i - 1 >= 0))
            for (j = 0; j <= i - 1; j++)
            {
                g = 0.0;
                for (k = 0; k <= i - 1; k++)
                    g = g + q[i * n + k] * q[k * n + j];
                for (k = 0; k <= i - 1; k++)
                {
                    u = k * n + j;
                    q[u] = q[u] - g * q[k * n + i];
                }
            }
        u = i * n + i;
        b[i] = q[u];
        q[u] = 1.0;
        if (i - 1 >= 0)
            for (j = 0; j <= i - 1; j++)
            {
                q[i * n + j] = 0.0;
                q[j * n + i] = 0.0;
            }
    }
    return;
}

void Householder(double *a, int n, double* q, double* b, double* c)
{
    int i, j, k, u, v;
    double f, g, h2;
    for (i = n - 1; i >= 1; i--)
    {
        double h = 0.0;
        if (i > 1)
            for (k = 0; k <= i - 1; k++)
            {
                u = i * n + k;
                h = h + q[u] * q[u];
            }
        if (h + 1.0 == 1.0)
        {
            c[i] = 0.0;
            if (i == 1)
                c[i] = q[i * n + i - 1];
            b[i] = 0.0;
        }
        else
        {
            c[i] = sqrt(h);
            u = i * n + i - 1;
            if (q[u] > 0.0)
                c[i] = -c[i];
            h = h - q[u] * c[i];
            q[u] = q[u] - c[i];
            f = 0.0;
            for (j = 0; j <= i - 1; j++)
            {
                q[j * n + i] = q[i * n + j] / h;
                g = 0.0;
                for (k = 0; k <= j; k++)
                    g = g + q[j * n + k] * q[i * n + k];
                if (j + 1 <= i - 1)
                    for (k = j + 1; k <= i - 1; k++)
                        g = g + q[k * n + j] * q[i * n + k];
                c[j] = g / h;
                f = f + g * q[j * n + i];
            }
            h2 = f / (h + h);
            for (j = 0; j <= i - 1; j++)
            {
                f = q[i * n + j];
                g = c[j] - h2 * f;
                c[j] = g;
                for (k = 0; k <= j; k++)
                {
                    u = j * n + k;
                    q[u] = q[u] - f * c[k] - g * q[i * n + k];
                }
            }
            b[i] = q[i * n + i];
        }
    }
    for (i = 0; i <= n - 2; i++)
        c[i] = c[i + 1];
    c[n - 1] = 0.0;
    b[0] = q[0];
    b[1] = q[n + 1];
}

//QR
int QR(int N, double *b, double *c, double *q, double eps, int limit)
{
    int k, m, it, u, v;
    double d, f, h, g, p, r, e, s;
    c[N - 1] = 0.0;
    d = 0.0;
    f = 0.0;
    for (int j = 0; j <= N - 1; j++)
    {
        it = 0;
        h = eps * (fabs(b[j]) + fabs(c[j]));
        if (h > d)
            d = h;
        m = j;
        while ((m <= N - 1) && (fabs(c[m]) > d))
            m = m + 1;
        if (m != j)
        {
            do
            {
                if (it == limit)
                {
                    printf("fail\n");
                    return (-1);
                }
                it = it + 1;
                g = b[j];
                p = (b[j + 1] - g) / (2.0 * c[j]);
                r = sqrt(p * p + 1.0);
                if (p >= 0.0)
                    b[j] = c[j] / (p + r);
                else
                    b[j] = c[j] / (p - r);
                h = g - b[j];
                for (int i = j + 1; i <= N - 1; i++)
                    b[i] = b[i] - h;
                f = f + h;
                p = b[m];
                e = 1.0;
                s = 0.0;
                for (int i = m - 1; i >= j; i--)
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

                    // for (k = 0; k <= N - 1; k++)
                    // {
                    //     u = k * N + i + 1;
                    //     v = u - 1;
                    //     h = q[u];
                    //     q[u] = s * q[v] + e * h;
                    //     q[v] = e * q[v] - s * h;
                    // }
                }
                c[j] = s * p;
                b[j] = e * p;
            } while (fabs(c[j]) > d);
        }
        b[j] = b[j] + f;
    }
    // for (int i = 0; i <= N - 1; i++)
    // {
    //     k = i;
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
    //             u = j * N + i;
    //             v = j * N + k;
    //             p = q[j * N + i];
    //             q[j * N + i] = q[j * N + k];
    //             q[j * N + k] = p;
    //         }
    //     }
    // }
    return (1);
}

//排序
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


int mysolver(int N, double *a, double *b)
{
    int l = 1000000;
    double *q = new double[N * N];
    double *b = new double[N];
    double *c = new double[N];

    double eps = 1e-7;
    Householder(a, N, q, b, c);
    QR(N, b, c, q, eps, l);
    sort(b,N);

    for(int i=0;i<N/2;i++)
        c[i]=b[i*2];

    return 1;
}
