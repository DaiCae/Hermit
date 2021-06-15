#include <stdio.h>               // printf
#include <stdlib.h>   
#include <math.h>
double eps =1e-8;

void Householder_0(double *a, int n, double* q, double* b, double* c)
{
    int k, j, i, u, v;
    for (k = n - 1; k >= 1; k--)
    {
        double h = 0.0;
        if (k > 1)
            for (i = 0; i <= k - 1; i++)
            {
                u = k * n + i;
                h = h + a[u] * a[u];
            }
        if (h + 1.0 == 1.0)
        {
            c[k] = 0.0;
            if (k == 1)
                c[k] = a[k * n + k - 1];
            b[k] = 0.0;
        }
        else
        {
            c[k] = sqrt(h);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                c[k] = -c[k];
            h = h - a[u] * c[k];
            a[u] = a[u] - c[k];

            double f = 0.0;
            double g = 0.0;
            for (j = 0; j <= k - 1; j++)
            {
                a[j * n + k] = a[k * n + j] / h;
                g = 0.0;
                for (int l = 0; l <= j; l++)
                    g = g + a[j * n + l] * a[k * n + l];
                if (j + 1 <= k - 1)
                    for (int l = j + 1; l <= k - 1; l++)
                        g = g + a[l * n + j] * a[k * n + l];
                c[j] = g / h;
                f = f + g * a[j * n + k];
            }
            double h2 = f / (h + h);
            for (j = 0; j <= k - 1; j++)
            {
                f = a[k * n + j];
                g = c[j] - h2 * f;
                c[j] = g;
                for (int l = 0; l <= j; l++)
                {
                    u = j * n + l;
                    a[u] = a[u] - f * c[l] - g * a[k * n + l];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    for (k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}

//Householder1
void Householder_1(double *a, int n, double* q, double* b, double* c)
{
    int u, v;
    for (int k = n - 1; k >= 1; k--)
    {
        double h = 0.0;
        if (k > 1)
            for (int i = 0; i <= k - 1; i++)
            {
                u = k * n + i;
                h = h + a[u] * a[u];
            }
        if (h + 1.0 == 1.0)
        {
            c[k] = 0.0;
            if (k == 1)
                c[k] = a[k * n + k - 1];
            b[k] = 0.0;
        }
        else
        {
            c[k] = sqrt(h);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                c[k] = -c[k];
            h = h - a[u] * c[k];
            a[u] = a[u] - c[k];

            double f = 0.0;
            double g = 0.0;
            for (int i = 0; i <= k - 1; i++)
            {
                a[i * n + k] = a[k * n + i] / h;
                g = 0.0;
                for (int j = 0; j <= i; j++)
                    g = g + a[i * n + j] * a[k * n + j];
                
                if (i + 1 <= k - 1)
                    for (int j = i + 1; j <= k - 1; j++)
                        g = g + a[j * n + i] * a[k * n + j];
                c[i] = g / h;
                
                f = f + g * a[i * n + k];
            }
            
            double h2 = f / (h + h);
            for (int i = 0; i <= k - 1; i++)
            {
                f = a[k * n + i];
                g = c[i] - h2 * f;
                c[i] = g;
                for (int j = 0; j <= i; j++)
                {
                    u = i * n + j;
                    a[u] = a[u] - f * c[j] - g * a[k * n + j];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    for (int k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}

//Householder2
void Householder_2(double *a, int n, double* q, double* b, double* c)
{
    int u, v;
    for (int k = n - 1; k > 1; k--)
    {
        double sum = 0.0;

        for (int i = 0; i <= k - 1; i++)
        {
            u = k * n + i;
            sum = sum + a[u] * a[u];
        }

        if (sum == 0.0)
        {
            c[k] = 0.0;
            b[k] = 0.0;
        }
        else
        {
            c[k] = sqrt(sum);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                c[k] = -c[k];
            sum = sum - a[u] * c[k];
            a[u] = a[u] - c[k];

            double f = 0.0;
            double g = 0.0;
            for (int i = 0; i <= k - 1; i++)
            {
                a[i * n + k] = a[k * n + i] / sum;
                g = 0.0;
                for (int j = 0; j <= i; j++)
                    g = g + a[i * n + j] * a[k * n + j];
                
                if (i + 1 <= k - 1)
                    for (int j = i + 1; j <= k - 1; j++)
                        g = g + a[j * n + i] * a[k * n + j];
                c[i] = g / sum;
                
                f = f + g * a[i * n + k];
            }
            
            double h2 = f / (sum + sum);
            for (int i = 0; i <= k - 1; i++)
            {
                f = a[k * n + i];
                g = c[i] - h2 * f;
                c[i] = g;
                for (int j = 0; j <= i; j++)
                {
                    u = i * n + j;
                    a[u] = a[u] - f * c[j] - g * a[k * n + j];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    c[1] = a[n];
    for (int k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}

//Householder3
void Householder_3(double *a, int n, double* q, double* b, double* c)
{
    int u, v;
    for (int k = n - 1; k > 1; k--)
    {
        double sum = 0.0;

        for (int i = 0; i <= k - 1; i++)
        {
            u = k * n + i;
            sum = sum + a[u] * a[u];
        }

        if (sum == 0.0)
        {
            continue;
            // c[k] = 0.0;
            // b[k] = 0.0;
        }
        else
        {
            double mol = sqrt(sum);
            // c[k] = sqrt(sum);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                mol = -mol;
            c[k] = mol;
            sum = sum - a[u] * mol;
            a[u] = a[u] - mol;

            double f = 0.0;
            double g = 0.0;
            for (int i = 0; i <= k - 1; i++)
            {
                a[i * n + k] = a[k * n + i] / sum;
                g = 0.0;
                for (int j = 0; j <= i; j++)
                    g = g + a[i * n + j] * a[k * n + j];
                
                // if (i + 1 <= k - 1)
                //     for (int j = i + 1; j <= k - 1; j++)
                //         g = g + a[j * n + i] * a[k * n + j];

                for (int j = i + 1; j <= k - 1; j++)
                    g = g + a[j * n + i] * a[k * n + j];

                c[i] = g / sum;
                
                f = f + g * a[i * n + k];
            }
            
            double h2 = f / (sum + sum);
            for (int i = 0; i <= k - 1; i++)
            {
                f = a[k * n + i];
                g = c[i] - h2 * f;
                c[i] = g;
                for (int j = 0; j <= i; j++)
                {
                    u = i * n + j;
                    a[u] = a[u] - f * c[j] - g * a[k * n + j];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    c[1] = a[n];
    for (int k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}

//Householder4
void Householder_4(double *a, int n, double* q, double* b, double* c)
{
    int u, v;
    for (int k = n - 1; k > 1; k--)
    {
        double sum = 0.0;

        for (int i = 0; i <= k - 1; i++)
        {
            u = k * n + i;
            sum = sum + a[u] * a[u];
        }

        if (sum == 0.0)
        {
            continue;
            // c[k] = 0.0;
            // b[k] = 0.0;
        }
        else
        {
            double mol = sqrt(sum);
            // c[k] = sqrt(sum);
            u = k * n + k - 1;
            if (a[u] > 0.0)
                mol = -mol;
            c[k] = mol;
            sum = sum - a[u] * mol;
            a[u] = a[u] - mol;

            double f = 0.0;
            double g = 0.0;

            for (int i = 0; i <= k - 1; i++)
                a[i * n + k] = a[k * n + i] / sum;

            for (int i = 0; i <= k - 1; i++)
            {
                g = 0.0;
                for (int j = 0; j <= k - 1; j++)
                    if(j<=i) g = g + a[i * n + j] * a[k * n + j];
                    else g = g + a[j * n + i] * a[k * n + j];

                c[i] = g / sum;
                f = f + g * a[i * n + k];
            }
            
            double h2 = f / (sum + sum);
            for (int i = 0; i <= k - 1; i++)
            {
                f = a[k * n + i];
                g = c[i] - h2 * f;
                c[i] = g;
                for (int j = 0; j <= i; j++)
                {
                    u = i * n + j;
                    a[u] = a[u] - f * c[j] - g * a[k * n + j];
                }
            }
            b[k] = a[k * n + k];
        }
    }
    c[1] = a[n];
    for (int k = 0; k <= n - 2; k++)
        c[k] = c[k + 1];
    c[n - 1] = 0.0;
    b[0] = a[0];
    b[1] = a[n + 1];
}
