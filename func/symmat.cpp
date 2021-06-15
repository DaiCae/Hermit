#include "math.h"
#include "time.h"
#include "stdlib.h"

void symmat(double B[][100], double A[], int n)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (j > i)
                B[i][j] = rand() % 10 + 1;
            else if (i > j)
                B[i][j] = B[j][i];
            else
                B[i][j] = rand() % 10 + 1;
        }
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            A[i * n + j] = B[i][j];
    }
}
