/*
	The program takes an array as input, multiply the elements with 2 and stores the output in another array.
	For the array size less than 4000, CPU runs faster than GPU and then GPU takes over CPU's performance.
*/

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

__global__ void vecMult_d (int *A, int *B, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	if  (i  <  N)
	{
		B [i] = A [i] * 2;
	}
}

void vecMult_h(int *A, int *B, int N)
{
	for (int i  =0 ; i < N; i++)
	{
		B[i] = A [i] * 2;
	}
}

int main()
{
	int *a_h, *b_h;
	int *a_d, *b_d;
	int blocksize = 512, n;
	struct timeval t1_start, t1_end, t2_start, t2_end;
	double time_d, time_h;
	remove ("gpuresult");
	remove ("cpuresult");
	FILE *fp1, *fp2;
	fp1 = fopen ("gpuresult", "a+");
	fp2 = fopen ("cpuresult", "a+");

	for (n = 0; n < 8000; n += 1)
	{
		a_h = (int *)malloc(sizeof(int)*n);
		b_h = (int *)malloc(sizeof(int)*n);

		cudaMalloc ((void **) &a_d, n * sizeof (int));
		cudaMalloc ((void **) &b_d, n * sizeof (int));
		dim3 dimBlock (blocksize);
		dim3 dimGrid (ceil (float (n) / float (dimBlock.x)));

		for (int j = 0; j < n; j++)
			a_h [j] = j;

		cudaMemcpy (a_d, a_h, n * sizeof (int), cudaMemcpyHostToDevice);
		gettimeofday (&t1_start, 0);
		vecMult_d <<<dimGrid, dimBlock>>> (a_d, b_d, n);
		cudaThreadSynchronize ();
		gettimeofday (&t1_end, 0);
		cudaMemcpy (b_h, b_d, n * sizeof (int), cudaMemcpyDeviceToHost);

		gettimeofday (&t2_start, 0);
		vecMult_h (a_h, b_h, n);
		gettimeofday (&t2_end, 0);

		time_d = (t1_end.tv_sec-t1_start.tv_sec) * 1000000 + t1_end.tv_usec - t1_start.tv_usec;
		time_h = (t2_end.tv_sec-t2_start.tv_sec) * 1000000 + t2_end.tv_usec - t2_start.tv_usec;

		fprintf (fp1, "%d\t%lf\t\n", n, time_d);
		fprintf (fp2, "%d\t%lf\t\n", n, time_h);
		free (a_h);
		free (b_h);
		cudaFree (a_d);
		cudaFree (b_d);
	}
	return (0);
}
