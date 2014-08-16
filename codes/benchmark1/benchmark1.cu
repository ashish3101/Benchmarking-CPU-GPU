/*
	The program computes the square root of the elements of an array on both CPU and GPU and stores the output in another array.
	The ratio of time taken on CPU and GPU increases drastically as number of elements increases but gradually achieves a constant slope following AMDAHL's LAW.
*/

#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void calculate (float *a_d, float *b_d, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		b_d [idx] = sqrt (a_d [idx]);
}

int main (void)
{
	printf ("Sample\tGPU\tCPU\tRatio\tGPU_Error\tCPU_Error\n");
	remove("myspeedComparedata");
	FILE *file;
	for (int repeat = 0; repeat <= 50000; repeat += 100)
	{
		printf ("Repeat %d\n", repeat);
		int samplesize=10;
		float GPUsum=0, CPUsum=0, GPUsumsq=0, CPUsumsq=0;
		for (int sample = 0; sample < samplesize; sample++)
		{
			clock_t start, stop, mid;
			double GPUt = 0.0, CPUt = 0.0;
			start = clock ();

			float *x_h, *y_h, *a_d, *b_d;
			const int N = 262144;

			size_t size = N * sizeof (float);
			x_h = (float *) malloc (size);
			y_h = (float *) malloc (size);

			cudaMalloc ((void **) &a_d, size);
			cudaMalloc ((void **) &b_d, size);

			for (int i = 0; i < N; i++)
				x_h [i] = (float)i;

			cudaMemcpy (a_d, x_h, size, cudaMemcpyHostToDevice);

			int block_size = 4;
			int n_blocks = N/block_size + (N % block_size ? 1 : 0);

			for (int i = 0; i <= repeat; i++)
				calculate <<< n_blocks, block_size >>> (a_d, b_d, N);

			cudaMemcpy (y_h, b_d, size, cudaMemcpyDeviceToHost);

			mid = clock ();
			GPUt = (double)(mid - start) / CLOCKS_PER_SEC;
			printf ("%d\t%.2f\t", sample + 1, GPUt);
			GPUsum += GPUt;
			GPUsumsq += GPUt * GPUt;

			float *a_h, *b_h;
			a_h = (float *) malloc (size);
			b_h = (float *) malloc (size);

			for (int i = 0; i < N; i++)
				a_h [i] = (float)i;

			for (int j = 0; j <= repeat; j++)
				for (int i = 0; i <= N; i++)
					b_h [i] = sqrt (a_h [i]);

			stop = clock ();
			CPUt = (double)(stop - mid) / CLOCKS_PER_SEC;

			printf ("%.2f\t%.0f\n", CPUt, CPUt / GPUt);

			CPUsum += CPUt;
			CPUsumsq += CPUt * CPUt;
		}
		float GPUavg = GPUsum / samplesize;
		float CPUavg = CPUsum / samplesize;

		printf ("Avg.\t%.3f\t%.3f\t%.1f\t%f\t%f\n", GPUavg, CPUavg, CPUavg / GPUavg, sqrt ((GPUsumsq / samplesize) - (GPUavg * GPUavg)), sqrt ((CPUsumsq / samplesize) - (CPUavg * CPUavg)));

		file = fopen ("myspeedComparedata", "a+");

		fprintf (file, "%d\t%.3f\t%.3f\t%.1f\t%f\t%f\n", repeat, GPUavg, CPUavg, CPUavg / GPUavg, sqrt ((GPUsumsq / samplesize) - (GPUavg * GPUavg)), sqrt ((CPUsumsq / samplesize) - (CPUavg * CPUavg)));

		fclose (file);
	}
}
