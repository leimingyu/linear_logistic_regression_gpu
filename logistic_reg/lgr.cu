#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <string.h>


#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

#define DBG 0

typedef uint32_t uint32;


void usage(char *argv0); 
void print_array(float *array, int rows, int cols, const char*msg);
inline int BLK(int len, int blksize);

//---------------------------------------------------------------------------//
// Usage
//---------------------------------------------------------------------------//
void usage(char *argv0) {
	const char *help =
		"\nUsage: %s [options]\n\n"
		"    -n input_size       :length of input samples  [default=1023]\n"		
		"    -f feature_size     :length of input features [default=1023]\n"		
		"    -i max_iterations   :maximum iterations [default=10]\n"
		"    -a alpha            :gradient step size [default=0.01]\n";
	fprintf(stderr, help, argv0);
	exit(-1);
}


void print_array(float *array, int rows, int cols, const char*msg)
{
	if(msg != NULL) printf("\n%s\n", msg);

	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			printf("%f ", array[i * cols + j]);
		}
		printf("\n");
	}
		printf("\n");
}


inline int BLK(int len, int blksize) {
	return (len + blksize - 1) / blksize;
}

//---------------------------------------------------------------------------//
// Kernels 
//---------------------------------------------------------------------------//
__global__ void kernel_sigmoid (const float* __restrict__ d_xt,
		const float* __restrict__ d_y,
		const int input_size,
		float *d_sigmoid_error)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);	
	if(gx < input_size) {
		float z = -d_xt[gx];
		float h = 1.f / ( 1.f + expf(z) );
		d_sigmoid_error[gx] = d_y[gx] - h;
	}
}

__global__ void kernel_update_weight (const float* __restrict__ d_theta_tmp, 
		const float alpha,
		const int feature_size,
		float *d_theta)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);	
	if(gx < feature_size) {
		float weight = d_theta[gx];
		d_theta[gx]  = weight + d_theta_tmp[gx] * alpha;
	}
}

//---------------------------------------------------------------------------//
// Main 
//---------------------------------------------------------------------------//
int main(int argc, char **argv)
{
	/// intialize parameters
	//uint32 input_size   = 1023;
	//uint32 feature_size = 1023;

	int input_size   = 95;
	int feature_size = 63;
	int iterations   = 100;
	float  alpha        = 0.0001;

	float cu_alpha      = 1.0f;
	float cu_beta       = 0.0f;

	/// cublas
    cublasStatus_t status;
    cublasHandle_t handle;

	
	/// command line options
	int opt;
	extern char   *optarg;
	while ((opt = getopt (argc, argv, "n:f:i:a:")) != EOF)
	{
		switch(opt)
		{
			case 'n':
				input_size = atoi(optarg);
				break;

			case 'f':
				feature_size = atoi(optarg);
				break;

			case 'i':
				iterations = atoi(optarg);
				break;

			case 'a':
				alpha = atof(optarg);
				break;

			case '?':
				usage(argv[0]);	
				break;
				
			default:
				usage(argv[0]);
				break;
		}
	}
	printf("input size :   %d\n", input_size);
	printf("feature size : %d\n", feature_size);
	printf("iterations :   %d\n", iterations);
	printf("alpha:         %f\n", alpha);

	/// initialize cublas
	// host side reference handler
	status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "failed to create handle.\n");
        return EXIT_FAILURE;
    }


	/// initalize data sets
	srand(time(NULL));

	// host
	float *x, *y, *theta;

	// device
	float *d_x, *d_y, *d_theta;

	// intermediate data 
	float *d_xt;                // x * theta
	float *d_theta_tmp; 
	float *d_sigmoid_error;		// sigmoid(xt - y)

	/// add ones to input x
	input_size = input_size + 1;
	feature_size = feature_size + 1;

	x     = (float *) malloc (sizeof(float) * input_size * feature_size);
	y     = (float *) malloc (sizeof(float) * input_size);

	// add theta(0) for x(0)
	// set theta to zeros
	theta = (float *) malloc (sizeof(float) * feature_size);
	//memset(theta, 0, sizeof(float) * feature_size);
	for(int i=0; i<feature_size; i++) {
		theta[i] = 0.1f;	
	}

	
	// pad ones to the 1st colum of input x
	for(int i = 0; i < input_size; i++)
	{
		// j = 0
		x[i * feature_size] = 1.0;

		for(int j = 1; j < feature_size; j++)
		{
			x[i * feature_size + j] = rand() / (float) RAND_MAX;	
		}
		y[i] = rand() / (float) RAND_MAX; 
	}

#if DBG
	print_array(x, 		input_size, 	feature_size, 	"input x");
	print_array(y, 		input_size, 	1, 				"y");
	print_array(theta, 	feature_size, 	1, 				"theta");

	float *xt = (float*) malloc(sizeof(float) * input_size);
	float *sigmoid_error = (float*) malloc(sizeof(float) * input_size);
	float *theta_tmp = (float*) malloc(sizeof(float) * feature_size);
#endif


	/// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&d_x,         input_size * feature_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_y,         input_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_theta,     feature_size * sizeof(float)));	// weight
	checkCudaErrors(cudaMalloc((void **)&d_xt,        input_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_theta_tmp, feature_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_sigmoid_error,    input_size * sizeof(float)));


	/// cublas for d_x
	status = cublasSetMatrix (input_size, 
							feature_size, 
							sizeof(float), 
							x, 
							input_size, 
							d_x, 
							input_size); 

	if (status != CUBLAS_STATUS_SUCCESS) { 
		printf ("Error : d_x failed to allocate.\n"); 
		cublasDestroy(handle); 
		return EXIT_FAILURE; 
	}

	/// cublas for d_y
	status = cublasSetVector (input_size, 
								sizeof(float), 
								y, 
								1, 
								d_y, 
								1); 
	if (status != CUBLAS_STATUS_SUCCESS) { 
		printf ("Error : d_y failed to allocate.\n"); 
		cublasDestroy(handle); 
		return EXIT_FAILURE; 
	}

	/// cublas for d_theta
	status = cublasSetVector(feature_size,
			sizeof(float),
			theta,
			1,
			d_theta,
			1);
	if (status != CUBLAS_STATUS_SUCCESS) { 
		printf ("Error : d_theta failed to allocate.\n"); 
		return EXIT_FAILURE; 
	}

	//-----------------------------------------------------------------------//
	// Gradient Descent
	//-----------------------------------------------------------------------//
	printf("\nStart linear regression using gradient descent.\n");

	for(int id = 0; id < iterations; id++)
	{

		///--------------------------------------------------------------------
		///  X * theta  = xt
		///--------------------------------------------------------------------
		status = cublasSgemv(handle, 
				CUBLAS_OP_T, 
				feature_size, 
				input_size, 
				&cu_alpha, 
				d_x, 
				feature_size, 
				d_theta,
				1, 
				&cu_beta, 
				d_xt,
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to compute x * theta.\n"); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

#if DBG
		cublasGetMatrix(input_size, feature_size, sizeof(float), d_x, input_size, x, input_size);
		print_array(x, 		input_size, 	feature_size, 	"input x");

		//cublasGetVector(feature_size, sizeof(float), d_theta, 1, theta, 1);
		//print_array(theta, 	feature_size, 	1, 				"theta");

		//cublasGetVector(input_size, sizeof(float), d_xt, 1, xt, 1);
		//print_array(xt, 	input_size, 	1, 				"x * theta =  xt");
#endif

		///--------------------------------------------------------------------
		/// h =sigmoid (xt)
		/// error =  (y - h)
		///--------------------------------------------------------------------
		dim3 blkDim = dim3(256, 1, 1);
		dim3 grdDim = dim3(BLK(input_size, blkDim.x), 1, 1);

		kernel_sigmoid <<< grdDim, blkDim >>> (d_xt, d_y, input_size, d_sigmoid_error);


#if DBG
		cublasGetVector(input_size, sizeof(float), d_sigmoid_error, 1, sigmoid_error, 1);
		print_array(sigmoid_error, 	input_size, 	1, 				"sigmoid(xt - y) =  sigmoid_error");
#endif


		///--------------------------------------------------------------------
		/// d_x.transpose() x sigmoid_error = errr_along_feature_dim 
		///--------------------------------------------------------------------
		/// multiply the difference for each column (feature) in X, sum along column
		/// generate a feature_size vector to update theta

		cu_alpha = 1.f;

		status = cublasSgemv(handle, 
				CUBLAS_OP_N, 
				feature_size, 
				input_size, 
				&cu_alpha, 
				d_x,				// input 
				feature_size, 
				d_sigmoid_error,	// input
				1, 
				&cu_beta, 
				d_theta_tmp,		// output
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

#if DBG
		cublasGetVector(feature_size, sizeof(float), d_theta_tmp, 1, theta_tmp, 1);
		print_array(theta_tmp, 	feature_size, 	1, "theta_tmp : error along the feature dim");
#endif


		///--------------------------------------------------------------------
		/// update theta : d_theta
		///--------------------------------------------------------------------
		dim3 blkDim_1 = dim3(256, 1, 1);
		dim3 grdDim_1 = dim3(BLK(feature_size, blkDim.x), 1, 1);

		kernel_update_weight <<< grdDim_1, blkDim_1 >>> (d_theta_tmp, alpha, feature_size, d_theta);
	}

	printf("End logistic regression.\n");
	printf("Exit Program.\n");


	// release
	free(x);
	free(y);
	free(theta);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_theta);
	cudaFree(d_xt);
	cudaFree(d_sigmoid_error);
	cudaFree(d_theta_tmp);

#if DBG
	free(xt);
	free(sigmoid_error);
	free(theta_tmp);
#endif

	cublasDestroy(handle); 

	cudaDeviceReset();

	return 0;
}
