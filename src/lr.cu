#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <string.h>


#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cublas_v2.h>


typedef uint32_t uint32;


void usage(char *argv0); 

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



//---------------------------------------------------------------------------//
// Main 
//---------------------------------------------------------------------------//
int main(int argc, char **argv)
{
	/// intialize parameters
	uint32 input_size   = 1023;
	uint32 feature_size = 1023;
	uint32 iterations   = 10;
	float  alpha        = 0.01;

    float cu_alpha      = 1.0f;
    float cu_beta       = 0.0f;

	/// cublas
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasHandle_t handle_dev;

	
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

	// device side reference handler
	status = cublasCreate(&handle_dev);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "failed to create handle_dev.\n");
        return EXIT_FAILURE;
    }
    cublasSetPointerMode(handle_dev, CUBLAS_POINTER_MODE_DEVICE);

	/// initalize data sets
	srand(time(NULL));

	// host
	float *x, *y, *theta;

	// device
	float *d_x, *d_y, *d_theta;
	float *d_costJ;             // keep track of cost
	// intermediate data 
	float *d_xt;                // x * theta
	float *d_theta_tmp; 
	float *d_xt_tmp;

	//float costJ = 0.f;

	/// add ones to input x
	input_size = input_size + 1;
	feature_size = feature_size + 1;

	x     = (float *) malloc (sizeof(float) * input_size * feature_size);
	y     = (float *) malloc (sizeof(float) * input_size);

	// add theta(0) for x(0)
	// set theta to zeros
	theta = (float *) malloc (sizeof(float) * feature_size);
	memset(theta, 0, sizeof(float) * feature_size);

	
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

	/// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&d_x,         input_size * feature_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_y,         input_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_theta,     feature_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_xt,        input_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_theta_tmp, feature_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_costJ,     iterations * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_xt_tmp,    input_size * sizeof(float)));


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
		cudaFree (d_x); 
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
		cudaFree (d_y); 
		cublasDestroy(handle); 
		return EXIT_FAILURE; 
	}

	/// cublas for d_theta
	status = cublasSetVector (feature_size, 
							  sizeof(float), 
							  theta, 
							  1, 
							  d_theta, 
							  1); 

	if (status != CUBLAS_STATUS_SUCCESS) { 
		printf ("Error : d_theta failed to allocate.\n"); 
		cudaFree (d_theta); 
		cublasDestroy(handle); 
		return EXIT_FAILURE; 
	}

	/// FIXME: add feature normalization 

	//-----------------------------------------------------------------------//
	// Gradient Descent
	//-----------------------------------------------------------------------//

	for(int id = 0; id < iterations; id++)
	{

		//----------------------//
		// Update theta
		//----------------------//

		///  X * theta 
		status = cublasSgemv(handle, 
				CUBLAS_OP_N, 
				input_size, 
				feature_size, 
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
			cudaFree (d_x); 
			cudaFree (d_theta); 
			cudaFree (d_xt); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		/// compute X * theta - y, by  ( xt - y )
		cu_alpha = -1.f;
		status = cublasSaxpy(handle, 
				input_size,
				&cu_alpha, 
				d_y, 
				1, 
				d_xt,
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to compute xt - y .\n"); 
			cudaFree (d_y); 
			cudaFree (d_xt); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		/// compute sum( X * xt, 1) 
		/// multiply the difference for each column (feature) in X, sum along column
		/// generate a feature_size vector to update theta
		cu_alpha = 1.f;
		status = cublasSgemv(handle, 
				CUBLAS_OP_T, 
				input_size, 
				feature_size, 
				&cu_alpha, 
				d_x, 
				input_size, 
				d_xt,
				1, 
				&cu_beta, 
				d_theta_tmp,
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to compute theta_temp.\n"); 
			cudaFree (d_x); 
			cudaFree (d_xt); 
			cudaFree (d_theta_tmp); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		/// update theta 
		/// theta - (alpha / input_size) .* theta_tmp
		cu_alpha = - (alpha / (float) input_size);
		status = cublasSaxpy(handle, 
				feature_size,
				&cu_alpha, 
				d_theta_tmp, 
				1, 
				d_theta,
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to update theta.\n"); 
			cudaFree (d_theta_tmp); 
			cudaFree (d_theta); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		//------------------------------------//
		// compute costJ with the updated theta
		//------------------------------------//

		/// (1) xt = X * theta - y
		/// (2) costJ = xt' * xt / (2 * input_size)

		cu_alpha = 1.f;
		status = cublasSgemv(handle, 
				CUBLAS_OP_N, 
				input_size, 
				feature_size, 
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
			cudaFree (d_x); 
			cudaFree (d_theta); 
			cudaFree (d_xt); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		/// compute ( xt - y )
		cu_alpha = -1.f;
		status = cublasSaxpy(handle, 
				input_size,
				&cu_alpha, 
				d_y, 
				1, 
				d_xt,
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to compute xt - y .\n"); 
			cudaFree (d_y); 
			cudaFree (d_xt); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		// initialize to 0
		checkCudaErrors(cudaMemset(d_xt_tmp, 0, input_size * sizeof(float))); 

		/// d_xt_tmp = ( 0.5 / input_size)  * d_xt
		cu_alpha = 0.5 / (float)input_size;
		status = cublasSaxpy(handle, 
				input_size,
				&cu_alpha, 
				d_xt, 
				1, 
				d_xt_tmp,
				1);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to compute xt_tmp .\n"); 
			cudaFree (d_xt); 
			cudaFree (d_xt_tmp); 
			cublasDestroy(handle); 
			return EXIT_FAILURE; 
		}

		/// dot product on d_xt, the cost remains on the device
		status = cublasSdot(handle_dev, 
				input_size,
				d_xt_tmp, 
				1, 
				d_xt,
				1,
				&d_costJ[id]);

		if (status != CUBLAS_STATUS_SUCCESS) { 
			printf ("Error : failed to compute costJ.\n"); 
			cudaFree (d_xt_tmp); 
			cudaFree (d_xt); 
			cublasDestroy(handle_dev); 
			return EXIT_FAILURE; 
		}
	}



	// release
	free(x);
	free(y);
	free(theta);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_theta);
	cudaFree(d_xt);
	cudaFree(d_theta_tmp);
	cudaFree(d_costJ);
	cudaFree(d_xt_tmp);

	cublasDestroy(handle); 
	cublasDestroy(handle_dev); 

	cudaDeviceReset();

	return 0;
}
