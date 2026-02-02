// Name: Shishir Adhikari & Ricardo Ayala
// Discription: This program implements a CUDA accelerated CNN for Cats vs Dogs classification
// It includes normalization, convolution, pooling, 
// dense layer, loss, gradients, Adam optimizer, and kernel timing.

// Include standard I/O functions
#include <stdio.h>

// Include general utilities
#include <stdlib.h>

// Include math functions like expf, logf
#include <math.h>

// Include directory reading for loading image file names
#include <dirent.h>

// Include string manipulation functions
#include <string.h>

// Include CPU timing functions
#include <time.h>

// Include CUDA runtime API
#include <cuda_runtime.h>


// Enable STB image loader implementation
#define STB_IMAGE_IMPLEMENTATION

// Include the STB image loader library
#include "stb_image.h"



// Define the image width (pixels)
#define IMG_W 150

// Define the image height (pixels)
#define IMG_H 150

// Number of convolution filters
#define FILTERS 8

// Tile size used for CUDA blocks
#define TILE 16

// MaxPool kernel size (2x2)
#define POOL 2

// Adam optimizer learning rate
#define LR 0.001f

// Adam exponential decay for first moment
#define BETA1 0.9f

// Adam exponential decay for second moment
#define BETA2 0.999f

// Epsilon to avoid division by zero
#define EPS 1e-8f

// Maximum number of images to load
#define MAX_IMAGES 25000

// Ratio of training images (rest is test)
#define TRAIN_RATIO 0.8f

// Number of training epochs
#define EPOCHS 10



// GPU timing variable for normalize kernel
float tNormalize = 0;

// GPU timing variable for convolution kernel
float tConv = 0;

// GPU timing variable for pooling kernel
float tPool = 0;

// GPU timing variable for dense layer kernels
float tDense = 0;

// GPU timing variable for loss kernel
float tLoss = 0;

// GPU timing variable for Adam update kernel
float tAdam = 0;

// Normalize kernel: converts unsigned char pixels (0–255) into float (0–1)
__global__ void normalizeKernel(unsigned char *in, float *out, int n){

    // Compute global thread ID (1D indexing)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid indices
    if(i < n)

        // Store normalized pixel into output array
        out[i] = in[i] * (1.0f / 255.0f);
}

// Convolution kernel applies 3×3 filters to the input image
__global__ void convKernel(float *x, float *f, float *y, int w, int h, int oc){

    // Compute X coordinate this thread handles
    int X = blockIdx.x * TILE + threadIdx.x;

    // Compute Y coordinate this thread handles
    int Y = blockIdx.y * TILE + threadIdx.y;

    // If thread is outside the image, exit
    if(X >= w || Y >= h) return;

    // Loop over output channels (filters)
    for(int k = 0; k < oc; k++){

        // Accumulator for the convolution sum
        float s = 0;

        // Loop over 3×3 kernel vertical direction (ky = -1, 0, 1)
        for(int ky = -1; ky <= 1; ky++)

            // Loop over 3×3 kernel horizontal direction (kx = -1, 0, 1)
            for(int kx = -1; kx <= 1; kx++){

                // Compute clamped input X (avoid going outside boundary)
                int ix = min(max(X + kx, 0), w - 1);

                // Compute clamped input Y (avoid going outside boundary)
                int iy = min(max(Y + ky, 0), h - 1);

                // Accumulate weighted sum for this filter element
                s += x[iy * w + ix] * f[k * 9 + (ky + 1) * 3 + (kx + 1)];
            }

        // Store ReLU output (max(0, sum)) into output feature map
        y[k * w * h + Y * w + X] = fmaxf(s, 0.0f);
    }
}
// MaxPool kernel performs 2×2 pooling on every channel
__global__ void maxPoolKernel(float *in, float *out, int w, int h, int c){

    // Compute X coordinate in the downsampled (pooled) output
    int X = blockIdx.x * TILE + threadIdx.x;

    // Compute Y coordinate in the downsampled (pooled) output
    int Y = blockIdx.y * TILE + threadIdx.y;

    // Compute half-width (pooled width)
    int W2 = w / 2;

    // Compute half-height (pooled height)
    int H2 = h / 2;

    // If this position is outside the pooled output, exit
    if(X >= W2 || Y >= H2) return;

    // Loop over each output channel
    for(int ch = 0; ch < c; ch++){

        // Initialize max value to a very small number
        float m = -1e9f;

        // Loop through pooling window vertically (2 rows)
        for(int py = 0; py < 2; py++)

            // Loop through pooling window horizontally (2 columns)
            for(int px = 0; px < 2; px++){

                // Compute X coordinate in the input feature map
                int ix = X * 2 + px;

                // Compute Y coordinate in the input feature map
                int iy = Y * 2 + py;

                // Update max value for this channel and location
                m = fmaxf(m, in[(ch * h + iy) * w + ix]);
            }

        // Write the pooled maximum to the output array
        out[(ch * H2 + Y) * W2 + X] = m;
    }
}
// Dense layer partial reduction kernel
// Each block reduces a chunk of the input into a single partial sum
__global__ void dense_partial(float *input, float *W, float *partial, int n){

    // Declare dynamic shared memory buffer for reduction
    extern __shared__ float s[];

    // Local thread index (0–1023)
    int tid = threadIdx.x;

    // Global index of the element this thread will multiply
    int i = blockIdx.x * blockDim.x + tid;

    // Compute value = input[i] * W[i] if within bounds, else 0
    float v = (i < n ? input[i] * W[i] : 0);

    // Store the product in shared memory
    s[tid] = v;

    // Synchronize all threads before reduction
    __syncthreads();

    // Perform parallel reduction halving the active threads each step
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){

        // Only threads in the active half participate
        if(tid < stride)

            // Add the corresponding partner value
            s[tid] += s[tid + stride];

        // Sync after each reduction step
        __syncthreads();
    }

    // Thread 0 writes the block's reduced result
    if(tid == 0)
        partial[blockIdx.x] = s[0];
}

// Final dense reduction kernel
// Takes all partial sums and reduces them to one output
__global__ void dense_final(float *partial, float *bias, float *out, int n){

    // Declare shared memory for second-stage reduction
    extern __shared__ float s[];

    // Thread index inside reduction block
    int tid = threadIdx.x;

    // Load partial sum into shared memory (or 0 if thread index >= n)
    s[tid] = (tid < n ? partial[tid] : 0);

    // Sync before reduction
    __syncthreads();

    // Reduce values down to a single sum
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){

        // Active threads perform sum
        if(tid < stride)
            s[tid] += s[tid + stride];

        // Sync after adding
        __syncthreads();
    }

    // Thread 0 computes final activation (sigmoid)
    if(tid == 0){

        // Add bias to final dense sum
        float z = s[0] + *bias;

        // Apply sigmoid activation and save output
        out[0] = 1.0f / (1.0f + expf(-z));
    }
}
// Loss kernel computes binary cross-entropy loss for a single prediction
__global__ void lossParallel(float *pred, int label, float *loss){

    // Load predicted probability
    float p = pred[0];

    // Compute binary cross-entropy loss and write to output
    loss[0] = -label * logf(p + 1e-7f) - (1 - label) * logf(1 - p + 1e-7f);
}



// Gradient kernel computes dLoss/dW contribution for every weight
__global__ void gradParallel(float *grad, float p, int label, int n){

    // Compute global index for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid indices
    if(i < n)

        // Each gradient is (prediction - label)
        grad[i] = (p - label);
}



// Adam optimizer update kernel
// Updates W, m, v arrays in parallel using Adam formula
__global__ void adamUpdate(float *W, float *m, float *v, float *grad,
                           int n, float lr){

    // Compute global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Only update valid weight indices
    if(i < n){

        // Update biased 1st moment estimate: m = β1*m + (1−β1)*grad
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];

        // Update biased 2nd moment estimate: v = β2*v + (1−β2)*grad^2
        v[i] = BETA2 * v[i] + (1 - BETA2) * grad[i] * grad[i];

        // Apply bias correction to first moment
        float mh = m[i] / (1 - BETA1);

        // Apply bias correction to second moment
        float vh = v[i] / (1 - BETA2);

        // Update weight using Adam equation
        W[i] -= lr * mh / (sqrtf(vh) + EPS);
    }
}
// Function to load image file paths from a directory
// It fills paths[], labels[], and returns the next write index
int load_images(const char *dir, char paths[][512], int labels[],
    int start, int max, int val)
{
// Attempt to open the directory
DIR *d = opendir(dir);

// If directory cannot be opened, return start index unchanged
if(!d) return start;

// Structure used to iterate through directory entries
struct dirent *ent;

// Counter starts at the provided starting index
int c = start;

// Read directory entries until no more or until reaching max images
while((ent = readdir(d)) && c < max){

// Check if filename contains ".jpg" or ".png"
if(strstr(ent->d_name, ".jpg") || strstr(ent->d_name, ".png")){

// Construct full path: dir + "/" + filename
snprintf(paths[c], 512, "%s/%s", dir, ent->d_name);

// Assign label (0 for cat, 1 for dog)
labels[c] = val;

// Move to the next index
c++;
}
}

// Close directory when finished
closedir(d);

// Return updated counter (new starting index for next call)
return c;
}
// Entry point of the CUDA CNN program
int main(){

    // Start CPU wall-clock timer for total program time
    clock_t progStart = clock();


    // Create array to store full image paths
    char paths[MAX_IMAGES][512];

    // Array to store labels (0 = cat, 1 = dog)
    int labels[MAX_IMAGES];

    // Counter for total number of images loaded
    int n = 0;


    // Load cat images from folder and assign label 0
    n = load_images("PetImages/Cat", paths, labels, n, MAX_IMAGES, 0);

    // Load dog images from folder and assign label 1
    n = load_images("PetImages/Dog", paths, labels, n, MAX_IMAGES, 1);


    // If no images were found, exit the program
    if(n == 0){ printf("No images found.\n"); return 0; }


    // Compute the number of training images
    int trainN = n * TRAIN_RATIO;

    // Compute the number of testing images
    int testN  = n - trainN;


    // Display program header
    printf("CUDA CNN Parallel 1024\n");

    // Show how many images were loaded in total, train, and test
    printf("Loaded %d images (train %d, test %d)\n", n, trainN, testN);


    // Compute number of pixels in an image (150 × 150)
    int size = IMG_W * IMG_H;

    // Compute dense layer input length after pooling
    int len  = (IMG_W / 2) * (IMG_H / 2) * FILTERS;


    // Device pointer for weights
    float *W;

    // Device pointer for bias
    float *b;

    // Device pointer for Adam first moment (m)
    float *m;

    // Device pointer for Adam second moment (v)
    float *v;

    // Device pointer for gradients
    float *grad;

    // Device pointer for prediction output
    float *d_pred;

    // Device pointer for loss value
    float *d_loss;


    // Allocate GPU memory for weights
    cudaMalloc(&W, len * sizeof(float));

    // Allocate GPU memory for bias
    cudaMalloc(&b, sizeof(float));

    // Allocate GPU memory for Adam moment m
    cudaMalloc(&m, len * sizeof(float));

    // Allocate GPU memory for Adam moment v
    cudaMalloc(&v, len * sizeof(float));

    // Allocate GPU memory for gradients
    cudaMalloc(&grad, len * sizeof(float));

    // Allocate GPU memory for predicted output
    cudaMalloc(&d_pred, sizeof(float));

    // Allocate GPU memory for loss value
    cudaMalloc(&d_loss, sizeof(float));


    // Allocate host memory for initializing weights
    float *hW = (float*) malloc(len * sizeof(float));

    // Initialize weights with small random values
    for(int i = 0; i < len; i++)
        hW[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Initialize bias to zero
    float hb = 0;


    // Copy initialized weights to device memory
    cudaMemcpy(W, hW, len * sizeof(float), cudaMemcpyHostToDevice);

    // Copy bias value to device memory
    cudaMemcpy(b, &hb, sizeof(float), cudaMemcpyHostToDevice);


    // Allocate host memory for convolution filters (8 filters × 9 weights)
    float *hf = (float*) malloc(FILTERS * 9 * sizeof(float));

    // Initialize each filter weight randomly
    for(int i = 0; i < FILTERS * 9; i++)
        hf[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;


    // Device pointer for convolution filters
    float *d_filter;

    // Allocate GPU memory for filters
    cudaMalloc(&d_filter, FILTERS * 9 * sizeof(float));

    // Copy filters from host to device
    cudaMemcpy(d_filter, hf, FILTERS * 9 * sizeof(float), cudaMemcpyHostToDevice);


    // Configure CUDA block size (16 × 16 threads)
    dim3 block(TILE, TILE);

    // Configure CUDA grid size to cover whole image
    dim3 grid((IMG_W + TILE - 1) / TILE, (IMG_H + TILE - 1) / TILE);


    // CUDA events for timing kernel execution
    cudaEvent_t ks, ke;

    // Create start event
    cudaEventCreate(&ks);

    // Create end event
    cudaEventCreate(&ke);

        // Begin training loop over all epochs
        for(int e = 1; e <= EPOCHS; e++){

            // Accumulator for training loss
            float TL = 0;
    
            // Accumulator for training accuracy
            float TA = 0;
    
            // Start CPU timer for epoch duration
            clock_t epochStart = clock();
    
    
            // Loop through each training image
            for(int idx = 0; idx < trainN; idx++){
    
                // Retrieve label (0 cat, 1 dog)
                int label = labels[idx];
    
                // Variables for width, height, channels
                int ww, hh, ch;
    
                // Load image from file as grayscale (1-channel)
                unsigned char *img = stbi_load(paths[idx], &ww, &hh, &ch, 1);
    
                // If image failed to load, skip it
                if(!img) continue;
    
    
                // Device pointer for raw image
                unsigned char *d_img;
    
                // Device pointer for normalized image
                float *d_norm;
    
                // Device pointer for convolution output
                float *d_conv;
    
                // Device pointer for pooled output
                float *d_pool;
    
    
                // Allocate GPU memory for raw image bytes
                cudaMalloc(&d_img, size);
    
                // Allocate GPU memory for normalized output
                cudaMalloc(&d_norm, size * sizeof(float));
    
                // Allocate GPU memory for conv output (FILTERS × size)
                cudaMalloc(&d_conv, FILTERS * size * sizeof(float));
    
                // Allocate GPU memory for pooled output (FILTERS × size/4)
                cudaMalloc(&d_pool, FILTERS * (size / 4) * sizeof(float));
    
    
                // Copy image from CPU to GPU
                cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    
    
                // Variable to store kernel runtime (ms)
                float ms;
    
    
    
                // Start timing for normalize kernel
                cudaEventRecord(ks);
    
                // Launch normalization kernel (1024 threads per block)
                normalizeKernel<<<(size + 1023) / 1024, 1024>>>(d_img, d_norm, size);
    
                // Mark end of kernel
                cudaEventRecord(ke);
    
                // Wait for kernel to finish
                cudaEventSynchronize(ke);
    
                // Compute elapsed time
                cudaEventElapsedTime(&ms, ks, ke);
    
                // Add to normalize time accumulator
                tNormalize += ms;
    
    
    
                // Start timing for convolution kernel
                cudaEventRecord(ks);
    
                // Launch 3×3 convolution + ReLU
                convKernel<<<grid, block>>>(d_norm, d_filter, d_conv,
                                            IMG_W, IMG_H, FILTERS);
    
                // Mark end event
                cudaEventRecord(ke);
    
                // Synchronize to measure time
                cudaEventSynchronize(ke);
    
                // Add elapsed time to convolution accumulator
                cudaEventElapsedTime(&ms, ks, ke);
    
                // Add to global convolution time
                tConv += ms;
    
    
    
                // Start timing for max-pool kernel
                cudaEventRecord(ks);
    
                // Launch 2×2 max-pooling kernel
                maxPoolKernel<<<grid, block>>>(d_conv, d_pool,
                                               IMG_W, IMG_H, FILTERS);
    
                // End timing
                cudaEventRecord(ke);
    
                // Wait for completion
                cudaEventSynchronize(ke);
    
                // Compute elapsed time for pooling
                cudaEventElapsedTime(&ms, ks, ke);
    
                // Accumulate pool time
                tPool += ms;
            // Set number of threads per block for dense reduction
            int threads = 1024;

            // Compute number of blocks needed to cover full dense length
            int blocks = (len + threads - 1) / threads;

            // Device pointer for partial sums from dense partial kernel
            float *d_partial;

            // Allocate GPU memory for partial sums
            cudaMalloc(&d_partial, blocks * sizeof(float));



            // Start timing dense partial kernel
            cudaEventRecord(ks);

            // Launch first stage of dense reduction (parallel dot product)
            dense_partial<<<blocks, threads, threads * sizeof(float)>>>
                (d_pool, W, d_partial, len);

            // End timing event
            cudaEventRecord(ke);

            // Wait for kernel completion
            cudaEventSynchronize(ke);

            // Get runtime in milliseconds
            cudaEventElapsedTime(&ms, ks, ke);

            // Accumulate dense kernel time
            tDense += ms;



            // Start timing final dense reduction kernel
            cudaEventRecord(ks);

            // Launch the second stage of reduction to produce output value
            dense_final<<<1, threads, threads * sizeof(float)>>>
                (d_partial, b, d_pred, blocks);

            // End timing event
            cudaEventRecord(ke);

            // Wait for kernel to finish
            cudaEventSynchronize(ke);

            // Measure the time for dense_final
            cudaEventElapsedTime(&ms, ks, ke);

            // Add to dense total
            tDense += ms;


            // Free the temporary partial buffer
            cudaFree(d_partial);



            // Start timing loss kernel
            cudaEventRecord(ks);

            // Launch loss kernel (binary cross entropy)
            lossParallel<<<1, 1>>>(d_pred, label, d_loss);

            // End event for timing
            cudaEventRecord(ke);

            // Wait for kernel
            cudaEventSynchronize(ke);

            // Measure time for loss kernel
            cudaEventElapsedTime(&ms, ks, ke);

            // Accumulate loss time
            tLoss += ms;



            // Host variable to store loss
            float loss;

            // Host variable to store predicted probability
            float pred;

            // Copy loss value from GPU to CPU
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

            // Copy predicted value from GPU to CPU
            cudaMemcpy(&pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);


            // Add this image's loss to epoch total
            TL += loss;

            // Increase accuracy counter if prediction matches label
            if((pred > 0.5) == label)
                TA++;



            // Launch gradient computation kernel (parallel)
            gradParallel<<<blocks, threads>>>(grad, pred, label, len);



            // Start timing Adam optimizer kernel
            cudaEventRecord(ks);

            // Update weights using Adam in parallel
            adamUpdate<<<blocks, threads>>>(W, m, v, grad, len, LR);

            // Record end event
            cudaEventRecord(ke);

            // Synchronize for timing
            cudaEventSynchronize(ke);

            // Measure Adam update time
            cudaEventElapsedTime(&ms, ks, ke);

            // Accumulate Adam time
            tAdam += ms;



            // Free GPU memory for original image
            cudaFree(d_img);

            // Free GPU memory for normalized image
            cudaFree(d_norm);

            // Free GPU memory for convolution output
            cudaFree(d_conv);

            // Free GPU memory for pooling output
            cudaFree(d_pool);

            // Free CPU memory for the loaded image
            stbi_image_free(img);
        }
        // End CPU timer for this epoch
        clock_t epochEnd = clock();

        // Compute epoch duration in seconds
        float epochSec = (float)(epochEnd - epochStart) / CLOCKS_PER_SEC;

        // Compute average milliseconds per training step
        float msPerStep = (epochSec * 1000.0f) / trainN;

        // Print epoch summary with accuracy and loss
        printf("Epoch %d/%d — %.0fs — %.1fms/step — train_accuracy: %.4f — train_loss: %.4f\n",
               e, EPOCHS, epochSec, msPerStep,
               TA / trainN, TL / trainN);
    }

    // Accumulator for test loss
    float TL2 = 0;

    // Accumulator for test accuracy
    float TA2 = 0;


    // Loop through testing images (no training or updating)
    for(int idx = trainN; idx < n; idx++){

        // Get label (0 or 1)
        int label = labels[idx];

        // Variables for image metadata (ignored mostly)
        int ww, hh, ch;

        // Load test image as grayscale
        unsigned char *img = stbi_load(paths[idx], &ww, &hh, &ch, 1);

        // Skip if image fails to load
        if(!img) continue;


        // Device pointer for raw image
        unsigned char *d_img;

        // Device pointer for normalized pixels
        float *d_norm;

        // Device pointer for convolution output
        float *d_conv;

        // Device pointer for pooled output
        float *d_pool;


        // Allocate memory on GPU for raw image
        cudaMalloc(&d_img, size);

        // Allocate memory for normalized output
        cudaMalloc(&d_norm, size * sizeof(float));

        // Allocate memory for convolution output for all filters
        cudaMalloc(&d_conv, FILTERS * size * sizeof(float));

        // Allocate memory for pooled output (downsampled)
        cudaMalloc(&d_pool, FILTERS * (size / 4) * sizeof(float));


        // Copy image to device memory
        cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);



        // Launch normalization kernel (no timing here)
        normalizeKernel<<<(size + 1023) / 1024, 1024>>>(d_img, d_norm, size);

        // Launch convolution kernel
        convKernel<<<grid, block>>>(d_norm, d_filter, d_conv,
                                    IMG_W, IMG_H, FILTERS);

        // Launch max pooling kernel
        maxPoolKernel<<<grid, block>>>(d_conv, d_pool,
                                       IMG_W, IMG_H, FILTERS);



        // Use the same number of threads and blocks as in training
        int threads = 1024;
        int blocks = (len + threads - 1) / threads;

        // Device pointer for partial dense sums
        float *d_partial;

        // Allocate buffer for partial dense sums
        cudaMalloc(&d_partial, blocks * sizeof(float));


        // Launch dense partial reduction
        dense_partial<<<blocks, threads, threads * sizeof(float)>>>
            (d_pool, W, d_partial, len);

        // Launch final dense reduction
        dense_final<<<1, threads, threads * sizeof(float)>>>
            (d_partial, b, d_pred, blocks);


        // Free temporary reduction memory
        cudaFree(d_partial);



        // Compute loss for test sample
        lossParallel<<<1, 1>>>(d_pred, label, d_loss);


        // Variables to store loss and prediction
        float loss;
        float pred;

        // Copy loss from GPU to CPU
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

        // Copy predicted value from GPU to CPU
        cudaMemcpy(&pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);


        // Accumulate total test loss
        TL2 += loss;

        // Count correct predictions
        if((pred > 0.5) == label)
            TA2++;



        // Free GPU memory for test image
        cudaFree(d_img);
        cudaFree(d_norm);
        cudaFree(d_conv);
        cudaFree(d_pool);

        // Free CPU image memory
        stbi_image_free(img);
    }

    // Print final test accuracy and test loss
    printf("\nTest Accuracy: %.4f — Test Loss: %.4f\n",
           TA2 / testN, TL2 / testN);
    // Print header for GPU kernel timing summary
    printf("\n==== FINAL GPU Kernel Time Summary ====\n");

    // Print total time spent in normalization kernels
    printf("normalize: %.3f ms\n", tNormalize);

    // Print total time spent in convolution kernels
    printf("conv     : %.3f ms\n", tConv);

    // Print total time spent in maxpool kernels
    printf("pool     : %.3f ms\n", tPool);

    // Print total time spent in dense layer kernels
    printf("dense    : %.3f ms\n", tDense);

    // Print total time spent in loss kernels
    printf("loss     : %.3f ms\n", tLoss);

    // Print total time spent in Adam update kernels
    printf("adam     : %.3f ms\n", tAdam);


    // Compute total GPU kernel time for all major operations
    float totalGPU = tNormalize + tConv + tPool + tDense + tLoss + tAdam;

    // Print total GPU kernel time in milliseconds and seconds
    printf("Total GPU kernel time: %.3f ms (%.3f s)\n",
           totalGPU, totalGPU / 1000.0f);


    // Stop CPU timer for full program runtime
    clock_t progEnd = clock();

    // Print total CPU wall-clock time for entire program
    printf("Total program time (CPU wall clock): %.3f s\n",
           (double)(progEnd - progStart) / CLOCKS_PER_SEC);


    // Return success code
    return 0;
}
    
