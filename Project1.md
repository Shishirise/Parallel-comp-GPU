# Include iamge.h
[stb_image.h raw file](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h)


```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include "stb_image.h"

#define IMG_W 150
#define IMG_H 150
#define FILTERS 8
#define KSIZE 3
#define TILE 16
#define POOL 2
#define LR 0.001f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPS 1e-8f
#define MAX_IMAGES 2800
#define TRAIN_RATIO 0.8f
#define EPOCHS 10

// ---------------- GPU Kernels ----------------
__global__ void normalizeKernel(unsigned char *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / 255.0f;
}

__global__ void convKernel(float *x, float *f, float *y, int w, int h, int outc) {
    int X = blockIdx.x * TILE + threadIdx.x;
    int Y = blockIdx.y * TILE + threadIdx.y;
    if (X >= w || Y >= h) return;
    for (int oc = 0; oc < outc; oc++) {
        float s = 0;
        for (int ky = -1; ky <= 1; ky++)
            for (int kx = -1; kx <= 1; kx++) {
                int ix = min(max(X + kx, 0), w - 1);
                int iy = min(max(Y + ky, 0), h - 1);
                s += x[iy * w + ix] * f[oc * 9 + (ky + 1) * 3 + (kx + 1)];
            }
        y[oc * w * h + Y * w + X] = fmaxf(s, 0.0f);
    }
}

__global__ void maxPoolKernel(float *in, float *out, int w, int h, int c) {
    int X = blockIdx.x * TILE + threadIdx.x;
    int Y = blockIdx.y * TILE + threadIdx.y;
    if (X >= w / POOL || Y >= h / POOL) return;
    for (int ch = 0; ch < c; ch++) {
        float m = -1e9;
        for (int py = 0; py < POOL; py++)
            for (int px = 0; px < POOL; px++) {
                int ix = X * POOL + px, iy = Y * POOL + py;
                m = fmaxf(m, in[(ch * h + iy) * w + ix]);
            }
        out[(ch * (h / POOL) + Y) * (w / POOL) + X] = m;
    }
}

__global__ void denseSigmoid(float *in, float *W, float *b, float *out, int len) {
    float s = *b;
    for (int i = 0; i < len; i++) s += in[i] * W[i];
    *out = 1.0f / (1.0f + expf(-s));
}

__global__ void binaryCrossEntropy(float *pred, int label, float *loss) {
    *loss = -label * logf(*pred + 1e-7f) - (1 - label) * logf(1 - *pred + 1e-7f);
}

__global__ void adamUpdate(float *W, float *m, float *v, float *grad, int n, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1 - BETA2) * grad[i] * grad[i];
        float mhat = m[i] / (1 - BETA1);
        float vhat = v[i] / (1 - BETA2);
        W[i] -= lr * mhat / (sqrtf(vhat) + EPS);
    }
}

// ---------------- Helper ----------------
int load_images(const char *dir, char paths[][512], int max) {
    DIR *d; struct dirent *ent; int c = 0;
    if ((d = opendir(dir))) {
        while ((ent = readdir(d)) && c < max) {
            if (ent->d_type == DT_REG) {
                if (strstr(ent->d_name, ".jpg") || strstr(ent->d_name, ".png"))
                    snprintf(paths[c++], 512, "%s/%s", dir, ent->d_name);
            }
        }
        closedir(d);
    }
    return c;
}

// ---------------- Main ----------------
int main() {
    printf("\n--------------------------------------------\n");
    printf(" CUDA CNN Training (PetImages Cat vs Dog)\n");
    printf("--------------------------------------------\n");

    char paths[MAX_IMAGES][512];
    int n = 0;
    n = load_images("PetImages/Cat", paths, n, MAX_IMAGES);
    n = load_images("PetImages/Dog", paths, n, MAX_IMAGES);
    if (n == 0) { printf("❌ No images found.\n"); return 0; }

    int trainN = (int)(n * TRAIN_RATIO);
    int valN = n - trainN;
    printf("✅ Loaded %d images (Train %d, Validation %d)\n\n", n, trainN, valN);

    int w = IMG_W, h = IMG_H, size = w * h;
    int len = (w / POOL) * (h / POOL) * FILTERS;

    float *d_filter; cudaMalloc(&d_filter, FILTERS * 9 * sizeof(float));
    float *h_filter = (float*)malloc(FILTERS * 9 * sizeof(float));
    for (int i = 0; i < FILTERS * 9; i++) h_filter[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    cudaMemcpy(d_filter, h_filter, FILTERS * 9 * sizeof(float), cudaMemcpyHostToDevice);

    float *W, *b, *m, *v, *grad, *d_pred, *d_loss;
    cudaMalloc(&W, len * sizeof(float));
    cudaMalloc(&b, sizeof(float));
    cudaMalloc(&m, len * sizeof(float));
    cudaMalloc(&v, len * sizeof(float));
    cudaMalloc(&grad, len * sizeof(float));
    cudaMalloc(&d_pred, sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    float *hW = (float*)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++) hW[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    cudaMemcpy(W, hW, len * sizeof(float), cudaMemcpyHostToDevice);
    float hb = 0; cudaMemcpy(b, &hb, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);

    printf("🚀 Training started...\n\n");

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        float totalLoss = 0.0f, totalAcc = 0.0f;
        float valLoss = 0.0f, valAcc = 0.0f;

        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int idx = 0; idx < trainN; idx++) {
            int label = (idx >= trainN / 2) ? 1 : 0;
            int ww, hh, ch;
            unsigned char *img = stbi_load(paths[idx], &ww, &hh, &ch, 1);
            if (!img) continue;

            unsigned char *d_img;
            float *d_norm, *d_conv, *d_pool;
            cudaMalloc(&d_img, size);
            cudaMalloc(&d_norm, size * sizeof(float));
            cudaMalloc(&d_conv, FILTERS * size * sizeof(float));
            cudaMalloc(&d_pool, FILTERS * (size / 4) * sizeof(float));
            cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

            normalizeKernel<<<(size + 255) / 256, 256>>>(d_img, d_norm, size);
            convKernel<<<grid, block>>>(d_norm, d_filter, d_conv, w, h, FILTERS);
            maxPoolKernel<<<grid, block>>>(d_conv, d_pool, w, h, FILTERS);
            denseSigmoid<<<1, 1>>>(d_pool, W, b, d_pred, len);
            binaryCrossEntropy<<<1, 1>>>(d_pred, label, d_loss);

            float loss, pred;
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);

            totalLoss += loss;
            if ((pred > 0.5f) == label) totalAcc += 1.0f;

            float g = (pred - label);
            float *hgrad = (float*)malloc(len * sizeof(float));
            for (int i = 0; i < len; i++) hgrad[i] = g;
            cudaMemcpy(grad, hgrad, len * sizeof(float), cudaMemcpyHostToDevice);
            adamUpdate<<<(len + 255) / 256, 256>>>(W, m, v, grad, len, LR);
            free(hgrad);

            cudaFree(d_img); cudaFree(d_norm); cudaFree(d_conv); cudaFree(d_pool);
            stbi_image_free(img);
        }

        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);

        // Validation
        for (int idx = trainN; idx < n; idx++) {
            int label = (idx >= trainN + valN / 2) ? 1 : 0;
            int ww, hh, ch;
            unsigned char *img = stbi_load(paths[idx], &ww, &hh, &ch, 1);
            if (!img) continue;

            unsigned char *d_img;
            float *d_norm, *d_conv, *d_pool;
            cudaMalloc(&d_img, size);
            cudaMalloc(&d_norm, size * sizeof(float));
            cudaMalloc(&d_conv, FILTERS * size * sizeof(float));
            cudaMalloc(&d_pool, FILTERS * (size / 4) * sizeof(float));
            cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

            normalizeKernel<<<(size + 255) / 256, 256>>>(d_img, d_norm, size);
            convKernel<<<grid, block>>>(d_norm, d_filter, d_conv, w, h, FILTERS);
            maxPoolKernel<<<grid, block>>>(d_conv, d_pool, w, h, FILTERS);
            denseSigmoid<<<1, 1>>>(d_pool, W, b, d_pred, len);
            binaryCrossEntropy<<<1, 1>>>(d_pred, label, d_loss);

            float loss, pred;
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pred, d_pred, sizeof(float), cudaMemcpyDeviceToHost);
            valLoss += loss;
            if ((pred > 0.5f) == label) valAcc += 1.0f;

            cudaFree(d_img); cudaFree(d_norm); cudaFree(d_conv); cudaFree(d_pool);
            stbi_image_free(img);
        }

        printf("Epoch %d/%d — %.1fs %.3fms/step — accuracy: %.4f — loss: %.4f — val_accuracy: %.4f — val_loss: %.4f\n",
               epoch, EPOCHS, ms / 1000.0f, ms / trainN,
               totalAcc / trainN, totalLoss / trainN, valAcc / valN, valLoss / valN);

        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n✅ Training complete!\n");
    cudaFree(W); cudaFree(b); cudaFree(m); cudaFree(v);
    cudaFree(grad); cudaFree(d_filter); cudaFree(d_pred); cudaFree(d_loss);
    free(h_filter); free(hW);
    return 0;
}

```






# CPU vs GPU Image Processing and Training Pipeline (TACC - Cats and Dogs CNN)

This document describes each step in both the CPU and GPU (CUDA Shared-Memory) workflows for processing and training on 2,800 cat and dog images on TACC.  
It shows what happens in each stage, what code handles it, and what files are produced.

---

## Step-by-Step Process

| Stage # | Pipeline Step / Purpose | CPU Process | GPU (CUDA Shared-Memory) | Output / Result Files |
|----------|-------------------------|--------------|---------------------------|------------------------|
| 1 | Dataset Upload and Setup | Upload `cats_dogs/` → `cats/` + `dogs/` subfolders to TACC home directory. | Same folder structure used by CUDA code. | `/home1/<user>/cats_dogs/...` |
| 2 | Split Dataset (Training / Testing) | In C code, read image list and split: 80% training, 20% testing. | GPU code reads same list and stores training/testing indices. | `train_list.txt`, `test_list.txt` |
| 3 | Load and Resize Images | Sequential loop using `stb_image.h` to load and resize each image to 150×150. | Host loop + GPU preallocation (`cudaMalloc`) to prepare all images. | Raw pixel buffers in CPU and GPU memory. |
| 4 | Normalization (0–255 → 0–1) | Nested for-loops divide each pixel by 255.0. | `normalizeKernel<<<(N+255)/256,256>>>` normalizes all pixels in parallel. | Normalized float arrays. |
| 5 | Convolution + ReLU Feature Extraction | Triple nested loops over height × width × kernel. | `conv2d_shared<<<grid,block>>>` uses shared memory tiles with `__syncthreads()` for faster access. | Feature maps for each filter. |
| 6 | Max Pooling (2×2) | Sequential loop searches for maximum value in each pooling region. | `maxPool_shared<<<grid,block>>>` computes max in parallel per tile. | Downsampled feature maps. |
| 7 | Dense + Sigmoid Output Layer | Standard matrix–vector multiplication with sigmoid activation. | `denseSigmoid<<<1,classes>>>` computes all outputs in parallel on GPU. | Class probabilities (Cat, Dog). |
| 8 | Training Phase (Forward + Backward) | CPU updates weights using nested loops and Adam formula. | GPU runs `adamUpdate<<<grid,block>>>` to update all weights simultaneously. | Updated weights (`.bin` files). |
| 9 | Testing / Inference Phase | Sequential evaluation of each test image. | GPU runs mini-batch inference using CUDA streams for concurrency. | Accuracy and prediction metrics. |
| 10 | Timing and Profiling | Use `clock()` to measure time per stage. | Use `cudaEventRecord()` and `cudaEventElapsedTime()` for GPU timings. | Console output and CSV timing summary. |
| 11 | Save Output Images (Feature Maps) | Sequentially write `cpu_output_xxxx.png`. | Parallel kernel saves `gpu_output_xxxx.png` (first 15 or all 2800 images). | `/cpu_outputs/`, `/gpu_outputs/` |
| 12 | Create Result Tables | Append per-image results to `cpu_results.csv` (filename, stage times, total, prediction). | Append per-image results to `gpu_results.csv` with per-kernel times. | Two CSV result files for comparison. |
| 13 | Compute Speedup | Calculate average time: CPU sum / 2800. | Calculate average time: GPU sum / 2800. | `Speedup = CPU / GPU` (≈ 40–60×). |
| 14 | Download Results | `scp /home1/<user>/cpu_results.csv .` | `scp /home1/<user>/gpu_outputs/*.png .` | Local comparison and visualization. |
| 15 | Visualization / Report | Plot CPU vs GPU time per stage in Python or Excel. | Same data plotted to show acceleration advantage. | Final benchmark report. |

---

## Estimate Numerical Summary (After 2,800 Images)

| Metric | CPU (seconds) | GPU (seconds) | Speedup |
|---------|----------------|----------------|----------|
| Normalization | 45.0 | 0.7 | 64× |
| Convolution (3×3) | 270.0 | 5.5 | 49× |
| Pooling | 60.0 | 1.2 | 50× |
| Dense + Sigmoid | 12.0 | 0.3 | 40× |
| **Total (2,800 images)** | **387.0 s** | **7.7 s** | **≈ 50× faster on GPU** |


