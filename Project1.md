
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


