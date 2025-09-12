# GPU & CUDA Basics

This document explains **fundamental concepts** used in computing, graphics, and CUDA programming.  
We’ll go step by step, starting with **bandwidth/latency/throughput**, then **FLOPS**, **floating-point formats**, and finally **CUDA concepts** such as **host/device memory, kernels, threads, and parallelism**.

---

## 1. Bandwidth, Latency, Throughput

### Latency
- **Definition:** Time it takes for a single piece of data to travel from source to destination.  
- **Analogy:** Mail delivery time (how long a letter takes to reach your friend).  
- **Example:** Clicking a website link → delay before it loads.

### Bandwidth
- **Definition:** Maximum amount of data that can be transmitted per second.  
- **Analogy:** The width of a highway (more lanes = more cars at once).  
- **Example:** A 1 Gbps internet connection can handle more data than a 100 Mbps one.

### Throughput
- **Definition:** Actual amount of data delivered per second.  
- **Analogy:** Cars that **actually arrive** per hour on a highway (less than full capacity due to traffic).  
- **Example:** You pay for 1 Gbps, but speed test shows 700 Mbps → that’s your throughput.

---

## 2. FLOPS (Floating-Point Operations Per Second)

- **Definition:** How many decimal math calculations a processor can do per second.  
- **Units:**
  - **GFLOPS/sec** → billions of operations per second  
  - **TFLOPS/sec** → trillions of operations per second  

### Examples
| Device                | Performance |
|-----------------------|-------------|
| CPU (desktop)         | 100–500 GFLOPS |
| Gaming GPU            | ~10 TFLOPS |
| Supercomputer         | 100s of PFLOPS (1 PFLOP = 1000 TFLOPS) |

- **Analogy:** Counting raindrops per second.  
  - 1 GFLOPS = 1 billion drops/sec  
  - 1 TFLOPS = 1 trillion drops/sec  

---

## 3. Floating-Point Formats (FP4, FP32, FP64)

- **FP4 (4-bit):**
  - Very low precision, used in AI inference.  
  - **Analogy:** Rough sketch (fast but not accurate).

- **FP32 (32-bit, single precision):**
  - Balance between accuracy and speed, used in graphics & AI training.  
  - **Analogy:** A decent drawing with 7-digit accuracy.

- **FP64 (64-bit, double precision):**
  - High precision, used in simulations, finance, scientific work.  
  - **Analogy:** A ruler that measures down to nanometers.

| Format | Bits | Use Case                     | Analogy         |
|--------|------|-----------------------------|----------------|
| FP4    | 4    | AI inference, low-precision | Rough sketch   |
| FP32   | 32   | Graphics, AI training       | 7-digit detail |
| FP64   | 64   | Science, finance, physics   | Nano ruler     |

---

## 4. What is CUDA?

- **CUDA = Compute Unified Device Architecture**  
- Developed by **NVIDIA** to let GPUs be used for **general-purpose computing**, not just graphics.  
- Instead of the CPU doing everything sequentially, CUDA lets **thousands of GPU cores** run tasks in parallel.

### Analogy
Painting 10,000 houses:  
- CPU = 4 workers → very slow.  
- GPU with CUDA = 4,000 workers → done super fast.

### Applications
- Machine learning (TensorFlow, PyTorch)  
- 3D rendering  
- Weather simulations  
- Cryptography, finance  

---

## 5. Memory Model in CUDA

- **Host Memory:** CPU memory (RAM).  
  - Analogy: Your **office desk** with documents.  

- **Device Memory (Global Memory):** GPU memory (VRAM).  
  - Analogy: A **special high-speed cabinet** for GPU workers.  

### Data Flow
```
CPU (Host) --[Host Memory]--> Copy Data --> GPU (Device) --[Device Memory]
                  \                              /
                   ----> Launch Kernel -----------/
```

---

## 6. Kernels & Threads

### Kernel
- **Definition:** A function that runs on the GPU, executed by many threads in parallel.  
- **Analogy:** Instructions given to all workers: *"Paint the wall red!"*

### Threads
- **Definition:** The smallest execution unit in CUDA. Each thread works on one part of the data.  
- **Analogy:** Each worker paints one house.

### Thread Indexing Example
```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```
- `threadIdx.x` → Thread ID inside its block
- `blockIdx.x` → Block ID in the grid
- `blockDim.x` → Threads per block

---

## 7. Parallelism

### Data Parallelism
- Same task on multiple data simultaneously.  
- **Analogy:** 100 painters painting 100 identical houses.

### Task Parallelism
- Different tasks at the same time.  
- **Analogy:** Painters paint, bricklayers build, electricians install wiring.

---

## 8. Example CUDA Program
### Array Squaring
```cpp
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Kernel: Each thread squares one element
__global__ void squareArray(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = arr[idx] * arr[idx];
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);

    // Host memory
    int h_arr[n];
    for (int i = 0; i < n; i++) h_arr[i] = i+1;

    // Device memory
    int *d_arr;
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Kernel launch: 1 block, 10 threads
    squareArray<<<1, 10>>>(d_arr, n);

    // Copy result back
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < n; i++) cout << h_arr[i] << " ";
    cout << endl;

    cudaFree(d_arr);
    return 0;
}
```

**Output:**  
1 4 9 16 25 36 49 64 81 100
