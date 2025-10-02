in CPU(host)
int n = 8;
float *arr = (float*)malloc(n * sizeof(float));


in GPU(device)
float *arr_d;
cudaMalloc((void**)&arr_d, n * sizeof(float));

cudaFree(arr_d);= prevents nenory leaks;

int arr[3] = {10, 20, 30};
int *ptr = &arr;   // ???

int *ptr = &arr[0]; 
int *ptr = arr; 

send to TACC
sbatch abc

CUDA= Compute Device Unified Artictecture

Parallel processing is the method of dividing a large computational task
into smaller sub-tasks and executing them simultaneously
on multiple processors (or cores), instead of doing them one after another on a single processor.

Data parallelism → Same operation performed on chunks of data(e.g., matrix multiplication on GPU).

Task parallelism → Different tasks executed at the same time(one core sorts numbers, another core searches).

#define PI 3.1416 

int arr[3] = {"A", "B", "C"};   // wrong
int arr[3] = {'A', 'B', 'C'};   // arr = {65, 66, 67}

char str[5] = {H, e, l, l, o};   // wrong
char str[5] = {'H', 'e', 'l', 'l', 'o'};

int arr[];   // ERROR → compiler doesn’t know size
int arr[5];   // compiler knows size = 5
int arr[] = {1, 2, 3, 4};   // size deduced = 4

int x = 3;
int y;
y=++x (Preincrement)=4,4 
y=x++ (post increment)= 3,4(assign value of y ,increase in x)

Largest element in array
int arr[] = {12, 45, 7, 89, 34};
    int n = 5;  // array size
    int max = arr[0];                     // assume first element is largest

    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];   // update max if current element is larger
        }
    }
    
    ##
    
    arr → |10|20|30|
       ^
       arr, &arr[0], ptr (all point here)

&arr → points to the whole block (type int (*)[3])


// Pointer variable
#include <stdio.h>

int main() {
    int num = 42;
    int *ptr = &num;   // pointer variable

    printf("num = %d\n", num);
    printf("&num = %p\n", (void*)&num);
    printf("ptr = %p\n", (void*)ptr);
    printf("*ptr = %d\n", *ptr);

    return 0;
}
num = 42
&num = 0x7ffee4b7a9ac
ptr = 0x7ffee4b7a9ac
*ptr = 42


int vals[] = {4, 7, 11};
*vals;   // gives 4
*vals → dereferencing that pointer → value of first element → 4

int *valptr = vals;  // pointer points to first element
printf("%d\n", valptr[1]);  // displays 7
Accesses second element of the array (vals[1])

Multicore Processors:
A multicore processor has two or more CPU cores on a single chip.
Each core can independently execute instructions, so multiple tasks can run in parallel

Manycore Processors:
A manycore processor has a large number of simpler cores, often tens or hundreds.
Designed for massively parallel computations rather than complex single-thread performance













