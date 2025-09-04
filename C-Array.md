
# Arrays in C – Complete Guide

## Table of Contents
1. [What is an Array?](#what-is-an-array)
2. [Accessing Array Elements](#accessing-array-elements)
3. [Inputting and Displaying Array Contents](#inputting-and-displaying-array-contents)
4. [Array Initialization](#array-initialization)
5. [Processing Array Contents](#processing-array-contents)
6. [Arrays as Function Arguments](#arrays-as-function-arguments)
7. [Two-Dimensional Arrays](#two-dimensional-arrays)
8. [Passing Two-Dimensional Arrays to Functions](#passing-two-dimensional-arrays-to-functions)

---

## What is an Array?
An array is a **collection of elements of the same data type** stored in **contiguous memory locations**.

- **Real-world example:** storing **daily temperatures** of a week.

```c
#include <stdio.h>
int main() {
    int temperatures[7]; // temperatures for a week
    return 0;
}
```

---

## Accessing Array Elements
- Arrays use **0-based indexing**.

```c
int numbers[5] = {10, 20, 30, 40, 50};
printf("First element: %d\n", numbers[0]);  // 10
printf("Third element: %d\n", numbers[2]);  // 30
```

---

## Inputting and Displaying Array Contents
```c
#include <stdio.h>
int main() {
    int scores[5];
    for(int i=0; i<5; i++){
        printf("Enter score %d: ", i+1);
        scanf("%d", &scores[i]);
    }

    printf("Entered scores are: ");
    for(int i=0; i<5; i++){
        printf("%d ", scores[i]);
    }
    return 0;
}
```
**Real-world scenario:** Collecting **exam scores** from 5 students.

---

## Array Initialization
```c
int a[5] = {1, 2, 3, 4, 5};    // fully initialized
int b[5] = {10, 20};           // rest are 0 → {10, 20, 0, 0, 0}
int c[] = {7, 8, 9};           // size determined by compiler
```

---

## Processing Array Contents

**Example 1: Sum and Average of an Array**
```c
int arr[5] = {2, 4, 6, 8, 10};
int sum = 0;
for(int i=0; i<5; i++){
    sum += arr[i];
}
printf("Sum = %d, Average = %.2f", sum, (float)sum/5);
```
**Real-world:** Calculating **total and average sales** for 5 days.

**Example 2: Finding Maximum and Minimum**
```c
int arr[5] = {23, 45, 12, 67, 34};
int max = arr[0], min = arr[0];
for(int i=1; i<5; i++){
    if(arr[i] > max) max = arr[i];
    if(arr[i] < min) min = arr[i];
}
printf("Max = %d, Min = %d", max, min);
```

---

## Arrays as Function Arguments
```c
#include <stdio.h>
void printArray(int arr[], int size){
    for(int i=0; i<size; i++){
        printf("%d ", arr[i]);
    }
}

int main(){
    int nums[5] = {1,2,3,4,5};
    printArray(nums, 5);
    return 0;
}
```
**Real-world:** Passing **monthly expenses** to a function for analysis.

---

## Two-Dimensional Arrays
- Like a **table** with **rows and columns**.
```c
#include <stdio.h>
int main(){
    int matrix[2][3] = {{1,2,3}, {4,5,6}};
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    return 0;
}
```
**Real-world:** Storing **grades of 2 students in 3 subjects**.

---

## Passing Two-Dimensional Arrays to Functions
```c
#include <stdio.h>

void printMatrix(int arr[][3], int rows){
    for(int i=0; i<rows; i++){
        for(int j=0; j<3; j++){
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
}

int main(){
    int matrix[2][3] = {{1,2,3}, {4,5,6}};
    printMatrix(matrix, 2);
    return 0;
}
```
**Real-world:** Printing a **class performance matrix** for rows = students, columns = subjects.

---

### Notes / Tips
- Arrays are **fixed in size** (static arrays). Use **pointers** or **dynamic memory** for flexible sizes.  
- For 2D arrays, always **specify number of columns** when passing to functions.  
- Arrays are **passed by reference**, so changes in functions affect the original array.

