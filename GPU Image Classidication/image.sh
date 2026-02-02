#!/bin/bash
#SBATCH -J cuda_cnn
#SBATCH -o image_%j.out
#SBATCH -e image_%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ASC23018
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sadhikari0902@my.msutexas.edu

module purge
module load cuda/12.2

echo "-------------------------------------------"
echo " CUDA CNN Job Started"
echo " Time: $(date)"
echo " Host: $(hostname)"
echo " PWD : $(pwd)"
echo "-------------------------------------------"

echo "Compiling image.cu ..."
nvcc image.cu -o image.out -O2 2> compile_error.log

if [ ! -f image.out ]; then
    echo "Compilation failed!"
    echo "----------- NVCC ERROR LOG -----------"
    cat compile_error.log
    echo "--------------------------------------"
    exit 1
fi

echo "Compilation successful."
echo "-------------------------------------------"

echo "Running program..."
./image.out 2>&1 | tee output_${SLURM_JOB_ID}.txt

echo "Execution finished."
