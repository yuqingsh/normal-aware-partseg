#!/bin/bash

#SBATCH --job-name=train_partseg
#SBATCH --partition=A800
#SBATCH --output=train_partseg_%j.out
#SBATCH --error=train_partseg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G

# --- 开始执行任务 ---
echo "========================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Requested GPUs: $SLURM_GPUS_ON_NODE"
echo "Requested CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Requested Memory: ${SLURM_MEM_PER_NODE}MB (Total for node)"
echo "Working directory: $(pwd)"
echo "========================================================"

# 1. 加载编译所需的模块
echo "--- Loading Modules ---"
module purge
module load gcc/11.4.0
module load cuda/11.8
export CUDA_HOME=${CUDA_HOME:-/share/apps/cuda-11.8}

# 2. 激活Conda环境
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then echo "Error initializing Conda hook"; exit 1; fi
conda activate point-bert

# GPU kNN
python train_partseg.py --normal