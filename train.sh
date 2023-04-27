#!/bin/bash
#SBATCH --job-name=Bert_test
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%j.log
#SBATCH --partition=kshdexclu04
#SBATCH --exclusive
#SBATCH --gres=dcu:4


# load the environment


module switch compiler/rocm/dtk-22.04.2
source activate asr-1.10

#python -m torch.distributed.launch --nproc_per_node=4 run.py hparams/train.yaml --distributed_launch --distributed_backend='nccl' --auto_mix_prec --grad_accumulation_factor=4
./run.sh
# Output file is ...:         PythonTest.log

