#!/bin/bash

num_gpus=4
ddp_file=ckpt/ddp_init
config=config/ddpm_cifar10.yaml
task=cifar10
for((i=0; i<$num_gpus; ++i)); do
{
  gpu_id=$i
  init_file=file://$(readlink -f $ddp_file)
  echo running on rank "$gpu_id"
  python train.py --config "$config" --task "$task" --ddp --rank "$gpu_id" --init_method "$init_file"  --world_size "$num_gpus"
} &
done
wait
