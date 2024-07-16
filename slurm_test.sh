#!/bin/bash
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1


set -ex

conda run --no-capture-output -n pt2 python main_linprobe_v2.py --batch_size 2048 --model $MODEL --input_size $INPUT_SIZE --data_path $DATA_PATH --aug_every 15 --n_last_layers 1 --shuffle_subsets 1  --cls_features cls --agg_method rep --output_dir ${OUT}  --finetune $CKPT

