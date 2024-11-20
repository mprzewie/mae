#!/bin/bash -l
#SBATCH --ntasks-per-node=1


set -ex

module load ML-bundle

which python

. ~/pt2/bin/activate

which python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



python main_linprobe_v2.py --batch_size 2048 --model $MODEL --input_size $INPUT_SIZE --data_path $DATA_PATH --aug_every 90 --n_last_layers 1 --shuffle_subsets 1  --cls_features cls --agg_method rep --output_dir ${OUT}  --finetune $CKPT

