#!/bin/bash
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1


set -ex


CVD=(${CUDA_VISIBLE_DEVICES//,/ })
N_GPUS=${#CVD[@]}

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR=${HOSTS[0]}
MASTER_IP=$(nslookup $MASTER_ADDR | awk '/^Address: / { print $2 }')

if [ -n "$RESUME_EPOCH" ]; then
  RESUME="--resume $OUT/checkpoint-${RESUME_EPOCH}.pth"
fi


for N in `seq 0 $(($SLURM_NNODES-1))`;
do
    srun -N1 --nodelist=${HOSTS[N]} --cpus-per-task $SLURM_CPUS_PER_TASK --mem-per-cpu=6G \
	  conda run --no-capture-output  -n pt2 \
	    torchrun --nproc_per_node $N_GPUS --nnodes $SLURM_NNODES --node_rank $N --master-addr $MASTER_IP  \
	      main_pretrain.py --batch_size $BS --model "mae_${MODEL}" --norm_pix_loss --mask_ratio 0.75 --epochs $EPOCHS --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --num_workers 32 --data_path $DATA_PATH --lamb 0.01 --umae_reg $REG --lpred_lambda $LPRED_LAMBDA --norm_pix_loss $RESUME --output_dir $OUT --log_dir $OUT --amp $AMP &

done

wait

