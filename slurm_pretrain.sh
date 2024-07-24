#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1


set -ex


CVD=(${CUDA_VISIBLE_DEVICES//,/ })
N_GPUS=${#CVD[@]}

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
ADDR=${HOSTS[0]}
MASTER_IP=$(nslookup $MASTER_ADDR | awk '/^Address: / { print $2 }')

if [ -n "$RESUME_EPOCH" ]; then
  RESUME="--resume $OUT/checkpoint-${RESUME_EPOCH}.pth"
fi

BASE_PORT=29500


# Function to check if a port is free
is_port_free() {
    local port=$1
    ! ss -ltn | grep -q ":$port "
    #! grep -q ":$port" /proc/net/tcp
    #nc -w 5 -z $ADDR $port &>/dev/null
    return $?
}

# Loop to find a free port
while true; do
    if is_port_free $BASE_PORT; then
        echo "Found free port: $BASE_PORT"
        break
    else
        ((BASE_PORT++))
    fi
done

for N in `seq 0 $(($SLURM_NNODES-1))`;
do
    srun -N1 --nodelist=${HOSTS[N]} --cpus-per-task $SLURM_CPUS_PER_TASK --mem-per-cpu=6G \
	  conda run --no-capture-output  -n pt2 \
	    torchrun --nproc_per_node $N_GPUS --nnodes $SLURM_NNODES --node_rank $N --master-addr $ADDR --master-port $BASE_PORT \
	      main_pretrain.py --batch_size $BS --model "mae_${MODEL}" --input_size $INPUT_SIZE --mask_ratio 0.75 --epochs $EPOCHS --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --num_workers 6 --data_path $DATA_PATH --lamb $UMAE_LAMBDA --lpred_decoder_depth $LPRED_DECODER_DEPTH --lpred_decoder_heads $LPRED_DECODER_HEADS --umae_reg $REG --lpred_loss $LPRED_LOSS --lpred_lambda $LPRED_LAMBDA $LPRED_DETACH --lpred_decoder_arch $LDA -lci $LCI $LLNT $NPL $RESUME --output_dir $OUT --log_dir $OUT --amp $AMP & 
# --norm_pix_loss

done

wait

