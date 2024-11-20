#!/bin/bash -l
#SBATCH --ntasks-per-node=1


set -ex


CVD=(${CUDA_VISIBLE_DEVICES//,/ })
N_GPUS=${#CVD[@]}

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
ADDR=${HOSTS[0]}
MASTER_IP=$(nslookup $MASTER_ADDR | awk '/^Address: / { print $2 }')


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

module load ML-bundle
. ~/pt2/bin/activate

export OMP_NUM_THREADS=1

### Plgrid additions

#export NCCL_BUFFSIZE=$((4*1024*1024))
#export NCCL_NET_SHARED_BUFFERS=0
#export NCCL_CUMEM_ENABLE=0
#export CXI_FORK_SAFE="1"
#export FI_CXI_DISABLE_CQ_HUGETLB="1"
#export FI_MR_CACHE_MONITOR="userfaultfd"
#export CXI_FORK_SAFE_HP="1"
#export NCCL_CROSS_NIC="1"
#export FI_CXI_DISABLE_HOST_REGISTER="1"
#export FI_HMEM_CUDA_USE_GDRCOPY=1
#export FI_HMEM_CUDA_ENABLE_XFER=1

export ADDR=`hostname -i | cut -d' ' -f1`
export MASTER=$ADDR
export MASTER_ADDR=$MASTER
export MASTER_PORT=7779

#export NCCL_SOCKET_IFNAME=bond0

#export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3


###:

#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


#ip a

#srun torchrun --nproc-per-node=4 --nnodes=$SLURM_NNODES --rdzv-id=$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER:7779 \
#    main_linprobe.py --amp $AMP --batch_size $BS --model ${MODEL}   --num_workers 16 --data_path $DATA_PATH  --accum_iter $AIT --optimizer $OPT --cls_features $CLSFT --output_dir $OUT --finetune $CKPT --dataloader_affinity_hack --abmilp_act $ABMILP_ACT --abmilp_sa $ABMILP_SA --abmilp_depth $ABMILP_DEPTH --abmilp_cond $ABMILP_COND --suffix $SUFFIX 

for N in `seq 0 $(($SLURM_NNODES-1))`;
do
    #tracepath $MASTER_ADDR #:$MASTER_PORT
    #! ss -ltn | grep -q "$MASTER_ADDR:$MASTER_PORT "
    #ping $MASTER -c 4 # &
    #srun -N1 --nodelist=${HOSTS[N]} --cpus-per-task $SLURM_CPUS_PER_TASK --gpus-per-node $N_GPUS  torchrun --nproc_per_node $N_GPUS --nnodes $SLURM_NNODES --rdzv-id=$SLURM_JOB_ID --rdzv-endpoint=$MASTER:7779 --rdzv-backend=c10d simple_dist.py &
    
    srun -N1 --nodelist=${HOSTS[N]} --cpus-per-task $SLURM_CPUS_PER_TASK \
    	    torchrun --nproc_per_node $N_GPUS --nnodes $SLURM_NNODES --rdzv-id=$SLURM_JOB_ID --rdzv-endpoint=$MASTER:7779 --rdzv-backend=c10d \
    	      main_linprobe.py --amp $AMP --batch_size $BS --model ${MODEL}   --num_workers 16 --data_path $DATA_PATH  --epochs $EPOCHS --accum_iter $AIT --optimizer $OPT --cls_features $CLSFT --output_dir $OUT --finetune $CKPT --dataloader_affinity_hack --abmilp_act $ABMILP_ACT --abmilp_sa $ABMILP_SA --abmilp_depth $ABMILP_DEPTH --abmilp_cond $ABMILP_COND --abmilp_content $ABMILP_CONTENT --return_block $RETURN_BLOCK --suffix $SUFFIX & 
# --norm_pix_loss
# --finetune $CKPT
# --rdzv-backend=c10d #--node_rank $N --master-addr $ADDR --master-port $BASE_PORT \
done

wait

