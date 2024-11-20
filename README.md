# Supplementary code for "Beyond [cls]: Exploring the true potential of Masked Image Modeling representations"

These scripts are based on the official MAE codebase.

Evaluating  MAE ViT-B + AbMILP on ImageNet-1k classification:
```
torchrun --nproc_per_node 4 --nnodes 1 --rdzv-id=$RDZV_ID --rdzv-endpoint=$HOST:$PORT --rdzv-backend=c10d \
    main_linprobe.py --amp bfloat16  --num_workers 16  --dataloader_affinity_hack \
        --epochs 90 --accum_iter 2 --optimizer lars --batch_size 2048 \
        --model vit_base_patch16 --finetune vit_base_patch16_224.mae \
        --data_path $IMAGENET_PATH --output_dir $OUT_DIR  \
        --cls_features abmilp  --abmilp_act relu --abmilp_sa none \
        --abmilp_depth 1 --abmilp_cond none --abmilp_content patch 
         
```

Calculating the attention statistics of the MAE (WANDB required:)
```
export WANDB_API_KEY=...
export WANDB_PROJECT=...
export WANDB_ENTITY=...

python main_attention_stats.py --batch_size 512 --num_workers 16 \
    --model vit_base_patch16 --finetune vit_base_patch16_224.mae --input_size 224 \
    --data_path  $IMAGENET_PATH --output_dir $OUT_DIR
```
