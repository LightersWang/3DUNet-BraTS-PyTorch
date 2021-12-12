python train_brats2021.py \
    --model_type mencoder_plain_unet \
    --data_type brats2021 \
    --num_modality 4 \
    --data_root data/brats21/ \
    --cases_split data/brats2021_split.csv \
    --patch_size 128 \
    --gpus 0 1 \
    --num_workers 6 \
    --epochs 100 \
    --train_batch_size 4 \
    --save_root data/experiments \
    --input_channels 1 \
    --num_classes 3 \
    --deep_supervision \
    --lr 1e-2 \
    --optim adamw \
    --scheduler warmup_cosine \
    --amp

