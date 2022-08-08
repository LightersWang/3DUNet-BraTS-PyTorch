CUDA_VISIBLE_DEVICES=1 python train_brats2021.py \
    --comment train \
    --gpus 0 \
    --seed 1000 \
    --num_workers 8 \
    --save_root exps \
    --dataset brats2021 \
    --cases_split data/split/brats2021_split_fold0.csv \
    --input_channels 4 \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-3 \
    --optim adamw \
    --wd 1e-4 \
    --scheduler warmup_cosine \
    --warmup_epochs 5 \
    --num_classes 3 \
    --unet_arch multiencoder_unet \
    --block plain \
    --channels_list 32 64 128 256 320 320 \
    --deep_supervision \
    --ds_layer 4 \
    --patch_size 128 \
    --pos_ratio 1.0 \
    --neg_ratio 1.0 \
    --save_model \
    --save_pred \
    --eval_freq 10 \
    --print_freq 5