python -u main.py \
        --nsteps 700000 \
        --save-after 125000 \
        --steps-done 0 \
        --log-after 500 \
        --val-after 2000 \
        --train-path "/data/data25/scratch/sunderv/librispeech/train_full_960.csv" \
        --valid-path "/data/data25/scratch/sunderv/librispeech/dev_other.csv" \
        --test-path "/data/data25/scratch/sunderv/librispeech/test_other.csv" \
        --logging-file "logs/librispeech960_pretrain_las.log" \
        --tokenizer-path "tokenizers/librispeech.json" \
        --dict-path "" \
        --save-path "/data/data24/scratch/sunderv/saved_models/kids_librispeech960_pretrain_las" \
        --batch-size 32 \
        --lr 0.0001 \
        --norm-epoch 3 \
        --pyr-layer 3 \
        --nlayer 6 \
        --nspeech-feat 80 \
        --sample-rate 16000 \
        --multi-gpu \
        --cuda \
        --pretrain \
        --seed 1111
