for i in 0 2 4 6 8
do
    python -u main.py \
        --patience 35 \
        --train-path "/data/data25/scratch/sunderv/rraces/folds_cmu_only/train_${i}.csv" \
        --valid-path "/data/data25/scratch/sunderv/rraces/folds_cmu_only/valid_${i}.csv" \
        --test-path "/data/data25/scratch/sunderv/rraces/folds_cmu_only/test_${i}.csv" \
        --logging-file "logs/slu_libri960_125k_classify_cmu_nopt_asr_mfw.log" \
        --tokenizer-path "tokenizers/librispeech.json" \
        --dict-path "/data/data24/scratch/sunderv/saved_models/kids_librispeech960_pretrain_las_steps_125000.pt" \
        --batch-size 32 \
        --lr 0.0001 \
        --asr-wt 0.5 \
        --nclasses 7 \
        --norm-epoch 3 \
        --pyr-layer 3 \
        --nlayer 6 \
        --nspeech-feat 80 \
        --sample-rate 16000 \
        --multi-gpu \
        --cuda \
        --seed 1111
done
