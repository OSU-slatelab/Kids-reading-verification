for i in 0 2 4 6 8
do
    for j in 0.9
    do
        python -u main.py \
                --patience 35 \
                --train-path "/data/data25/scratch/sunderv/rraces/folds/train_${i}.csv" \
                --valid-path "/data/data25/scratch/sunderv/rraces/folds/valid_${i}.csv" \
                --test-path "/data/data25/scratch/sunderv/rraces/folds/test_${i}.csv" \
                --logging-file "logs/slu_libri960_125k_multiclass_accum.log" \
                --tokenizer-path "tokenizers/librispeech.json" \
                --dict-path "/data/data24/scratch/sunderv/saved_models/kids_librispeech960_pretrain_las_steps_125000.pt" \
                --batch-size 4 \
                --lr 0.0004 \
                --con-wt $j \
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
done
