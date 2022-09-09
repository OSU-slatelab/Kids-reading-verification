for i in 0 2 4 6 8
do
    for j in 0.9
    do
        python -u main.py \
                --patience 35 \
                --train-path "/data/data25/scratch/sunderv/rraces/folds_word/train_${i}.csv" \
                --valid-path "/data/data25/scratch/sunderv/rraces/folds_word/valid_${i}.csv" \
                --test-path "/data/data25/scratch/sunderv/rraces/folds_word/test_${i}.csv" \
                --logging-file "logs/slu_libri960_125k_classify_wordbyword2.log" \
                --tokenizer-path "tokenizers/librispeech.json" \
                --dict-path "/data/data24/scratch/sunderv/saved_models/kids_librispeech960_pretrain_las_steps_125000.pt" \
                --save-path "" \
                --batch-size 32 \
                --lr 0.0001 \
                --nclasses 7 \
                --norm-epoch 3 \
                --pyr-layer 3 \
                --nlayer 6 \
                --nspeech-feat 80 \
                --sample-rate 16000 \
                --multi-gpu \
                --word-by-word \
                --cuda \
                --seed 1111
    done
done
