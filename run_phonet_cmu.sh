for i in 0 2 4 6 8
do
    for j in 0.9
    do
        python -u main.py \
                --patience 35 \
                --train-path "/data/data25/scratch/sunderv/rraces/forced_aligned/folds_cmu/train_${i}.csv" \
                --valid-path "/data/data25/scratch/sunderv/rraces/forced_aligned/folds_cmu/valid_${i}.csv" \
                --test-path "/data/data25/scratch/sunderv/rraces/forced_aligned/folds_cmu/test_${i}.csv" \
                --logging-file "logs/slu_libri960_125k_detect_phonet_cmu.log" \
                --tokenizer-path "tokenizers/librispeech.json" \
                --vocab-path "/data/data25/scratch/sunderv/rraces/forced_aligned/vocab_cmu.json" \
                --batch-size 32 \
                --lr 0.0001 \
                --nclasses 2 \
                --norm-epoch 3 \
                --nspeech-feat 80 \
                --sample-rate 16000 \
                --multi-gpu \
                --phonet \
                --cuda \
                --seed 1111
    done
done
