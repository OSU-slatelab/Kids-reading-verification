for i in 0 2 4 6 8 
do
    python -u main.py \
            --patience 10 \
            --train-path "/data/data25/scratch/sunderv/rraces/folds/train_${i}.csv" \
            --valid-path "/data/data25/scratch/sunderv/rraces/folds/valid_${i}.csv" \
            --test-path "/data/data25/scratch/sunderv/rraces/folds/test_${i}.csv" \
            --logging-file "logs/slu_lib100pt_cotrain.log" \
            --tokenizer-path "tokenizers/librispeech.json" \
            --dict-path "/data/data24/scratch/sunderv/saved_models/kids_librispeech100_pretrain_las_best.pt" \
            --batch-size 32 \
            --lr 0.0001 \
            --asr-wt 0.1 \
            --nclasses 2 \
            --norm-epoch 3 \
            --listener-layer 3 \
            --nspeech-feat 80 \
            --cuda \
            --seed 1111
done
