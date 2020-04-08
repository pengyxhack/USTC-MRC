#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python /home/pyx/USTC-MRC/examples/run_squad.py     --model_type bert \
    --model_name_or_path /home/pyx/bert_model/ro_bert_a_wwm_large_ext_pytorch \
    --do_train     --do_lower_case  --warmup_ratio 0.07  --gradient_accumulation_steps 16  --save_steps 300 \
    --weight_decay 0.01 \
    --train_file /home/mrc/data/dureader_robust-data/train.json \
    --predict_file /home/mrc/data/dureader_robust-data/dev.json \
    --learning_rate 3e-5  --num_train_epochs 3     --max_seq_length 384     --doc_stride 128 \
    --output_dir /home/pyx/models_output/roberta_large_wr0.07_ep3/ \
    --per_gpu_train_batch_size=2 \
    --overwrite_output_dir

sleep 60s

CUDA_VISIBLE_DEVICES=6 python /home/pyx/USTC-MRC/examples/run_squad.py     --model_type bert \
    --model_name_or_path /home/pyx/models_output/roberta_large_wr0.07_ep3/ \
    --do_eval     --eval_all_checkpoints   --do_lower_case \
    --predict_file /home/mrc/data/dureader_robust-data/dev.json \
    --max_seq_length 384     --doc_stride 128 \
    --output_dir /home/pyx/models_output/roberta_large_wr0.07_ep3/ \
    --train_file /home/mrc/data/dureader_robust-data/train.json \
    >/home/pyx/models_output/roberta_large_wr0.07_ep3/eval.log 2>&1
