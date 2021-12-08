#!/usr/bin/env bash
set -x
shopt -s extglob
PYTHONPATH="." CUDA_VISIBLE_DEVICES=2,3,4,5,9,12,14,15 benchmarker/cli/l5/train.py \
--model_name_or_path ${INIT_LARGE_MODEL_PATH}/ \
--relative_bias_args='[{"type":"1d"},{"type":"horizontal"},{"type":"vertical"}]' \
--dropout_rate 0.15 \
--label_smoothing 0 \
--model_type=t5  \
--data_dir ${DATASETS_ROOT}/WikiTableQuestions/${OCR_ENGINE}/train \
--val_data_dir ${DATASETS_ROOT}/WikiTableQuestions/${OCR_ENGINE}/dev \
--test_data_dir  ${DATASETS_ROOT}/WikiTableQuestions/${OCR_ENGINE}/test \
--gpus 8 \
--num_workers 16 \
--train_batch_size 1 \
--eval_batch_size 1 \
--accumulate_grad_batches 8 \
--max_epochs 30 \
--val_check_interval 0.20 \
--output_dir ${OUT_DIR}
--max_target_length 256 \
--eval_max_gen_length 256 \
--warmup_steps 100 \
--learning_rate 2e-4  \
--lr_scheduler constant \
--val_metric anls \
--do_train \
--do_predict \
--additional_data_fields doc_id label_name \
--segment_levels tokens pages \
--optimizer adamw \
--weight_decay 1e-5 \
--adam_epsilon 1e-8 \
--output_dir ${OUT_DIR}
--seed 4 \
--gradient_checkpointing \
--trim_batches \
--accelerator=ddp \
--early_stopping_patience 20 \
--max_source_length 1024 \
;

