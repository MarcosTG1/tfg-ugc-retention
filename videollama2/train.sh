#!/bin/bash
# Train VideoLLaMA2 EVQA — Experimento L1 (audio-visual-text)
#
# Este script vive aquí como referencia de configuración.
# Copiarlo al servidor en:
#   ~/TFG/code/1 - Engagement.../LMM-EVQA/VideoLLaMA2-audio_visual/train.sh
# Y ejecutar desde ese directorio:
#   conda activate videollama2
#   tmux new -s train_l1
#   sh train.sh

RUN_NAME=L1_videollama2_av

python -u videollama2/train_EVQA.py \
    --model_type videollama2_qwen2 \
    --model_path /media/2tbraid/martugue/TFG/models-weights/videollama2_weights \
    --data_folder /media/5tbraid/data/martugue/SnapUGC/raw \
    --data_path /media/5tbraid/data/martugue/SnapUGC/processed/train.json \
    --vision_tower google/siglip-so400m-patch14-384 \
    --audio_tower /media/2tbraid/martugue/TFG/models-weights/videollama2_weights/audio_tower.bin \
    --pretrain_mm_mlp_adapter_a /media/2tbraid/martugue/TFG/models-weights/videollama2_weights/mm_projector_a.bin \
    --mm_projector_type stc_connector_v35 \
    --mm_projector_a_type mlp2x_gelu \
    --va True \
    --tune_audio_tower True \
    --tune_adapter_llm True \
    --tune_mm_mlp_adapter_a True \
    --mm_vision_select_layer -2 \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --loss_type mse \
    --output_dir /media/2tbraid/martugue/TFG/models/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 17 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name $RUN_NAME | tee /media/2tbraid/martugue/TFG/models/${RUN_NAME}_training.log
