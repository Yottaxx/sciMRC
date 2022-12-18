items="all withoutExtractive withoutGenerative withoutUnanswerable withoutYes withoutTask1 withoutTask2 withoutTask3 "
for item in $items:
do
  export CUDA_VISIBLE_DEVICES='6,7'
  echo ./filter_data/annotate/cleanTrain$item.json
  python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29515 run_summarization.py \
  --learning_rate 1e-4 \
  --model_name_or_path 't5-base' \
  --output_dir $item-paperSummarization-QA-t5-1e4-16epoch-16bsz \
  --num_train_epochs 16 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --warmup_ratio 0.10 \
  --fp16 false \
  --eval_steps 100 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy 'steps' \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --save_steps 100 \
  --logging_steps 100 \
  --train_file ./filter_data/annotate/cleanTrain$item.json \
  --validation_file ./filter_data/annotate/cleanDevall.json \
  --test_file ./filter_data/annotate/cleanTestall.json \
  --max_source_length 512 \
  --max_target_length 512 \
  --pad_to_max_length false \
  --source_prefix "Question Answering: " \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --ddp_find_unused_parameters true \
  --overwrite_output_dir true \
  --prediction_loss_only false \
  --load_best_model_at_end true \
  --metric_for_best_model 'bleu' \
  --predict_with_generate true \
  --greater_is_better true \
  --num_beams 5  \
  --save_total_limit 3 \
  > $item-paperSummarization-QA-t5-1e4-16epoch-16bsz.log 2>&1

  python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29515 run_summarization.py \
  --learning_rate 1e-4 \
  --model_name_or_path $item-paperSummarization-QA-t5-1e4-16epoch-16bsz \
  --output_dir $item-paperSummarization-QA-t5-1e4-16epoch-16bsz-dev \
  --num_train_epochs 16 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --warmup_ratio 0.10 \
  --fp16 false \
  --eval_steps 100 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy 'steps' \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --save_steps 100 \
  --logging_steps 100 \
  --train_file ./filter_data/annotate/cleanTrain$item.json \
  --validation_file ./filter_data/annotate/cleanDevall.json \
  --test_file ./filter_data/annotate/cleanDevall.json \
  --max_source_length 512 \
  --max_target_length 512 \
  --pad_to_max_length false \
  --source_prefix "Question Answering: " \
  --do_train false \
  --do_eval false \
  --do_predict true \
  --ddp_find_unused_parameters true \
  --overwrite_output_dir false \
  --prediction_loss_only false \
  --load_best_model_at_end true \
  --metric_for_best_model 'bleu' \
  --predict_with_generate true \
  --greater_is_better true \
  --num_beams 5  \
  > $item-paperSummarization-QA-t5-1e4-16epoch-16bsz-dev.log 2>&1

  python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29515 run_summarization.py \
  --learning_rate 1e-4 \
  --model_name_or_path $item-paperSummarization-QA-t5-1e4-16epoch-16bsz \
  --output_dir $item-paperSummarization-QA-t5-1e4-16epoch-16bsz-test \
  --num_train_epochs 16 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --warmup_ratio 0.10 \
  --fp16 false \
  --eval_steps 100 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy 'steps' \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --save_steps 100 \
  --logging_steps 100 \
  --train_file ./filter_data/annotate/cleanTrain$item.json \
  --validation_file ./filter_data/annotate/cleanTestall.json \
  --test_file ./filter_data/annotate/cleanTestall.json \
  --max_source_length 512 \
  --max_target_length 512 \
  --pad_to_max_length false \
  --source_prefix "Question Answering: " \
  --do_train false \
  --do_eval false \
  --do_predict true \
  --ddp_find_unused_parameters true \
  --overwrite_output_dir false \
  --prediction_loss_only false \
  --load_best_model_at_end true \
  --metric_for_best_model 'bleu' \
  --predict_with_generate true \
  --greater_is_better true \
  --num_beams 5  \
  > $item-paperSummarization-QA-t5-1e4-16epoch-16bsz-test.log 2>&1
done
