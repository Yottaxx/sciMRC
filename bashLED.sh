items="withoutTask3 "
#items="all withoutTask3 withoutExtractive withoutGenerative withoutUnanswerable withoutYes withoutTask1 withoutTask2 "

for item in $items:
do
  export CUDA_VISIBLE_DEVICES='0,3'
  echo ./filter_data/annotate/cleanTrain$item.json
  python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29512 run_summarization.py \
  --learning_rate 2e-5 \
  --model_name_or_path "allenai/led-base-16384" \
  --output_dir $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz \
  --num_train_epochs 8 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --warmup_ratio 0.10 \
  --fp16 false \
  --eval_steps 100 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy 'steps' \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --save_steps 100 \
  --logging_steps 100 \
  --train_file ./filter_data/annotate/cleanTrain$item.json \
  --validation_file ./filter_data/annotate/cleanDevall.json \
  --test_file ./filter_data/annotate/cleanTestall.json \
  --max_source_length 4096 \
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
  > $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz.log 2>&1

  python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29512 run_summarization.py \
  --learning_rate 2e-5 \
  --model_name_or_path $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz \
  --output_dir $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz-dev \
  --num_train_epochs 8 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --warmup_ratio 0.10 \
  --fp16 false \
  --eval_steps 100 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy 'steps' \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --save_steps 100 \
  --logging_steps 100 \
  --train_file ./filter_data/annotate/cleanTrain$item.json \
  --validation_file ./filter_data/annotate/cleanDevall.json \
  --test_file ./filter_data/annotate/cleanDevall.json \
  --max_source_length 4096 \
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
  > $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz-dev.log 2>&1

  python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29512 run_summarization.py \
  --learning_rate 2e-5 \
  --model_name_or_path $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz \
  --output_dir $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz-test \
  --num_train_epochs 8 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --warmup_ratio 0.10 \
  --fp16 false \
  --eval_steps 100 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy 'steps' \
  --logging_strategy 'steps' \
  --save_strategy 'steps' \
  --save_steps 100 \
  --logging_steps 100 \
  --train_file ./filter_data/annotate/cleanTrain$item.json \
  --validation_file ./filter_data/annotate/cleanTestall.json \
  --test_file ./filter_data/annotate/cleanTestall.json \
  --max_source_length 4096 \
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
  > $item-paperSummarization-QA-led-2e5-16epoch-2x8bsz-test.log 2>&1
done