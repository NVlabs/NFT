pip install word2number
save_path=./NFT_7B
context_length=4096
max_prompt_length=512
n_samples=1
top_p=0.7
temperature=0.6
top_k=-1
max_response_length=$(($context_length-$max_prompt_length))


for data_name in aime-2024-boxed_w_answer math500_boxed minerva_math olympiadbench aime2025_32_dapo_boxed_w_answer amc2023_32_dapo_boxed_w_answer; do
    data_load_path=./data/$data_name.parquet; \
    data_save_path=${save_path}/${data_name}_max${context_length}_topp${top_p}topk${top_k}_temp${temperature}_@${n_samples}.parquet; \
    python -u -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$data_load_path \
        data.prompt_key=prompt \
        data.n_samples=$n_samples \
        data.output_path=$data_save_path \
        model.path=$save_path/huggingface \
        +model.trust_remote_code=True \
        rollout.temperature=$temperature \
        rollout.top_k=$top_k \
        rollout.top_p=$top_p \
        rollout.prompt_length=$max_prompt_length \
        rollout.response_length=$max_response_length \
        rollout.tensor_model_parallel_size=4 \
        rollout.gpu_memory_utilization=0.8; \
    python3 -m verl.trainer.compute_acc --input_path $data_save_path --verifier all;
done