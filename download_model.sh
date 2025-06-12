mkdir -p models
# required only for 7B models
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir ./models/Qwen2.5-Math-7B --local-dir-use-symlinks False

# required only for 32B models
huggingface-cli download Qwen/Qwen2.5-32B --local-dir ./models/Qwen2.5-32B --local-dir-use-symlinks False
# replace 32B tokenizer config to keep a consistent system prompt with 7B
cp models/32b_tokenizer_config.json models/Qwen2.5-32B/tokenizer_config.json

