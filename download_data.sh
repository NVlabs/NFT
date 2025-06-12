HF_REPO_BASE="https://huggingface.co/datasets/ChenDRAG/VeRL_math_validation/resolve/main"
mkdir -p data
wget -O data/dapo-math-17k_10boxed.parquet "${HF_REPO_BASE}/dapo-math-17k_10boxed.parquet?download=true"
wget -O data/aime-2024-boxed_w_answer.parquet "${HF_REPO_BASE}/aime-2024-boxed_w_answer.parquet?download=true"
wget -O data/math500_boxed.parquet "${HF_REPO_BASE}/math500_boxed.parquet?download=true"
wget -O data/minerva_math.parquet "${HF_REPO_BASE}/minerva_math.parquet?download=true"
wget -O data/olympiadbench.parquet "${HF_REPO_BASE}/olympiadbench.parquet?download=true"
wget -O data/aime2025_32_dapo_boxed_w_answer.parquet "${HF_REPO_BASE}/aime2025_32_dapo_boxed_w_answer.parquet?download=true"
wget -O data/amc2023_32_dapo_boxed_w_answer.parquet "${HF_REPO_BASE}/amc2023_32_dapo_boxed_w_answer.parquet?download=true"