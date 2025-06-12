<div align='center'>
<h1>Negative-aware FineTuning (NFT) <br> Bridging Supervised Learning and Reinforcement Learning in Math Reasoning </h1>

<!-- TODO:  Thread,Paper,Dataset,Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2505.18116)
[![Blog](https://img.shields.io/badge/Blog-3858bf?style=for-the-badge&logo=homepage&logoColor=white)](https://research.nvidia.com/labs/dir/Negative-aware-Fine-Tuning/)
[![Dataset](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ChenDRAG/VeRL_math_validation)
[![Weights](https://img.shields.io/badge/Model%20Weights-63cad3?style=for-the-badge&logo=huggingface&logoColor=white)](TBD)
</div>

<p align="center">
  <img src="./NFT.jpg" alt="seed logo" style="width:80%;">
</p>

> **🔥 News!!!**
> - [2025/06] We release training code and [checkpoints](TBD) of NFT. Our code is based on the [VeRL](https://github.com/volcengine/verl) framework.

NFT is a pure supervised learning method for improving LLMs' math-reasoning abilities with no external teachers.

- As an SL method, NFT outperforms leading RL algorithms like GRPO and DAPO in 7B model experiments and performs similarly to DAPO in 32B settings.
- NFT allows directly optimizing LLMs on negative data, thereby significantly outperforming other SL baselines such as Rejective sampling Fine-Tuning (RFT).
- NFT is equivalent to GRPO when training is strictly on-policy, desipte their entirely different theoretical foundations.

NFT shows self-reflective improvement is not an inherent priority of RL algorithms. Rather, the current gap between SL and RL methods actually stems from their ability to effectively leverage negative data.
<!-- <p align="center">
  <img src="./exp.jpg" alt="seed logo" style="width:80%;">
</p> -->

## Evaluation

### Environment setup
We use exactly the same environment configuration as the official DAPO codebase.
```base
pip install git+ssh://git@github.com/volcengine/verl.git@01ef7184821d0d7844796ec0ced17665c1f50673
```

### Released Models

We provide the model weights of [NFT-7B](TBD) and [NFT-32B](TBD), which are respectively trained based on Qwen2.5-Math-7B and Qwen2.5-32B using the NFT algorithm.

### Benchmarking

We provide the evaluation codebase integrated in the VeRL infra:

Please refer to `eval_local_7B.sh` and `eval_local_32B.sh` for evaluation scripts.

## Training

### Environment setup
We use exactly the same environment configuration as the official DAPO codebase.
```base
pip install git+ssh://git@github.com/volcengine/verl.git@01ef7184821d0d7844796ec0ced17665c1f50673
```

### Datasets
We employ public dataset [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) for training, and 6 public math benchmarks for validation. Download pre-sorted training and validation data by
```bash
bash download_data.sh
```

### Base Model
```bash
bash download_model.sh
```

### Starting Experiments
Please see `train_7B.sh` and `train_32B.sh` for a running script (one node). Note that we run 7B experiments using 4\*8 H100s, and 32B experiments using 16\*8 H100s. Please refer to the instruction of [VeRL](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main) for launching distributed tasks.

Hyperparameter:

- `neg_weight`: The weight of negative data in NFT's objective. Set to 1.0 for default NFT config. Set to 0.0 for RFT by masking out all negative data loss. Set to -1.0 for the DAPO algorithm for comparison.   
- `normalize`: Controls the prompt weight in NFT's objective. Set to 0 so that all question data is treated equally. Set to 1 (default) or 2 to prioritize harder questions. `normalize=1` matches Dr. GRPO algorithm in on-policy training, while `normalize=2` mathches standard GRPO.

## Acknowledgement

We thank the [verl](https://github.com/volcengine/verl) for providing the awesome open-source RL infrastructure.

## Citation
If you find our project helpful, please consider citing
```
@article{chen2025bridging,
      title         = {Bridging Supervised Learning and Reinforcement Learning in Math Reasoning},
      author        = {Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, Haoxiang Wang},
      journal       = {arXiv preprint arXiv:2505.18116},
      year          = {2025}
    }
```