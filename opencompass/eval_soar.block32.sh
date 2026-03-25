#!/bin/bash
#SBATCH --job-name="soar_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # 请求2块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar

# cp ../generate.soar.py ../generate.py
# python run.py llada_instruct_gen_humaneval_length256_block32.py -w outputs/llada_instruct_gen_humaneval_length256_block32_logits_soar

cp ../generate.soar.py ../generate.py
python run.py llada_instruct_gen_mbpp_length256_block32.py -w outputs/llada_instruct_gen_mbpp_length256_block32_confidence_soar

cp ../generate.soar.py ../generate.py
python run.py llada_instruct_gen_gsm8k_length256_block32.py -w outputs/llada_instruct_gen_gsm8k_length256_block32_confidence_soar