#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --partition=gpu_h100
#SBATCH --time=32:00:00
#SBATCH --mem=84G
#SBATCH --exclusive
#SBATCH --job-name=qwen1
#SBATCH -o ./log/qwen_%j.out

# =============== 生成时间戳 ===============
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_DIR="./log"
OUTPUT_DIR="./check_qwen3/checkpoints_${TIMESTAMP}"



# =============== 加载 Snellius 2023 工具链 + CUDA ===============
module load 2023
module load CUDA/12.4.0

nvidia-smi

if [ -z "$CUDA_HOME" ]; then
    echo "❌ CUDA_HOME not set after module load!"
    exit 1
fi
echo "✅ CUDA_HOME = $CUDA_HOME"

# =============== 激活 Conda 环境 ===============
source /home/npu/miniconda3/bin/activate qwen

# =============== 跳过 DeepSpeed CUDA 算子编译（关键！）===============
export DS_SKIP_CUDA_BUILD=1

# =============== 环境检查（直接内联，避免 /tmp 问题）===============
echo "🔍 Checking PyTorch and CUDA..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Number of GPUs:', torch.cuda.device_count())
if not torch.cuda.is_available():
    exit(1)
print('✅ CUDA check passed.')
"

if [ $? -ne 0 ]; then
    echo "❌ CUDA check failed!"
    exit 1
fi

# =============== 启动训练 ===============
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=4

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    qwenvl/train/train_qwen.py \
    --model_name_or_path "./check_qwen3/checkpoints_20251016_1639" \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --dataset_use "samm_data" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "./cache" \
    --bf16 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-7 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 4096 \
    --data_flatten True \
    --data_packing True \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --num_train_epochs 12 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --deepspeed ./scripts/zero3.json
