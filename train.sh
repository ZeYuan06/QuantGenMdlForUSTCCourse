#!/usr/bin/env bash
set -euo pipefail

cd /home/licunchun/QuantGenMdlForUSTCCourse

# NOTE:
# - `nohup` runs in a non-interactive shell, so `conda activate ...` often fails unless `conda init` was done.
# - Use `conda run -n qml_gpu ...` to run inside the existing env reliably.

mkdir -p logs

# 每次运行写入一个新的实验目录（方案A：不覆盖旧结果）
# 你也可以手动指定：EXP_DIR=data/mnist01_run_custom bash train.sh
EXP_DIR=${EXP_DIR:-data/mnist01_run_$(date +%F_%H%M%S)}

# 日志文件（默认固定 train.log；如果你想每次一个文件，改成 train_$(date...).log）
LOG_FILE=${LOG_FILE:-logs/train.log}

# 推荐 nohup 用法：
#   nohup bash train.sh > train.log 2>&1 &

# 可选：只用 0 号 GPU（不写就会看到 2 张卡）
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

conda run -n qml_gpu python scripts/mnist01_train_correct_gpu.py \
  --data data/mnist01 \
  --exp "$EXP_DIR" \
  --budget medium \
  --seed 42 \
  --project_product

# 训练+生成+eval 完成后，自动从 $EXP_DIR/gen/*/eval_report.json 画汇报用对比图
conda run -n qml_gpu python scripts/mnist01_plot_run_reports.py \
  --exp "$EXP_DIR" \
  --out "$EXP_DIR/plots"

echo "[train.sh] exp_dir: $EXP_DIR"
echo "[train.sh] log_file: $LOG_FILE"

# 如果只想复用已经准备好的数据/编码/分类器，可取消下面三项的注释：
#   --skip_prepare \
#   --skip_encode \
#   --skip_classifier

# cd /home/licunchun/QuantGenMdlForUSTCCourse

# export CUDA_VISIBLE_DEVICES=0
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export TF_CPP_MIN_LOG_LEVEL=2

# conda run -n qml_gpu python scripts/mnist01_train_correct_gpu.py \
#   --data data/mnist01 \
#   --budget full \
#   --seed 42

# 想在同一个 exp 里只重跑某个模型时，建议直接单独运行：
#   conda run -n qml_gpu python scripts/mnist01_train_correct_gpu.py --data data/mnist01 --exp "$EXP_DIR" --budget medium --seed 42 --project_product --skip_qdt --skip_qddpm --force