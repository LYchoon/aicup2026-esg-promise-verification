"""【選用，需 GPU】從頭訓練 per-task DL 模型，重新產生 main.py 用的 test 機率。

在 Colab T4 執行。配方 = 單一中文 MacBERT 四級聯頭 + trainval 全訓 + 3 seeds 平均。
需要的檔案：
  - src/dl_lib.py（同目錄）
  - data/trainval.csv（官方 train+val 合併、含標註；自備）
  - data/test.csv（已附）
產出：data/probs/{name}_avg_test.npz（各 task 機率）→ 對應命名為 T1_lr15 / T2_b32 / T4_base 供 main.py 用。

註：最終提交的 T3 來源是兩個 base-family 配置(d2/d3)的機率平均；其精確機率已附於
   data/probs/T3_d2.npz、T3_d3.npz（建議直接沿用以精確復現）。

執行：python src/train_dl.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_lib import run_predict

# 共用訓練配方（單一模型；各 per-task 來源僅在學習率 / batch size 上不同）
RECIPE = {"class_weight": "balanced", "lr": 3e-5, "pool": "attn_pt", "aux_attn": 0.5,
          "aux_attn_year": 0.3, "aux_attn_ev": 0.3, "task_w": [1.6, 1.0, 1.0, 0.6],
          "epochs": 8, "train_added": True, "dyn_layer_k": 4, "ckpt_soup": 3}

CONFIGS = {
    "lr15": {**RECIPE, "lr": 1.5e-5},     # → T1 來源
    "b32":  {**RECIPE, "batch": 32},      # → T2 來源
    "base": {**RECIPE},                   # → T4 來源（其 T3 頭可作 d2/d3 之一）
}

if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        run_predict({**cfg, "name": name}, seeds=(42, 142, 242),
                    train_path="data/trainval.csv", test_path="data/test.csv", oof_dir="data/probs")
        print(f"=== {name} done → data/probs/{name}_avg_test.npz ===", flush=True)
    print("TRAIN_DONE", flush=True)
