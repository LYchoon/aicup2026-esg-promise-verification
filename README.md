# ESG 永續承諾驗證競賽 2026 — 可復現程式（TEAM_10219）

繁體中文 ESG 報告四子任務級聯分類；最終提交 public weighted macro-F1 = **0.6122**。

---

## 復現流程

```bash
git clone https://github.com/<your-account>/aicup2026-esg-promise-verification.git
cd aicup2026-esg-promise-verification

pip install -r requirements.txt

# （選用）填入 Gemini 金鑰以實際重跑 Misleading 偵測：
cp .env.example .env        # 然後編輯 .env，填 GEMINI_API_KEY=你的金鑰

python main.py              # → prediction/submission.csv
```

`main.py` 一鍵跑完整流程：

1. **融合 + base-rate 校準**：讀 `data/probs/` + `data/test_ids.csv`，做 per-task 融合與級聯解碼，套校準閾值 T2 τ=0.40、T3 θ=0.65。
2. **Misleading 偵測**：若有 `GEMINI_API_KEY` 且 `data/test.csv`（官方測試文字）存在 → 跑 Gemini few-shot 偵測（Clear-only）；否則套用記錄的 3 列 Misleading（與最終提交一致）。
3. 輸出 `prediction/submission.csv`（2000 列、違規 0）。

> 不設金鑰時輸出與最終提交逐欄 0 差異（已驗證）。

---

## 目錄結構

```
.
├── main.py                 一鍵流程：融合+校準 → Misleading → prediction/submission.csv
├── requirements.txt
├── .env.example            GEMINI_API_KEY 範本（複製為 .env 填入）
├── src/
│   ├── fuse_calibrate.py   融合 + base-rate 校準（build_base / check_violations）
│   ├── gemini_mislead.py   Gemini few-shot Misleading 偵測（detect_misleading）
│   ├── train_dl.py         從頭訓練（選用，需 GPU，重產 data/probs）
│   └── dl_lib.py           核心 DL 庫（train_one / run_predict / cascade_decode / MultiHead）
├── data/
│   ├── probs/              5 個預存 test 機率 npz（T1_lr15 / T2_b32 / T3_d2 / T3_d3 / T4_base）
│   └── test_ids.csv        測試集 id 欄（供融合/校準；官方完整資料未隨附）
└── prediction/             輸出（submission.csv 寫於此）
```

---

## 資料說明

為遵守競賽資料使用規範，本 repo **未隨附主辦提供的完整官方資料**（train / val / test 的文字內容）：

- `data/test_ids.csv`：僅測試集的 `id` 欄，供融合與級聯解碼對齊（id 為流水號、不含內容）。
- 選用步驟所需的官方完整資料請自備（皆已列入 `.gitignore`，不會被提交）：
  - Gemini 偵測需 `data/test.csv`（官方 test，含 `data` 文字欄）。
  - 從頭訓練需 `data/trainval.csv`（官方 train+val 合併、含標註）。
