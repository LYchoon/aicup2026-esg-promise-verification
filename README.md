# ESG 永續承諾驗證競賽 2026 — 可復現程式（TEAM_10219）

繁體中文 ESG 報告四子任務級聯分類（T1 promise / T2 evidence / T3 quality / T4 timeline）。
最終提交 public weighted macro-F1 = **0.6122**（T1 0.7912 / T2 0.6931 / T3 0.4466 / T4 0.5983）。

方法：單一中文 MacBERT 四級聯頭 → per-task 機率融合 → **推論期 base-rate 校準** → **Gemini few-shot Misleading 偵測**。

---

## 一鍵復現

```bash
git clone https://github.com/<your-account>/aicup2026-esg-promise-verification.git
cd aicup2026-esg-promise-verification

pip install -r requirements.txt

# （選用）填入 Gemini 金鑰以執行 Misleading 偵測步驟：
cp .env.example .env        # 然後編輯 .env，填 GEMINI_API_KEY=你的金鑰

python main.py              # → prediction/submission.csv
```

`main.py` 會自動跑完整流程：

1. **融合 + base-rate 校準**：讀 `data/probs/`（5 個預存 test 機率）+ `data/test.csv`，
   做 per-task 融合與級聯解碼，套校準閾值 **T2 τ=0.40、T3 θ=0.65**。
2. **Misleading 偵測**：
   - 若有 `GEMINI_API_KEY` **且** `data/test.csv`（官方測試文字）存在 → 對 base 為 Clear 的列跑 **Gemini few-shot** 偵測（Clear-only 約束）。
   - 否則 → 直接套用記錄的 3 列 Misleading（與最終提交一致，不需金鑰即可精確復現）。
3. 輸出 `prediction/submission.csv`（2000 列、違規 0）。

> 本 repo 為遵守競賽資料規範，**未隨附官方完整資料**（僅附 test 的 id 欄）。
> 不設金鑰時輸出與最終提交逐欄 0 差異（已驗證）。
> 若要實際重跑 Gemini 偵測，請另將官方測試集置於 `data/test.csv`（含 `data` 文字欄）並設金鑰。

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

## 方法摘要

| 階段 | 做法 | 增益 |
|---|---|---|
| 架構 | 單 MacBERT 編碼器 + 4 級聯頭（T1→T2,T4→T3），per-task 異質融合 | base 0.6078 |
| 推論校正 | T3：Clear iff P≥0.65（模型過度預測 Clear，校回訓練集 base rate）| T3 +0.0062 → 0.6099 |
|  | T2：Yes iff P≥0.40（模型少判 Yes）| T2 +0.0035 |
| Misleading | Gemini few-shot 3 規則 + Clear-only → 3 列覆蓋（DL 頭結構上不輸出 Misleading）| T3 → 0.4466，**最終 0.6122** |

**核心洞察**：base-rate / 先驗校準（logit adjustment 的後置版）是可遷移的乾淨槓桿——
T3/T2 在 test 的 release-shift 上系統性偏離訓練集先驗，校回 base rate 即抬升 macro-F1。

---

## 從頭訓練（選用，需 GPU）

在 Colab T4 上放置 `data/trainval.csv`（官方 train+val 合併、含標註；自備）後執行：

```bash
python src/train_dl.py     # 訓練 lr15/b32/base 各 3 seeds → data/probs/*_avg_test.npz
```

配方（`src/dl_lib.py` 內 `DEFAULTS` + `train_dl.py` 的 `RECIPE`）：
`pool=attn_pt`、`dyn_layer_k=4`、`ckpt_soup=3`（SWA）、三引導注意力 KL
（`aux_attn=0.5 / aux_attn_year=0.3 / aux_attn_ev=0.3`）、`class_weight=balanced`、
`task_w=[1.6,1,1,0.6]`、`epochs=8`、`train_added=True`、3 seeds (42/142/242) 平均。
T3 來源為兩個 base-family 配置(d2/d3)的機率平均，精確機率已附於 `data/probs/`，建議直接沿用以精確復現。

---

## 資料說明

為遵守競賽資料使用規範，本 repo **未隨附主辦提供的完整官方資料**（train / val / test 的文字內容）：
- `data/test_ids.csv`：僅測試集的 `id` 欄，供融合與級聯解碼對齊（id 為流水號、不含內容）。
- 選用步驟所需的官方完整資料請自備：Gemini 偵測需 `data/test.csv`（官方 test，含 `data` 文字欄）；
  從頭訓練需 `data/trainval.csv`（官方 train+val 合併、含標註）。兩者皆已列入 `.gitignore`，不會被提交。
