"""融合 + base-rate 校準 → base 預測（尚未疊 Misleading）。

各子任務取其最強來源的 test 機率，統一做級聯解碼，並在推論期套用 base-rate 校準：
  T1 = argmax（模型已對齊真實先驗，不動）
  T2 = Yes iff P(Yes) >= tau (0.40)        ← 模型少判 Yes，降門檻補回
  T3 = Clear iff P(Clear) >= theta (0.65)  ← 模型過度預測 Clear，提高門檻壓回
  T4 = argmax（保守，不做先驗校正；理由見報告 §6.3）

級聯：T1=No → 全 N/A；T2=No → T3=N/A。N/A 為級聯結構性結果、不直接預測。
"""
import os
import numpy as np
import pandas as pd

HL = {"T1": ["Yes", "No"], "T2": ["Yes", "No"], "T3": ["Clear", "Not Clear"],
      "T4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]}
COLS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]


def load_csv(p):
    d = pd.read_csv(p, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    d.columns = [c.strip().strip("﻿").strip('"') for c in d.columns]
    return d


def build_base(probs_dir="data/probs", test_csv="data/test_ids.csv", tau=0.40, theta=0.65):
    """讀預存 per-task 機率 → 校準解碼，回傳 base 預測 DataFrame（無 Misleading）。"""
    T1 = np.load(os.path.join(probs_dir, "T1_lr15.npz"))["T1"].astype(float)   # T1 ← lr15
    T2 = np.load(os.path.join(probs_dir, "T2_b32.npz"))["T2"].astype(float)    # T2 ← b32
    T4 = np.load(os.path.join(probs_dir, "T4_base.npz"))["T4"].astype(float)   # T4 ← base
    T3 = (np.load(os.path.join(probs_dir, "T3_d2.npz"))["T3"].astype(float)    # T3 ← (d2+d3)/2
          + np.load(os.path.join(probs_dir, "T3_d3.npz"))["T3"].astype(float)) / 2

    ids = load_csv(test_csv)["id"].values
    assert len(ids) == len(T1), f"id 數 {len(ids)} 與機率列數 {len(T1)} 不符"

    rows = []
    for i in range(len(ids)):
        t1 = HL["T1"][int(T1[i].argmax())]
        if t1 == "No":
            rows.append((ids[i], "No", "N/A", "N/A", "N/A"))
            continue
        t2 = "Yes" if T2[i, 0] >= tau else "No"                      # τ on P(Yes)
        t3 = ("Clear" if T3[i, 0] >= theta else "Not Clear") if t2 == "Yes" else "N/A"  # θ on P(Clear)
        t4 = HL["T4"][int(T4[i].argmax())]
        rows.append((ids[i], t1, t4, t2, t3))
    return pd.DataFrame(rows, columns=COLS)


def check_violations(sub):
    """級聯一致性檢查：回傳違規列數（應為 0）。"""
    v = 0
    for _, r in sub.iterrows():
        if r.promise_status == "No" and not (r.verification_timeline == r.evidence_status == r.evidence_quality == "N/A"):
            v += 1
        if r.evidence_status == "No" and r.evidence_quality != "N/A":
            v += 1
    return v
