"""一鍵復現 ESG 永續承諾驗證競賽 2026 最終提交（TEAM_10219，public weighted macro-F1 = 0.6122）。

流程：
  1. 融合 + base-rate 校準（data/probs + data/test.csv）→ base 預測（T2 τ=0.40、T3 θ=0.65）
  2. Misleading 偵測：
       - 若已設環境變數 GEMINI_API_KEY → Gemini few-shot 偵測（Clear-only 約束）
       - 否則 → 使用記錄的 3 列 Misleading（與最終提交一致）
  3. 套 Clear-only 覆蓋 → 寫出 prediction/submission.csv

用法：
    pip install -r requirements.txt
    export GEMINI_API_KEY=你的金鑰     # 選用；不設則用記錄的 3 列
    python main.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fuse_calibrate import build_base, check_violations   # noqa: E402

PROBS_DIR = os.path.join(ROOT, "data", "probs")
TEST_IDS = os.path.join(ROOT, "data", "test_ids.csv")     # 隨附：僅 id 欄（供融合/校準）
TEST_FULL = os.path.join(ROOT, "data", "test.csv")        # 選用：官方測試文字（自備，供 Gemini 偵測）
OUT_DIR = os.path.join(ROOT, "prediction")
OUT_CSV = os.path.join(OUT_DIR, "submission.csv")

RECORDED_MIS = ["12735", "12799", "12971"]   # 最終提交實際覆蓋的 Misleading 列


def _load_env():
    """若根目錄有 .env，將其中的 KEY=VALUE 載入環境變數（不覆蓋已存在者）。"""
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_env()
    print("[1/3] 融合 + base-rate 校準（T1=lr15、T2=b32 τ=0.40、T3=(d2+d3)/2 θ=0.65、T4=base）…")
    sub = build_base(PROBS_DIR, TEST_IDS)
    print(f"      base 完成：{len(sub)} 列")

    print("[2/3] Misleading 偵測…")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and os.path.exists(TEST_FULL):
        try:
            from gemini_mislead import detect_misleading
            print("      GEMINI_API_KEY + 官方 test 文字 → 執行 Gemini few-shot 偵測（Clear-only，約數分鐘 / 數千次 API 呼叫）…")
            mis_ids = detect_misleading(TEST_FULL, sub, key)
            print(f"      Gemini 命中 {len(mis_ids)} 列：{mis_ids}")
        except Exception as e:
            print(f"      Gemini 失敗（{e}）→ 改用記錄的 3 列")
            mis_ids = RECORDED_MIS
    else:
        if key and not os.path.exists(TEST_FULL):
            print("      已設金鑰但未找到 data/test.csv（官方測試文字）→ 跳過 Gemini，使用記錄的 3 列。")
            print("      （如需實際重跑偵測，請將官方 test 置於 data/test.csv）")
        else:
            print("      未設 GEMINI_API_KEY → 使用記錄的 3 列 Misleading（與最終提交一致）")
        mis_ids = RECORDED_MIS

    msk = sub.id.isin(mis_ids) & (sub.evidence_quality == "Clear")   # Clear-only 約束
    sub.loc[msk, "evidence_quality"] = "Misleading"

    print("[3/3] 寫出 prediction/submission.csv…")
    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    viol = check_violations(sub)
    dist = sub.evidence_quality.value_counts().to_dict()
    print(f"完成 ✓ {OUT_CSV}")
    print(f"      {len(sub)} 列；Misleading {int(msk.sum())} 列；違規 {viol}")
    print(f"      evidence_quality 分布：{dist}")


if __name__ == "__main__":
    main()
