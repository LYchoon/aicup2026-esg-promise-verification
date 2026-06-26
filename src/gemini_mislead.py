"""Gemini few-shot Misleading 偵測器（補 DL 結構上無法輸出的 Misleading 類）。

模型 gemini-3.1-flash-lite、few-shot、temperature=0。
few-shot：每條規則的 prompt 內含「規則定義 + 1 個已知 Misleading 正例（取自訓練集）」作 in-context 參照——
  甲軸（rule1b_slogan / rule1_esg）以 10017（純口號型）為正例；乙軸（rule2_pay）以 11836（薪酬連財務型）為正例。
偵測器： Misleading = (rule1b_slogan ∧ rule1_esg) ∨ rule2_pay，套於 base 預測為 Clear 的列（Clear-only 約束）。

對外介面：detect_misleading(test_csv, base_sub, key) → 回傳被判為 Misleading 的 id 清單。
"""
import time
import pandas as pd

MODEL = "gemini-3.1-flash-lite"

HEADS = {
 "rule1b_slogan": (
  "只判一種誤導。【YES】僅當：整段『證據』通篇都是抽象的企業美德口號(穩健經營、創新、競爭力、"
  "多元布局、供應鏈韌性、以人為本、財務透明、誠信經營、持續成長、價值創造)的堆疊，"
  "完全沒有任何一個具體的事項——沒有任何具名方案、數字、年度、特定環境或社會或治理的作為，"
  "純粹用『公司經營得好』的空話來充當永續成果。【NO】只要證據含任何一個具體事項，一律 NO。"),
 "rule1_esg": (
  "判斷證據是否誤導地支持其永續(ESG)承諾。判別鍵：證據裡有沒有一個獨立的、以企業為行動者的具體 ESG 元素。\n"
  "【YES】整段把財務穩健/獲利成長/薪酬/競爭力/經營成功本身當作永續成果邀功；即使出現 ESG 詞彙也只是"
  "附著在財務主句上的修飾；全段找不到任何獨立成句、以企業為行動者的具體 ESG 行動/機制/可驗證數據。\n"
  "【NO】段落至少含一個獨立的具體 ESG 元素(可量化環境/社會數字、ESG治理機制或專責組織、具名永續方案、"
  "SDG對照、ESG績效以比率連結薪酬)；此時無論多少財務語言一律 NO。"),
 "rule2_pay": (
  "只判一種誤導。【YES】僅當：證據端出一套薪酬/獎酬/持股/激勵機制，其『首要、主軸』的連結指標是"
  "財務獲利(營收、合併毛利、營業利益、每股盈餘、股價/股東報酬率 TSR)，環境社會指標若有也只是次要附帶；"
  "並把這個以財務獲利為主軸的薪酬機制說成永續成果。\n"
  "【NO】若薪酬連結的『主軸』是環境/社會/減碳/淨零/排放/能源/水等 ESG 目標(即使同時也連到一些財務指標)，"
  "一律 NO；其餘(無薪酬機制、具體環境社會行動、一般績效制度、一般治理)也一律 NO。"),
}

# ── few-shot 正例（取自訓練集的 2 個已知 Misleading；依軸對應）──
FEWSHOT_A = (  # 甲軸正例：10017（純口號型）
  "統一企業致力於穩健經營，確保公司財務穩定與持續成長，並兼顧股東權益、員工發展與社會責任。"
  "我們透過創新產品與服務、多元化營運布局及強化供應鏈韌性，提升企業競爭力，並承諾遵循財務與稅務法規，"
  "維持高標準的財務透明度與公司治理。")
FEWSHOT_B = (  # 乙軸正例：11836（薪酬連財務型）
  "薪資報酬委員會會根據公司治理趨勢報告及整體薪酬市場競爭力檢視報告來定期評估董事及經理人薪資報酬，"
  "薪酬之給付除參考當年度個人經營績效外，亦依據公司營運之財務與財務相關績效達成狀況而定。"
  "本公司於 2021 年通過發行限制員工權利新股，指標為合併營收、合併毛利及毛利率、合併營業利益及營業利益率（並含部分 ESG 指標）。")
RULE_ANCHOR = {"rule1b_slogan": FEWSHOT_A, "rule1_esg": FEWSHOT_A, "rule2_pay": FEWSHOT_B}

FEWSHOT_TMPL = ("\n\n=== 正例參照（此文字依本規則應判 YES）===\n{ex}"
                "\n\n=== 待判文字 ===\n{t}\n\n依上述判別，只輸出一個詞：YES 或 NO。")


def _load(p):
    d = pd.read_csv(p, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    d.columns = [c.strip().strip("﻿").strip('"') for c in d.columns]
    return d


def detect_misleading(test_csv, base_sub, key):
    """對 base 預測為 Clear 的列跑 Gemini few-shot 偵測，回傳判為 Misleading 的 id 清單。"""
    from google import genai
    cli = genai.Client(api_key=key)
    DATA = {r["id"]: r["data"] for _, r in _load(test_csv).iterrows()}
    appl = list(base_sub[base_sub.evidence_quality == "Clear"]["id"])   # Clear-only 約束

    def ask(rule, tid):
        p = HEADS[rule] + FEWSHOT_TMPL.replace("{ex}", RULE_ANCHOR[rule]).replace("{t}", DATA[tid][:560].replace("\n", " "))
        for a in range(6):
            try:
                r = cli.models.generate_content(model=MODEL, contents=p,
                        config={"temperature": 0, "max_output_tokens": 1500})
                return (r.text or "").strip().upper().startswith("Y")
            except Exception as e:
                if any(k in str(e) for k in ("429", "RESOURCE", "503", "500")):
                    time.sleep(a + 1)
                else:
                    return False
        return False

    hits = []
    for j, tid in enumerate(appl):
        if (ask("rule1b_slogan", tid) and ask("rule1_esg", tid)) or ask("rule2_pay", tid):
            hits.append(tid)
        if (j + 1) % 100 == 0:
            print(f"      Gemini 進度 {j + 1}/{len(appl)}…")
    return hits
