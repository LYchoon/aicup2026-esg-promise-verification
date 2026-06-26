"""DL 探索線共用庫 — 在 Colab T4 上 fine-tune 輕量中文 encoder（多任務級聯）。

設計（對應 dl/PLAN_DL.md，協定事先寫死）：
- 決策指標：3-fold GroupKFold(ticker) on train.csv(900)，pooled OOF 一次計分，
  weighted macro-F1（T1 .20/T2 .30/T3 .35/T4 .15，與 loop/evalmetric.py 同義）。
  固定折（GroupKFold 決定性）+ 固定 seed=42，30 輪實驗共用 → 可比。
- 級聯感知訓練：四頭 = T1{Yes,No}、T2{Yes,No}、T3{Clear,Not Clear}、T4{4 期程}。
  T2/T4 loss 只算 gold T1=Yes 列、T3 只算 gold T2=Yes 列；Misleading（1 例）不學。
  推論：T1=No → T2/T3/T4=N/A；T2=No → T3=N/A（N/A 是結構，不是預測）。
  已驗證 gold：T1=Yes 時 T4 從不為 N/A → T4 頭不需 N/A 類。
- 每輪結束 append /content/results.jsonl 一列（cfg+score+per_task+per_class+時間），
  OOF 預測與機率存 /content/oof/{name}.{csv,npz} 供錯例分析與後續 ensemble。

cfg 鍵（全部有預設，exp 腳本只覆寫差異）：
  name, model_name, max_len, pool('mean'|'cls'), lr, head_lr_mult, epochs, batch,
  warmup_frac, weight_decay, seed, task_w(4 floats), class_weight(None|'balanced'),
  focal_gamma(0=off), label_smooth, rdrop_alpha(0=off), fgm_eps(0=off), llrd(0=off,
  else decay e.g. 0.9), grad_accum, trunc('head'|'headtail'), aug_csv(None|路徑，
  只併入訓練折), prompt(False|True 兩段式輸入), dropout
"""
import json, math, os, random, re, time, unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------- 評分（與 loop/evalmetric.py 同義） ----------------
TASK_FIELDS = {"T1": "promise_status", "T2": "evidence_status",
               "T3": "evidence_quality", "T4": "verification_timeline"}
FULL_LABELS = {
    "T1": ["Yes", "No"],
    "T2": ["Yes", "No", "N/A"],
    "T3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "T4": ["already", "within_2_years", "between_2_and_5_years",
           "more_than_5_years", "N/A"],
}
WEIGHTS = {"T1": 0.20, "T2": 0.30, "T3": 0.35, "T4": 0.15}
# 各頭的「可學」類別（級聯內）
HEAD_LABELS = {
    "T1": ["Yes", "No"],
    "T2": ["Yes", "No"],
    "T3": ["Clear", "Not Clear"],
    "T4": ["already", "within_2_years", "between_2_and_5_years",
           "more_than_5_years"],
}
TASKS = ["T1", "T2", "T3", "T4"]


def score_frames(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    per_task, per_class = {}, {}
    for t, f in TASK_FIELDS.items():
        yt, yp = gold[f].tolist(), pred[f].tolist()
        per_task[t] = float(f1_score(yt, yp, labels=FULL_LABELS[t],
                                     average="macro", zero_division=0))
        per_class[t] = {l: float(s) for l, s in zip(
            FULL_LABELS[t],
            f1_score(yt, yp, labels=FULL_LABELS[t], average=None, zero_division=0))}
    return {"score": sum(WEIGHTS[t] * per_task[t] for t in WEIGHTS),
            "per_task": per_task, "per_class": per_class}


def load_df(path):
    df = pd.read_csv(path, keep_default_na=False)
    for c in ("evidence_status", "evidence_quality", "verification_timeline"):
        if c in df.columns:
            df[c] = df[c].replace("", "N/A")
    return df


# ---------------- Dataset ----------------
class TextDS(Dataset):
    """tokenize 在 __getitem__ 內做（900 列很小，無所謂）；trunc='headtail' 時
    超長文本取前 3/4 + 後 1/4 視窗（年份/期程線索常在句尾）。"""

    def __init__(self, df, tok, cfg, train=False):
        self.rows = df.to_dict("records")
        self.tok, self.cfg = tok, cfg
        self.train = train

    def __len__(self):
        return len(self.rows)

    def _encode(self, text, want_offsets=False, company=""):
        cfg, tok = self.cfg, self.tok
        if cfg.get("company_prefix") and company:
            text = f"{company}：{text}"
        if cfg.get("norm"):
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"\s+", " ", text).strip()
        if want_offsets:
            return tok(text, truncation=True, max_length=cfg["max_len"],
                       return_offsets_mapping=True)
        if cfg["trunc"] == "tail":
            ids_full = tok(text, add_special_tokens=False)["input_ids"]
            budget = cfg["max_len"] - 2
            if len(ids_full) > budget:
                text = tok.decode(ids_full[-budget:])
        if cfg["trunc"] == "headtail":
            ids_full = tok(text, add_special_tokens=False)["input_ids"]
            budget = cfg["max_len"] - 2
            if len(ids_full) > budget:
                h = budget * 3 // 4
                ids_full = ids_full[:h] + ids_full[-(budget - h):]
            text = tok.decode(ids_full)
        if cfg.get("prompt"):
            return tok("這段ESG文本是否含有承諾、證據、期程？", text,
                       truncation=True, max_length=cfg["max_len"])
        return tok(text, truncation=True, max_length=cfg["max_len"])

    def __getitem__(self, i):
        r = self.rows[i]
        cfg = self.cfg
        want_off = (bool(cfg.get("aux_span")) or cfg.get("aux_attn", 0) > 0
                    or cfg.get("aux_attn_year", 0) > 0
                    or cfg.get("aux_attn_ev", 0) > 0)
        enc = self._encode(r["data"], want_offsets=want_off,
                           company=str(r.get("company", "")))
        offsets = enc.pop("offset_mapping", None)
        if self.train and cfg.get("token_drop", 0) > 0:
            ids = enc["input_ids"]
            mid = self.tok.mask_token_id or self.tok.unk_token_id
            for j in range(1, len(ids) - 1):
                if random.random() < cfg["token_drop"]:
                    ids[j] = mid
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if want_off and "promise_status" in r:
            n = len(enc["input_ids"])
            for ch, col, gate in (("p", "promise_string", r.get("promise_status") == "Yes"),
                                  ("e", "evidence_string", r.get("evidence_status") == "Yes")):
                need = (ch in cfg.get("aux_span", "")
                        or cfg.get("aux_attn", 0) > 0
                        or (ch == "e" and cfg.get("aux_attn_ev", 0) > 0))
                if not need:
                    continue
                tgt = torch.full((n,), -100.0)
                sub = str(r.get(col, ""))
                pos = r["data"].find(sub) if (gate and sub) else -1
                if pos >= 0:
                    lo, hi = pos, pos + len(sub)
                    tgt = torch.zeros(n)
                    for ti, (a, b2) in enumerate(offsets):
                        if a == b2 == 0 and ti != 0:
                            tgt[ti] = -100.0  # special/pad token
                        elif a < hi and b2 > lo:
                            tgt[ti] = 1.0
                    tgt[0] = -100.0  # CLS
                item[f"aux_span_{ch}"] = tgt
        if cfg.get("aux_attn_year", 0) > 0:
            n = len(enc["input_ids"])
            tgt = torch.zeros(n)
            if offsets is not None:
                for m in re.finditer(r"(19|20)\d{2}|\d{1,3}\s*年|民國", r["data"]):
                    lo, hi = m.span()
                    for ti, (a, b2) in enumerate(offsets):
                        if a < hi and b2 > lo and not (a == b2 == 0):
                            tgt[ti] = 1.0
            tgt[0] = -100.0
            item["aux_span_y"] = tgt
        if cfg.get("aux_esg", 0) > 0 and "esg_type" in r:
            es = str(r.get("esg_type", ""))
            if es:
                item["aux_esg"] = torch.tensor(
                    [1.0 if L in es else 0.0 for L in "ESG"])
            else:
                item["aux_esg"] = torch.tensor([-100.0, -100.0, -100.0])
        if "promise_status" in r:  # 帶標籤
            t1 = HEAD_LABELS["T1"].index(r["promise_status"])
            t2 = HEAD_LABELS["T2"].index(r["evidence_status"]) \
                if r["promise_status"] == "Yes" and r["evidence_status"] in HEAD_LABELS["T2"] else -100
            t3 = HEAD_LABELS["T3"].index(r["evidence_quality"]) \
                if r["evidence_status"] == "Yes" and r["evidence_quality"] in HEAD_LABELS["T3"] else -100
            t4 = HEAD_LABELS["T4"].index(r["verification_timeline"]) \
                if r["promise_status"] == "Yes" and r["verification_timeline"] in HEAD_LABELS["T4"] else -100
            item["labels"] = torch.tensor([t1, t2, t3, t4])
        return item


def collate(batch, pad_id):
    keys = [k for k in batch[0] if k not in ("labels", "aux_esg")]
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {}
    for k in keys:
        pad = pad_id if k == "input_ids" else (-100 if k.startswith("aux_span") else 0)
        out[k] = torch.stack([
            F.pad(b[k], (0, maxlen - len(b[k])), value=float(pad) if k.startswith("aux_span") else pad)
            for b in batch])
    for k in ("labels", "aux_esg"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------- Model ----------------
class LoRALinear(nn.Module):
    """低秩 adapter：W·x + (alpha/r)·B(A(x))，base 凍結，只訓 A/B（B 初始化為 0）。"""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


class KANLayer(nn.Module):
    """Kolmogorov-Arnold 層：每條邊一個可學一維函數 φ_ij(x)=Σ_k c_ijk·RBF_k(x)（樣條的
    GPU 友善近似）+ SiLU base；節點只做加總。out_j = Σ_i φ_ij(x_i)。簽名同 nn.Linear。"""
    def __init__(self, d_in, d_out, grid=8, krange=2.0):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.register_buffer("centers", torch.linspace(-krange, krange, grid))
        self.sigma = 2 * krange / (grid - 1)
        self.spline_w = nn.Parameter(torch.randn(d_out, d_in, grid) * 0.1)
        self.base_w = nn.Parameter(torch.randn(d_out, d_in) * (d_in ** -0.5))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):                                  # x:(...,d_in)
        base = F.silu(x) @ self.base_w.t()
        phi = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.sigma) ** 2)
        spline = torch.einsum("...ig,oig->...o", phi, self.spline_w)
        return base + spline + self.bias


class RMSNorm(nn.Module):
    """RMSNorm（survey II.1）：x / rms(x) · g，無去均值/bias（比 LayerNorm 省一階統計）。"""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class GLUHead(nn.Module):
    """閘控線性單元頭（survey II.2）：Linear→(a,b) 切半→a·act(b)→Dropout→Linear(dims)。
    kind: glu(sigmoid)/swiglu(silu)/geglu(gelu)。對極淺分類頭引入乘性閘控非線性。"""
    def __init__(self, d_in, d_out, drop, kind="glu"):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 2 * d_in)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(d_in, d_out)
        self.act = {"glu": torch.sigmoid, "swiglu": F.silu, "geglu": F.gelu}[kind]

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.drop(a * self.act(b)))


class ArcCosHead(nn.Module):
    """ArcFace 餘弦頭：輸出 cos θ = normalize(x)·normalize(W)（無 bias/scale）；
    加性角 margin 與 scale s 在 task_loss 用 labels 施加（推論期無 margin，argmax 等價）。"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * (d_in ** -0.5))

    def forward(self, x):
        return F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t()


def supcon_loss(z, y, tau=0.1):
    """Supervised Contrastive（同標籤互為正樣本的 NT-Xent）。z:(N,d) 已 L2 normalize。
    關 autocast 走 fp32（否則 matmul 被自動轉回 fp16，−1e9 mask/logsumexp 溢位）。"""
    with torch.amp.autocast("cuda", enabled=False):
        z = z.float()
        N = z.size(0)
        if N < 2:
            return z.sum() * 0.0
        sim = z @ z.t() / tau
        eye = torch.eye(N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
        cnt = pos.sum(1)
        valid = cnt > 0
        if valid.sum() == 0:
            return z.sum() * 0.0
        loss = -(logp * pos.float()).sum(1)[valid] / cnt[valid].clamp(min=1)
        return loss.mean()


class GradReverse(torch.autograd.Function):
    """梯度反轉層（DANN）：前向恆等、反向乘 −λ → encoder 學域不變特徵。"""
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb=1.0):
    return GradReverse.apply(x, lamb)


class MultiHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # transformers 5 預設依 checkpoint dtype 載入（mdeberta=fp16 → GradScaler
        # 報 "Attempting to unscale FP16 gradients"）；強制 fp32 master weights。
        _ed = cfg.get("enc_dropout", 0.0)
        if _ed > 0:  # survey: 增大 encoder 內部 attention/hidden dropout（過擬合對症正則）
            self.enc = AutoModel.from_pretrained(
                cfg["model_name"], dtype=torch.float32,
                hidden_dropout_prob=_ed, attention_probs_dropout_prob=_ed)
        else:
            self.enc = AutoModel.from_pretrained(cfg["model_name"], dtype=torch.float32)
        h = self.enc.config.hidden_size       # 在 LoRA 包裝前取 hidden_size
        ri = int(cfg.get("reinit_layers", 0))
        if ri > 0 and cfg.get("lora_r", 0) == 0:
            for layer in self.enc.encoder.layer[-ri:]:
                for m in layer.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data.normal_(0.0, self.enc.config.initializer_range)
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.LayerNorm):
                        m.weight.data.fill_(1.0); m.bias.data.zero_()
        if cfg.get("lora_r", 0) > 0:
            # Mapping Networks 精神：凍結 base、只訓低秩 adapter（低維流形參數化，抗過擬合）
            # 手寫 LoRA（無 peft/torchao 依賴，對 VM 回收更穩）：注入 attention query/value
            r = int(cfg["lora_r"]); alpha = int(cfg.get("lora_alpha", 2 * r))
            for p in self.enc.parameters():
                p.requires_grad = False
            for mod in self.enc.modules():
                for nm in ("query", "value"):
                    sub = getattr(mod, nm, None)
                    if isinstance(sub, nn.Linear):
                        setattr(mod, nm, LoRALinear(sub, r, alpha))
        self.drop = nn.Dropout(cfg["dropout"])
        self.cond = cfg.get("cond_heads", False)
        self.t4_ordinal = cfg.get("t4_ordinal", False)
        self.graft_task = cfg.get("graft_task", "")          # 雙分支嫁接：目標任務（""=關）
        self.graft_mode = cfg.get("graft_mode", "concat")    # concat|replace
        self._graft_arch = cfg.get("graft_arch", "")         # branch-B 中間層架構
        dims = {t: len(HEAD_LABELS[t]) for t in TASKS}
        if self.t4_ordinal:
            dims["T4"] = 3  # 3 個累積門檻 logits（all-threshold ordinal）
        inp = {t: h for t in TASKS}
        if cfg.get("t3_diff"):
            inp["T3"] = h * 3  # [v_T3; v_mean; v_T3−v_mean]（attn_pt 限定）
        if self.cond:
            inp["T3"] = inp["T4"] = h + 4  # 拼 T1(2)+T2(2) logits（detach）
        if self.graft_task:   # concat → head 吃 [vs_A; vs_B]（2h）；replace → 只吃 vs_B（h）
            inp[self.graft_task] = (inp[self.graft_task] + h
                                    if self.graft_mode == "concat" else h)
        ha = cfg.get("head_act", "")
        kan_g = int(cfg.get("kan_head", 0))
        glu = cfg.get("head_glu", "")
        if kan_g > 0:
            # KAN 頭：可學一維樣條激活放在邊上（RBF 基近似 B-spline，GPU 友善）+ SiLU base
            self.heads = nn.ModuleDict(
                {t: KANLayer(inp[t], dims[t], grid=kan_g) for t in TASKS})
        elif glu:
            self.heads = nn.ModuleDict(
                {t: GLUHead(inp[t], dims[t], cfg["dropout"], kind=glu) for t in TASKS})
        elif ha:
            _ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
                     "silu": nn.SiLU, "mish": nn.Mish, "elu": nn.ELU,
                     "leakyrelu": nn.LeakyReLU}
            Act = _ACTS.get(ha, nn.GELU)
            self.heads = nn.ModuleDict({t: nn.Sequential(
                nn.Linear(inp[t], inp[t]), Act(), nn.Dropout(cfg["dropout"]),
                nn.Linear(inp[t], dims[t])) for t in TASKS})  # 2層+activation
        else:
            self.heads = nn.ModuleDict(
                {t: nn.Linear(inp[t], dims[t]) for t in TASKS})
        # ArcFace：指定任務的頭換成餘弦頭（margin 在 task_loss 施加）
        self.arcface = bool(cfg.get("arcface", False))
        self.arc_tasks = list(cfg.get("arcface_tasks", ("T1",)))
        if self.arcface:
            for t in self.arc_tasks:
                self.heads[t] = ArcCosHead(inp[t], dims[t])
        # SupCon：對指定任務的池化向量加投影頭 → 對比空間
        self.supcon = float(cfg.get("supcon", 0.0))
        self.supcon_tasks = list(cfg.get("supcon_tasks", ("T1",)))
        if self.supcon > 0:
            self.supcon_proj = nn.ModuleDict({t: nn.Sequential(
                nn.Linear(inp[t], inp[t]), nn.ReLU(), nn.Linear(inp[t], 128))
                for t in self.supcon_tasks})
        # DANN：域判別頭（source=train vs target=未標 test）+ 梯度反轉
        self.dann = float(cfg.get("dann", 0.0))
        if self.dann > 0:
            self.dom_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                          nn.Dropout(cfg["dropout"]), nn.Linear(h, 2))
        if cfg.get("learned_taskw"):
            self.task_logsig = nn.Parameter(torch.zeros(4))
        self.aux_head_w = cfg.get("aux_head", 0.0)   # 深度監督：注意力前的輔助頭
        self.aux_head_layer = int(cfg.get("aux_head_layer", 0))  # >0 = 接中間 encoder 層(淺層)
        if self.aux_head_w > 0:
            self.aux_heads = nn.ModuleDict(
                {t: nn.Linear(h, len(HEAD_LABELS[t])) for t in TASKS})
        self.t3diff = bool(cfg.get("t3_diff"))
        if cfg.get("t3_xattn") and cfg["pool"] == "attn_pt":
            self.t3xattn = nn.Linear(h * 2, 1)
        self.pool = cfg["pool"]
        self.emb_noise = cfg.get("emb_noise", 0.0)
        self.msdrop_k = int(cfg.get("msdrop_k", 0))
        self.keep_hs = cfg.get("keep_hs", False)
        self.keep_layers = list(cfg.get("keep_layers", []))  # 多層蒸餾：暴露指定 encoder 層
        sh = cfg.get("seq_head", "")
        self.seq_kind = sh
        n_rbl = int(cfg.get("resid_bilstm_n", 1))
        if cfg.get("resid_bilstm") and n_rbl > 1:
            # 更深：N 個獨立殘差 BiLSTM 塊（非 DEQ 權重共享）
            self.rseq_stack = nn.ModuleList(
                [nn.LSTM(h, h // 2, batch_first=True, bidirectional=True)
                 for _ in range(n_rbl)])
            self.rseq_ln_stack = nn.ModuleList(
                [nn.LayerNorm(h) for _ in range(n_rbl)])
        elif cfg.get("resid_bilstm") or cfg.get("deq_iters", 0) > 1:
            self.rseq = nn.LSTM(h, h // 2, batch_first=True, bidirectional=True)
            self.rseq_ln = nn.LayerNorm(h)
        self.deq_iters = int(cfg.get("deq_iters", 0))
        # 真 DEQ（深度平衡）：輸出 = 不動點 z*=LN(x + f(z*))，求根器解 + JFB 反傳（等效無限深）
        self.deq_solver = bool(cfg.get("deq_solver", False))
        if self.deq_solver:
            self.deq_f = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h),
                                       nn.GELU(), nn.Linear(h, h))
            self.deq_ln = nn.LayerNorm(h)
        # 真 Neural ODE：dh/dt=f(h)，固定步 RK4/Euler 積分 t∈[0,1]（連續深度殘差）
        self.ode_steps = int(cfg.get("ode_steps", 0))
        self.ode_solver = cfg.get("ode_solver", "rk4")
        if self.ode_steps > 0:
            self.ode_f = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h),
                                       nn.GELU(), nn.Linear(h, h))
        self.hyper_k = int(cfg.get("hyper_mix", 0))
        if self.hyper_k > 1:
            self.hyper_w = nn.Parameter(torch.zeros(len(TASKS), self.hyper_k))
        # 動態內容門控層融合：per-token 由內容決定末K層權重（vs hyper_mix 靜態純量）
        self.dyn_k = int(cfg.get("dyn_layer_k", 0))
        if self.dyn_k > 1:
            self.dyn_gate = nn.Linear(h, self.dyn_k)
        if cfg["pool"] == "attn_lc":
            self.lc_q = nn.ParameterDict({
                t: nn.Parameter(torch.randn(len(HEAD_LABELS[t]), h) * 0.02)
                for t in TASKS})
            self.lc_scale = h ** 0.5
        if cfg.get("resid_attn") and cfg["pool"] == "attn_pt":
            self.attn_pt2 = nn.ModuleDict({t: nn.Linear(h * 2, 1) for t in TASKS})
        if cfg.get("head_resid"):
            self.hres = nn.ModuleDict({t: nn.Sequential(
                nn.Linear(h, h), nn.GELU(), nn.Linear(h, h)) for t in TASKS})
            self.hres_ln = nn.ModuleDict({t: nn.LayerNorm(h) for t in TASKS})
        if sh == "bilstm":
            self.seq = nn.LSTM(h, h // 2, batch_first=True, bidirectional=True)
        elif sh == "bigru":
            self.seq = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
        elif sh == "textcnn":
            self.seq = nn.ModuleList([nn.Conv1d(h, h // 3, k, padding=k // 2)
                                      for k in (3, 5, 7)])  # 奇數核，長度守恆
        if self.pool == "attn":
            self.attn_scorer = nn.Linear(h, 1)
        elif self.pool == "attn_pt":
            self.attn_pt = nn.ModuleDict({t: nn.Linear(h, 1) for t in TASKS})
        if self.graft_task:
            # branch-B：不同架構的中間層（平行於 branch-A）為目標任務產 vs_B；attn_ptB 為其專屬池化
            a = self._graft_arch
            if a in ("bilstm", "rbilstm", "resid_bilstm"):
                self.graft_seq = nn.LSTM(h, h // 2, batch_first=True, bidirectional=True)
                self.graft_resid = a in ("rbilstm", "resid_bilstm")
                if self.graft_resid:
                    self.graft_ln = nn.LayerNorm(h)
            elif a == "bigru":
                self.graft_seq = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
                self.graft_resid = False
            elif a == "textcnn":
                self.graft_cnn = nn.ModuleList(
                    [nn.Conv1d(h, h // 3, k, padding=k // 2) for k in (3, 5, 7)])
            # a == "self" → 無序列精煉（平行池化 = self-ensemble baseline）
            self.attn_ptB = nn.Linear(h, 1)
        # Squeeze-and-Excitation：對每任務池化向量(768維)做通道門控重標定（近零成本，
        # bottleneck r 縮放 → ReLU → 還原 → sigmoid 門）。se_head=r（0=關）。
        self.se_r = int(cfg.get("se_head", 0))
        if self.se_r > 0:
            _b = max(8, h // self.se_r)
            self.se = nn.ModuleDict({t: nn.Sequential(
                nn.Linear(h, _b), nn.ReLU(), nn.Linear(_b, h), nn.Sigmoid())
                for t in TASKS})
        # head_norm：池化向量進 head 前正規化（ln=LayerNorm/rms=RMSNorm；survey II.1）
        hn = cfg.get("head_norm", "")
        if hn == "ln":
            self.head_ln = nn.ModuleDict({t: nn.LayerNorm(h) for t in TASKS})
        elif hn == "rms":
            self.head_ln = nn.ModuleDict({t: RMSNorm(h) for t in TASKS})
        if cfg.get("aux_esg", 0) > 0:
            self.esg_head = nn.Linear(h, 3)
        self.aux_span_cfg = cfg.get("aux_span", "")
        if "p" in self.aux_span_cfg:
            self.tok_p = nn.Linear(h, 1)
        if "e" in self.aux_span_cfg:
            self.tok_e = nn.Linear(h, 1)
        if cfg.get("fixup"):
            self._apply_fixup()

    def _apply_fixup(self):
        """Fixup/SkipInit：把附加殘差/門控塊的末層權重歸零 → 起點≈恆等
        （dyn_gate 歸零=末K層均勻平均；hres/deq_f/ode_f/attn_pt2 歸零=殘差不動）。
        目的：更穩的訓練起點、可能解鎖更深堆疊（survey IV.1）。"""
        mods = []
        if hasattr(self, "dyn_gate"):
            mods.append(self.dyn_gate)
        if hasattr(self, "hres"):
            mods += [self.hres[t][-1] for t in TASKS]
        if hasattr(self, "deq_f"):
            mods.append(self.deq_f[-1])
        if hasattr(self, "ode_f"):
            mods.append(self.ode_f[-1])
        if hasattr(self, "attn_pt2"):
            mods += [self.attn_pt2[t] for t in TASKS]
        if hasattr(self, "se"):
            mods += [self.se[t][-2] for t in TASKS]   # SE 末線性層（-1 是 Sigmoid）
        for m in mods:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, **kw):
        enc_kw = {k: v for k, v in kw.items() if k == "token_type_ids"}
        if (self.hyper_k > 1 or self.dyn_k > 1 or self.keep_layers
                or getattr(self, "aux_head_layer", 0) > 0):
            enc_kw["output_hidden_states"] = True
        if self.emb_noise > 0 and self.training:
            we = self.enc.get_input_embeddings()(input_ids)
            we = we + torch.randn_like(we) * self.emb_noise
            out = self.enc(inputs_embeds=we, attention_mask=attention_mask, **enc_kw)
        else:
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask, **enc_kw)
        hs = out.last_hidden_state
        self._aux_src = (out.hidden_states[self.aux_head_layer]
                         if getattr(self, "aux_head_layer", 0) > 0 else None)
        if self.dyn_k > 1:
            # 動態層融合：gate(內容)→softmax over 末K層→per-token 加權和
            layers = torch.stack(out.hidden_states[-self.dyn_k:], dim=2)  # [B,T,K,H]
            gate = torch.softmax(self.dyn_gate(hs), dim=-1)               # [B,T,K]
            hs = (layers * gate.unsqueeze(-1)).sum(2)                     # [B,T,H]
        hyper_layers = None
        if self.hyper_k > 1:
            hyper_layers = torch.stack(out.hidden_states[-self.hyper_k:], dim=0)
        if hasattr(self, "rseq_stack"):
            for lstm, ln in zip(self.rseq_stack, self.rseq_ln_stack):
                r, _ = lstm(hs)
                hs = ln(hs + r)
        elif hasattr(self, "rseq"):
            iters = max(1, self.deq_iters)
            for _ in range(iters):  # 權重共享展開（DEQ 實用近似；iters=1 = 殘差BiLSTM）
                r, _ = self.rseq(hs)
                hs = self.rseq_ln(hs + r)
        if self.deq_solver:
            # 真 DEQ：阻尼定點迭代解 z*=LN(x+f(z))（no_grad），再對收斂點走一步帶梯度（JFB）
            x0 = hs
            with torch.no_grad():
                z = x0
                for _ in range(max(2, self.deq_iters or 24)):
                    zn = self.deq_ln(x0 + self.deq_f(z))
                    if (zn - z).norm() / (z.norm() + 1e-6) < 1e-3:
                        z = zn; break
                    z = zn
            hs = self.deq_ln(x0 + self.deq_f(z.detach()))
        if self.ode_steps > 0:
            # 真 Neural ODE：RK4（或 Euler）固定步積分 dh/dt=f(h)
            h_ = hs; dt = 1.0 / self.ode_steps
            for _ in range(self.ode_steps):
                if self.ode_solver == "euler":
                    h_ = h_ + dt * self.ode_f(h_)
                else:
                    k1 = self.ode_f(h_)
                    k2 = self.ode_f(h_ + 0.5 * dt * k1)
                    k3 = self.ode_f(h_ + 0.5 * dt * k2)
                    k4 = self.ode_f(h_ + dt * k3)
                    h_ = h_ + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            hs = h_
        if self.seq_kind in ("bilstm", "bigru"):
            hs, _ = self.seq(hs)
        elif self.seq_kind == "textcnn":
            x = hs.transpose(1, 2)
            hs = torch.cat([F.relu(c(x)) for c in self.seq], dim=1).transpose(1, 2)
            if hs.size(-1) != self.enc.config.hidden_size:
                hs = F.pad(hs, (0, self.enc.config.hidden_size - hs.size(-1)))
        extras = {}
        if self.keep_hs:
            extras["_hs"] = hs
        if self.keep_layers:
            extras.update({f"_hs_L{l}": out.hidden_states[l] for l in self.keep_layers})
        if self.pool == "attn_lc":
            o = {}
            extras2 = dict(extras)
            for t in TASKS:
                q = self.lc_q[t]                       # (C,h)
                sc = torch.einsum("blh,ch->bcl", hs, q) / self.lc_scale
                sc = sc.masked_fill(attention_mask.unsqueeze(1) == 0, -1e4)
                aw = torch.softmax(sc, dim=-1)         # (B,C,L)
                vc = torch.einsum("bcl,blh->bch", aw, hs)
                vc = self.drop(vc)
                w_ = self.heads[t].weight              # (C,h)
                o[t] = (vc * w_.unsqueeze(0)).sum(-1) + self.heads[t].bias
            o.update(extras2)
            return o
        if self.pool == "attn_pt":
            vs, aws = {}, {}
            for ti_, t in enumerate(TASKS):
                ht = hs
                if hyper_layers is not None:
                    w = torch.softmax(self.hyper_w[ti_], dim=0)
                    ht = (hyper_layers * w.view(-1, 1, 1, 1)).sum(0)
                if t == "T3" and hasattr(self, "t3xattn"):
                    # T3 注意力 condition on T2 pooled（cross-attention，非 logits concat）
                    q = vs["T2"].unsqueeze(1).expand(-1, ht.size(1), -1)
                    sc = self.t3xattn(torch.cat([ht, q], dim=-1)).squeeze(-1)
                else:
                    sc = self.attn_pt[t](ht).squeeze(-1)
                sc = sc.masked_fill(attention_mask == 0, -1e4)
                if hasattr(self, "attn_pt2"):
                    aw1 = torch.softmax(sc, dim=-1)
                    v1 = (ht * aw1.unsqueeze(-1)).sum(1)
                    ctx = torch.cat([ht, v1.unsqueeze(1).expand_as(ht)], dim=-1)
                    sc = self.attn_pt2[t](ctx).squeeze(-1).masked_fill(
                        attention_mask == 0, -1e4) + sc  # logits 殘差
                aw = torch.softmax(sc, dim=-1)
                aws[t] = aw
                vs[t] = (ht * aw.unsqueeze(-1)).sum(1)
            if hasattr(self, "se"):           # SE 通道門控：vs *= sigmoid(MLP(vs))（h 維重標定）
                for t in TASKS:
                    vs[t] = vs[t] * self.se[t](vs[t])
            if hasattr(self, "head_ln"):      # 池化向量進 head 前 LayerNorm（穩定/降方差）
                for t in TASKS:
                    vs[t] = self.head_ln[t](vs[t])
            extras["_attn"] = aws["T1"]       # promise-KL 監督 T1 的注意力
            extras["_attn_t4"] = aws["T4"]    # 年份-KL 監督 T4 的注意力
            extras["_attn_t3"] = aws["T3"]    # 證據-KL 監督 T3 的注意力
            if hasattr(self, "aux_heads"):    # 深度監督：注意力前 masked-mean 池化輔助分類
                src = self._aux_src if self._aux_src is not None else hs  # 淺層 or 最終融合
                m_ = attention_mask.unsqueeze(-1).float()
                vmean_ = (src * m_).sum(1) / m_.sum(1).clamp(min=1e-6)
                extras["_aux"] = {t: self.aux_heads[t](self.drop(vmean_)) for t in TASKS}
            if hasattr(self, "esg_head"):
                extras["_esg"] = self.esg_head(hs[:, 0])
            if hasattr(self, "tok_p"):
                extras["_tok_p"] = self.tok_p(hs).squeeze(-1)
            if hasattr(self, "tok_e"):
                extras["_tok_e"] = self.tok_e(hs).squeeze(-1)
            if self.training and getattr(self, "_mixup_lam", None) is not None:
                lam, perm = self._mixup_lam, self._mixup_perm
                for t in TASKS:
                    vs[t] = lam * vs[t] + (1 - lam) * vs[t][perm]
            if hasattr(self, "hres"):
                for t in TASKS:
                    vs[t] = self.hres_ln[t](vs[t] + self.hres[t](vs[t]))
            if hasattr(self, "t3diff") and self.t3diff:
                m = attention_mask.unsqueeze(-1).float()
                vmean = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
                vs["T3"] = torch.cat([vs["T3"], vmean, vs["T3"] - vmean], dim=-1)
            if self.supcon > 0:
                extras["_supcon"] = {t: F.normalize(self.supcon_proj[t](vs[t]), dim=-1)
                                     for t in self.supcon_tasks}
            if self.dann > 0:                 # 域判別 logits（梯度反轉後）
                lam = getattr(self, "_dann_lambda", 1.0)
                extras["_dann"] = self.dom_head(grad_reverse(vs["T1"], lam))
            if self.graft_task:
                # 雙分支：branch-B 於共享 encoder 原始輸出（平行 branch-A 的 dyn）跑自己的中間層
                # → 專屬注意力池化 vs_B[tgt]；concat([vs_A;vs_B]) 或 replace。
                tgt = self.graft_task
                hb = out.last_hidden_state
                if hasattr(self, "graft_seq"):
                    r, _ = self.graft_seq(hb)
                    hb = self.graft_ln(hb + r) if getattr(self, "graft_resid", False) else r
                elif hasattr(self, "graft_cnn"):
                    x = hb.transpose(1, 2)
                    hb = torch.cat([F.relu(c(x)) for c in self.graft_cnn], dim=1).transpose(1, 2)
                    if hb.size(-1) != hs.size(-1):
                        hb = F.pad(hb, (0, hs.size(-1) - hb.size(-1)))
                scB = self.attn_ptB(hb).squeeze(-1).masked_fill(attention_mask == 0, -1e4)
                vsB = (hb * torch.softmax(scB, dim=-1).unsqueeze(-1)).sum(1)
                if self.graft_mode == "concat":      # head 吃 [vs_A; vs_B]（2h，過參數）
                    vs[tgt] = torch.cat([vs[tgt], vsB], dim=-1)
                elif self.graft_mode == "add":       # 殘差式相加（h 維，無額外頭參數）
                    vs[tgt] = vs[tgt] + vsB
                else:                                # replace：只用 vs_B（不用 A 的）
                    vs[tgt] = vsB
            if self.msdrop_k > 1 and self.training:
                # 多樣本 dropout：K 個 mask 平均 logits（降方差，推論期單次）
                o = {t: torch.stack([self.heads[t](self.drop(vs[t]))
                                     for _ in range(self.msdrop_k)]).mean(0)
                     for t in TASKS}
            else:
                o = {t: self.heads[t](self.drop(vs[t])) for t in TASKS}
            o.update(extras)
            return o
        if self.pool == "cls":
            v = hs[:, 0]
        elif self.pool == "attn":
            sc = self.attn_scorer(hs).squeeze(-1)
            sc = sc.masked_fill(attention_mask == 0, -1e4)
            aw = torch.softmax(sc, dim=-1)
            extras["_attn"] = aw
            v = (hs * aw.unsqueeze(-1)).sum(1)
        else:
            m = attention_mask.unsqueeze(-1).float()
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
        if hasattr(self, "esg_head"):
            extras["_esg"] = self.esg_head(v if self.pool != "attn" else hs[:, 0])
        if hasattr(self, "tok_p"):
            extras["_tok_p"] = self.tok_p(hs).squeeze(-1)
        if hasattr(self, "tok_e"):
            extras["_tok_e"] = self.tok_e(hs).squeeze(-1)
        v = self.drop(v)
        if not self.cond:
            o = {t: self.heads[t](v) for t in TASKS}
        else:
            l1, l2 = self.heads["T1"](v), self.heads["T2"](v)
            ctx = torch.cat([v, l1.detach(), l2.detach()], dim=-1)
            o = {"T1": l1, "T2": l2, "T3": self.heads["T3"](ctx),
                 "T4": self.heads["T4"](ctx)}
        o.update(extras)
        return o


def task_loss(logits, labels, cfg, cls_w, tensor_parts=None):
    """labels: (B,4)，-100 = 此列此任務不學（級聯遮罩）。
    tensor_parts: 傳 dict 進來則收集未 detach 的 per-task loss（Kendall 用）。"""
    total, parts = 0.0, {}
    for i, t in enumerate(TASKS):
        y = labels[:, i]
        mask = y != -100
        if mask.sum() == 0:
            continue
        lg, yy = logits[t][mask], y[mask]
        if cfg.get("arcface") and t in cfg.get("arcface_tasks", ("T1",)):
            # lg = cos θ；對目標類加性角 margin cos(θ+m)，再 ×s 做 CE（推論期無 margin）
            s, m = cfg.get("arc_s", 20.0), cfg.get("arc_m", 0.2)
            cos = lg.clamp(-1 + 1e-6, 1 - 1e-6)
            oh = F.one_hot(yy, cos.size(1)).float()
            marg = torch.cos(torch.acos(cos) + m * oh)
            ce = F.cross_entropy(s * marg, yy, weight=cls_w.get(t))
            total = total + cfg["task_w"][i] * ce
            parts[t] = float(ce.detach())
            if tensor_parts is not None:
                tensor_parts[t] = ce
            continue
        if t == "T3" and cfg.get("t3_focal", 0) > 0:
            g = cfg["t3_focal"]
            logp = F.log_softmax(lg, -1)
            pp = logp.exp()
            ce = F.nll_loss(((1 - pp) ** g) * logp, yy, weight=cls_w.get(t),
                            reduction="mean")
            total = total + cfg["task_w"][i] * ce
            parts[t] = float(ce.detach())
            continue
        if t == "T4" and cfg.get("t4_gce", 0) > 0:
            q = cfg["t4_gce"]
            py = F.softmax(lg, -1).gather(1, yy.unsqueeze(1)).squeeze(1)
            ce = ((1 - py.clamp(min=1e-6) ** q) / q).mean()
            total = total + cfg["task_w"][i] * ce
            parts[t] = float(ce.detach())
            if tensor_parts is not None:
                tensor_parts[t] = ce
            continue
        if t == "T4" and cfg.get("t4_ordinal"):
            cum = (yy.unsqueeze(1) > torch.arange(3, device=yy.device)).float()
            ce = F.binary_cross_entropy_with_logits(lg, cum)
            total = total + cfg["task_w"][i] * ce
            parts[t] = float(ce.detach())
            if tensor_parts is not None:
                tensor_parts[t] = ce
            continue
        if cfg.get("la_tau", 0) > 0 and t in cfg.get("la_tasks", TASKS) and "_la_prior" in cfg:
            lg = lg + cfg["la_tau"] * cfg["_la_prior"][t]  # in-loss logit adjustment（推論不加）
        if cfg["focal_gamma"] > 0:
            logp = F.log_softmax(lg, -1)
            p = logp.exp()
            ce = F.nll_loss(((1 - p) ** cfg["focal_gamma"]) * logp, yy,
                            weight=cls_w.get(t), reduction="mean")
        else:
            ce = F.cross_entropy(lg, yy, weight=cls_w.get(t),
                                 label_smoothing=cfg["label_smooth"])
        total = total + cfg["task_w"][i] * ce
        parts[t] = float(ce.detach())
        if tensor_parts is not None:
            tensor_parts[t] = ce
    return total, parts


class FGM:
    """word-embedding 對抗擾動（cfg['fgm_eps']>0 時啟用）。"""

    def __init__(self, model, eps):
        self.model, self.eps, self.backup = model, eps, {}

    def attack(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and "word_embeddings" in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = p.grad.norm()
                if norm and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


class Lion(torch.optim.Optimizer):
    """Lion（survey IV.2）：θ−=lr·sign(β1·m+(1−β1)g)；m=β2·m+(1−β2)g。LR 需更小、wd 更大。"""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p))
                upd = (m * b1 + p.grad * (1 - b1)).sign_()
                p.add_(upd, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


class AdaBelief(torch.optim.Optimizer):
    """AdaBelief（survey IV.2）：以「梯度偏離動量」(g−m)² 取代 Adam 的 g² 作二階矩 →
    在梯度可預測方向放大步長、雜訊方向縮小。其餘同 AdamW（解耦 weight decay）。"""

    def __init__(self, params, lr=3e-5, betas=(0.9, 0.999), eps=1e-12, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                st = self.state[p]
                st.setdefault("step", 0)
                m = st.setdefault("m", torch.zeros_like(p))
                s = st.setdefault("s", torch.zeros_like(p))
                st["step"] += 1
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                diff = p.grad - m
                s.mul_(b2).addcmul_(diff, diff, value=1 - b2).add_(g["eps"])
                bc1 = 1 - b1 ** st["step"]; bc2 = 1 - b2 ** st["step"]
                p.addcdiv_(m / bc1, (s / bc2).sqrt_().add_(g["eps"]), value=-g["lr"])


def _newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float(); X = X / (X.norm() + eps)
    tp = G.size(0) > G.size(1)
    if tp:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if tp else X


class MuonHybrid(torch.optim.Optimizer):
    """Muon（survey IV.2，2024-25）：2D 隱權重用動量正交化(Newton-Schulz)更新；
    embedding/norm/bias/head 仍走內建 AdamW 更新。單一優化器，與 GradScaler 相容。
    group['use_muon'] 標記。"""

    def __init__(self, groups, lr=0.02, momentum=0.95, adamw_lr=3e-5,
                 betas=(0.9, 0.999), weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, adamw_lr=adamw_lr, betas=betas,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            if g.get("use_muon"):
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    buf = st.setdefault("mom", torch.zeros_like(p))
                    buf.mul_(g["momentum"]).add_(p.grad)
                    gr = p.grad.add(buf, alpha=g["momentum"])
                    o = _newton_schulz5(gr, steps=g["ns_steps"]).type_as(p)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(o, alpha=-g["lr"] * scale)
            else:
                b1, b2 = g["betas"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    if g["weight_decay"]:
                        p.mul_(1 - g["adamw_lr"] * g["weight_decay"])
                    st = self.state[p]
                    st.setdefault("step", 0)
                    m = st.setdefault("m", torch.zeros_like(p))
                    v = st.setdefault("v", torch.zeros_like(p))
                    st["step"] += 1
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]; bc2 = 1 - b2 ** st["step"]
                    p.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(1e-8), value=-g["adamw_lr"])


def build_optimizer(model, cfg):
    opt_name = cfg.get("optim", "")
    if opt_name == "lion":
        return Lion([p for p in model.parameters() if p.requires_grad],
                    lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if opt_name == "muon":
        muon_p, adamw_p = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (muon_p if (p.ndim == 2 and "enc.embeddings" not in n
                        and not n.startswith("heads")) else adamw_p).append(p)
        return MuonHybrid([{"params": muon_p, "use_muon": True},
                           {"params": adamw_p, "use_muon": False}],
                          lr=cfg.get("muon_lr", 0.02), adamw_lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"])
    if not cfg["llrd"]:
        named = list(model.named_parameters())
        groups = [
            {"params": [p for n, p in named if n.startswith("enc")],
             "lr": cfg["lr"]},
            {"params": [p for n, p in named if n.startswith("heads")],
             "lr": cfg["lr"] * cfg["head_lr_mult"]},
        ]
        if cfg.get("train_added"):
            # 修正：把附加模組（attn_pt/rseq/dyn_gate/head_ln…）納入訓練
            # （舊行為漏訓、凍結於隨機初始化；LoRA 也需此）
            other = [p for n, p in named
                     if not n.startswith("enc") and not n.startswith("heads")]
            groups.append({"params": other, "lr": cfg["lr"] * cfg["head_lr_mult"]})
        if opt_name == "radam":      # survey IV.2：前期方差修正（暖機免顯式 warmup）
            return torch.optim.RAdam(groups, weight_decay=cfg["weight_decay"])
        if opt_name == "adabelief":
            return AdaBelief(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    # LLRD：每往下一層 lr × decay
    layers = model.enc.config.num_hidden_layers
    decay = cfg["llrd"]
    groups = []
    for n, p in model.named_parameters():
        if n.startswith("heads"):
            lr = cfg["lr"] * cfg["head_lr_mult"]
        elif "embeddings" in n:
            lr = cfg["lr"] * decay ** layers
        elif ".layer." in n:
            li = int(n.split(".layer.")[1].split(".")[0])
            lr = cfg["lr"] * decay ** (layers - 1 - li)
        else:
            lr = cfg["lr"]
        groups.append({"params": [p], "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def class_weights(df, cfg, dev):
    if cfg["class_weight"] != "balanced":
        return {}
    out = {}
    for t in TASKS:
        f = TASK_FIELDS[t]
        sub = df[df[f].isin(HEAD_LABELS[t])]
        if t in ("T2", "T4"):
            sub = sub[sub.promise_status == "Yes"]
        if t == "T3":
            sub = sub[sub.evidence_status == "Yes"]
        cnt = sub[f].value_counts()
        w = torch.tensor([len(sub) / (len(HEAD_LABELS[t]) * cnt.get(l, 1))
                          for l in HEAD_LABELS[t]], dtype=torch.float32, device=dev)
        if t == "T4" and cfg.get("t4_w2y_boost", 0):
            w[1] = w[1] * cfg["t4_w2y_boost"]  # index 1 = within_2_years
        out[t] = w
    return out


def la_priors(df, cfg, dev):
    """in-loss logit adjustment 的 log 先驗（PDF TEAM_10218: 訓練時 logit += τ·log p_prior，推論不加）。
    級聯遮罩同 class_weights（T2/T4 限 promise=Yes、T3 限 evidence=Yes）。"""
    out = {}
    for t in TASKS:
        f = TASK_FIELDS[t]
        sub = df[df[f].isin(HEAD_LABELS[t])]
        if t in ("T2", "T4"):
            sub = sub[sub.promise_status == "Yes"]
        if t == "T3":
            sub = sub[sub.evidence_status == "Yes"]
        cnt = sub[f].value_counts()
        tot = max(len(sub), 1)
        p = torch.tensor([max(int(cnt.get(l, 0)), 1) / tot for l in HEAD_LABELS[t]],
                         dtype=torch.float32, device=dev)
        out[t] = torch.log(p)
    return out


# ---------------- 訓練一折 ----------------
def train_one(tr_df, te_df, cfg, dev="cuda", return_model=False):
    seed_all(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = MultiHead(cfg).to(dev)
    _pcg_ckpt = cfg.get("pcgrad_ckpt", None)
    if _pcg_ckpt is None:   # auto：T4 全配方需檢查點防 OOM；proxy(len≤256) 不需
        _pcg_ckpt = (cfg["max_len"] >= 320 or cfg.get("batch", 16) > 16)
    if cfg.get("pcgrad") and _pcg_ckpt:
        # PCGrad 對同一前向圖做 4 次 autograd.grad(retain_graph)，會同時保留全部 activation
        # → batch16/len384 在 14.5G T4 OOM。啟用 encoder 梯度檢查點（backward 重算、不存中間
        # activation）把峰值記憶體從 ~7G 降到 ~1-2G；單前向語義不變（dropout RNG 由檢查點保存）。
        # 但 checkpointing 使每折慢 ~2×；L4(24G) 全配方可關（pcgrad_ckpt=False）跑更快。
        try:
            model.enc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.enc.gradient_checkpointing_enable()
    pad = tok.pad_token_id or 0
    es_loader, es_df, best_es, best_state = None, None, -1.0, None
    if cfg.get("early_stop"):
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.get("es_frac", 0.15),
                                random_state=cfg["seed"])
        tri, esi = next(gss.split(tr_df, groups=tr_df["ticker"]))
        es_df = tr_df.iloc[esi]
        tr_df = tr_df.iloc[tri]
        es_loader = DataLoader(TextDS(es_df, tok, cfg), batch_size=64,
                               shuffle=False, collate_fn=lambda b: collate(b, pad))
    dl_tr = DataLoader(TextDS(tr_df, tok, cfg, train=True), batch_size=cfg["batch"],
                       shuffle=True, collate_fn=lambda b: collate(b, pad),
                       num_workers=2, drop_last=False)
    dl_te = DataLoader(TextDS(te_df, tok, cfg), batch_size=64,
                       shuffle=False, collate_fn=lambda b: collate(b, pad))
    dann_iter = None
    if cfg.get("dann", 0) > 0 and cfg.get("_target_df") is not None:
        tgt_loader = DataLoader(TextDS(cfg["_target_df"], tok, cfg), batch_size=cfg["batch"],
                                shuffle=True, collate_fn=lambda b: collate(b, pad),
                                num_workers=2, drop_last=True)
        def _cyc(dl):
            while True:
                for x in dl:
                    yield x
        dann_iter = _cyc(tgt_loader)
    fm_iter, fm_mask_id = None, None
    if cfg.get("fixmatch", 0) > 0 and cfg.get("_target_df") is not None:
        # FixMatch 半監督：未標 test 的弱視圖(原文)取高信心偽標 → 監督強視圖(token masking)
        fm_loader = DataLoader(TextDS(cfg["_target_df"], tok, cfg), batch_size=cfg["batch"],
                               shuffle=True, collate_fn=lambda b: collate(b, pad),
                               num_workers=2, drop_last=True)
        def _cyc2(dl):
            while True:
                for x in dl:
                    yield x
        fm_iter = _cyc2(fm_loader)
        fm_mask_id = tok.mask_token_id
        fm_special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.mask_token_id}
    opt = build_optimizer(model, cfg)
    steps = math.ceil(len(dl_tr) / cfg["grad_accum"]) * cfg["epochs"]
    if cfg.get("onecycle"):
        # One-Cycle（survey IV.3）：暖機升到峰值後 cosine 退火到近零（super-convergence 形狀）
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps,
            pct_start=max(0.05, cfg["warmup_frac"]), anneal_strategy="cos",
            div_factor=10.0, final_div_factor=100.0)
    elif cfg.get("cosine"):
        import math as _m
        wu_ = max(1, int(steps * cfg["warmup_frac"]))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / wu_, 1.0) * 0.5
            * (1 + _m.cos(_m.pi * min(1.0, max(0, s - wu_) / max(1, steps - wu_)))))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / max(1, int(steps * cfg["warmup_frac"])), 1.0)
            * max(0.0, (steps - s) / max(1, steps - int(steps * cfg["warmup_frac"]))))
    scaler = torch.amp.GradScaler("cuda")
    cls_w = class_weights(tr_df, cfg, dev)
    if cfg.get("la_tau", 0) > 0:
        cfg["_la_prior"] = la_priors(tr_df, cfg, dev)
    fgm = FGM(model, cfg["fgm_eps"]) if cfg["fgm_eps"] > 0 else None

    K_soup = int(cfg.get("ckpt_soup", 0))
    ema = None
    if cfg.get("ema", 0) > 0:
        ema = {n: p_.detach().clone() for n, p_ in model.named_parameters()
               if p_.requires_grad}
    soup = []
    # SAM 與 AMP/GradScaler 的兩次 unscale 衝突 → NaN；sam_rho>0 時整步走純 fp32（關 autocast、
    # 不過 scaler），數值乾淨。其餘配方維持 AMP。
    use_amp = not (cfg.get("sam_rho", 0) > 0 or cfg.get("pcgrad"))
    model.train()
    gstep = 0
    for ep in range(cfg["epochs"]):
        for step, b in enumerate(dl_tr):
            gstep += 1
            if dann_iter is not None:    # GRL λ 由 0 平滑升到 1（DANN 標準排程）
                p_ = gstep / max(1, steps)
                model._dann_lambda = 2.0 / (1.0 + math.exp(-10 * p_)) - 1.0
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            aux_p = b.pop("aux_span_p", None)
            aux_e = b.pop("aux_span_e", None)
            aux_y = b.pop("aux_span_y", None)
            aux_esg = b.pop("aux_esg", None)
            anneal = 1.0
            if cfg.get("aux_attn_anneal"):
                anneal = max(0.0, 1.0 - gstep / max(1, steps))
            mix_lam, mix_perm = None, None
            if cfg.get("mixup_alpha", 0) > 0:
                import numpy as _np
                mix_lam = float(_np.random.beta(cfg["mixup_alpha"],
                                                cfg["mixup_alpha"]))
                mix_perm = torch.randperm(labels.size(0), device=labels.device)
                model._mixup_lam, model._mixup_perm = mix_lam, mix_perm
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**b)
                if cfg.get("learned_taskw"):
                    tp = {}
                    _, _ = task_loss(logits, labels, cfg, cls_w, tensor_parts=tp)
                    loss = 0.0
                    for i_, t_ in enumerate(TASKS):
                        if t_ in tp:
                            loss = loss + torch.exp(-model.task_logsig[i_]) * tp[t_]                                    + 0.5 * model.task_logsig[i_]
                else:
                    pc_tp = {} if cfg.get("pcgrad") else None
                    loss, _ = task_loss(logits, labels, cfg, cls_w, tensor_parts=pc_tp)
                    if mix_lam is not None:
                        loss2, _ = task_loss(logits, labels[mix_perm], cfg, cls_w)
                        loss = mix_lam * loss + (1 - mix_lam) * loss2
                if mix_lam is not None:
                    model._mixup_lam = None
                if cfg.get("attn_rdrop", 0) > 0 and "_attn" in logits:
                    logits_b = model(**b)
                    kl_a = 0.0
                    for key in ("_attn", "_attn_t4", "_attn_t3"):
                        if key in logits and key in logits_b:
                            a1 = logits[key].clamp(min=1e-8).log()
                            a2 = logits_b[key].clamp(min=1e-8).log()
                            kl_a = kl_a + (F.kl_div(a1, a2, log_target=True,
                                                    reduction="batchmean")
                                           + F.kl_div(a2, a1, log_target=True,
                                                      reduction="batchmean")) / 2
                    loss = loss + cfg["attn_rdrop"] * kl_a
                for tgt, key in ((aux_p, "_tok_p"), (aux_e, "_tok_e")):
                    if tgt is not None and key in logits:
                        mk = tgt != -100
                        if mk.any():
                            loss = loss + cfg["aux_w"] * F.binary_cross_entropy_with_logits(
                                logits[key][mk], tgt[mk])
                if aux_esg is not None and "_esg" in logits:
                    mk = aux_esg[:, 0] != -100
                    if mk.any():
                        loss = loss + cfg["aux_esg"] * F.binary_cross_entropy_with_logits(
                            logits["_esg"][mk], aux_esg[mk])
                if cfg.get("aux_attn_ev", 0) > 0 and "_attn_t3" in logits and aux_e is not None:
                    se = (aux_e == 1.0).float()
                    re_ = se.sum(-1) > 0
                    if re_.any():
                        tgt_e = se[re_] / se[re_].sum(-1, keepdim=True)
                        awe = logits["_attn_t3"][re_].clamp(min=1e-8)
                        loss = loss + anneal * cfg["aux_attn_ev"] * (
                            tgt_e * (tgt_e.clamp(min=1e-8).log() - awe.log())
                        ).sum(-1).mean()
                if cfg.get("aux_attn_year", 0) > 0 and "_attn_t4" in logits and aux_y is not None:
                    sy = (aux_y == 1.0).float()
                    ry = sy.sum(-1) > 0
                    if ry.any():
                        tgt_y = sy[ry] / sy[ry].sum(-1, keepdim=True)
                        awy = logits["_attn_t4"][ry].clamp(min=1e-8)
                        loss = loss + anneal * cfg["aux_attn_year"] * (
                            tgt_y * (tgt_y.clamp(min=1e-8).log() - awy.log())
                        ).sum(-1).mean()
                if cfg.get("aux_head", 0) > 0 and "_aux" in logits:
                    aux_l, _ = task_loss(logits["_aux"], labels, cfg, cls_w)
                    loss = loss + cfg["aux_head"] * aux_l   # 深度監督輔助損失
                if cfg.get("supcon", 0) > 0 and "_supcon" in logits:
                    for t in cfg.get("supcon_tasks", ("T1",)):
                        if t not in logits["_supcon"]:
                            continue
                        ti = TASKS.index(t); yv = labels[:, ti]; mk = yv != -100
                        if mk.sum() > 1:
                            loss = loss + cfg["supcon"] * supcon_loss(
                                logits["_supcon"][t][mk], yv[mk], cfg.get("supcon_tau", 0.1))
                if dann_iter is not None and "_dann" in logits:
                    # DANN：source(域0) + target(域1) 的域判別損失（GRL 已在 forward 反轉梯度）
                    ds = logits["_dann"]
                    dl_s = F.cross_entropy(ds, torch.zeros(ds.size(0), dtype=torch.long, device=dev))
                    tb = next(dann_iter)
                    tb = {k: v.to(dev) for k, v in tb.items()
                          if k in ("input_ids", "attention_mask", "token_type_ids")}
                    dt = model(**tb)["_dann"]
                    dl_t = F.cross_entropy(dt, torch.ones(dt.size(0), dtype=torch.long, device=dev))
                    loss = loss + cfg["dann"] * (dl_s + dl_t)
                if fm_iter is not None:
                    # FixMatch：弱視圖(原 token)→softmax 取信心≥τ 的硬偽標 → CE 監督強視圖
                    # (隨機 token→[MASK] 強增強)。機制異於硬偽標(無教師重訓、即時一致性)。
                    fb = next(fm_iter)
                    fb = {k: v.to(dev) for k, v in fb.items()
                          if k in ("input_ids", "attention_mask", "token_type_ids")}
                    with torch.no_grad():
                        weak = model(**fb)
                    ids = fb["input_ids"].clone()
                    am = fb["attention_mask"].bool()
                    keep = torch.ones_like(ids, dtype=torch.bool)
                    for sid in fm_special:
                        keep &= ids != sid
                    rm = (torch.rand(ids.shape, device=dev) < cfg.get("fm_mask", 0.15)) & am & keep
                    ids[rm] = fm_mask_id
                    strong = model(**{**fb, "input_ids": ids})
                    fm_loss = 0.0
                    for t in cfg.get("fixmatch_tasks", ("T1",)):
                        pw = F.softmax(weak[t].float(), -1)
                        conf, pl = pw.max(-1)
                        sel = conf >= cfg.get("fm_tau", 0.95)
                        if sel.any():
                            fm_loss = fm_loss + F.cross_entropy(strong[t][sel], pl[sel])
                    if not isinstance(fm_loss, float):
                        loss = loss + cfg["fixmatch"] * fm_loss
                if cfg.get("aux_attn", 0) > 0 and "_attn" in logits and aux_p is not None:
                    sp = (aux_p == 1.0).float()
                    if cfg.get("aux_attn_union") and aux_e is not None:
                        sp = torch.clamp(sp + (aux_e == 1.0).float(), max=1.0)
                    rows = sp.sum(-1) > 0
                    if rows.any():
                        tgt_d = sp[rows] / sp[rows].sum(-1, keepdim=True)
                        aw = logits["_attn"][rows].clamp(min=1e-8)
                        loss = loss + anneal * cfg["aux_attn"] * (
                            tgt_d * (tgt_d.clamp(min=1e-8).log() - aw.log())
                        ).sum(-1).mean()
                if cfg["rdrop_alpha"] > 0:
                    logits2 = model(**b)
                    loss2, _ = task_loss(logits2, labels, cfg, cls_w)
                    kl = 0.0
                    for i, t in enumerate(TASKS):
                        m = labels[:, i] != -100
                        if m.sum() == 0:
                            continue
                        p, q = F.log_softmax(logits[t][m], -1), F.log_softmax(logits2[t][m], -1)
                        kl = kl + (F.kl_div(p, q, log_target=True, reduction="batchmean")
                                   + F.kl_div(q, p, log_target=True, reduction="batchmean")) / 2
                    loss = (loss + loss2) / 2 + cfg["rdrop_alpha"] * kl
            sam_done = False
            if cfg.get("sam_rho", 0) > 0:
                # 純 fp32 SAM（use_amp=False，loss 已是 fp32）：①backward 求擾動方向 e=ρ·g/‖g‖
                # → ②w+e → ③第二次 fp32 backward 求 SAM 梯度 → ④還原 w；由共用區塊直接 opt.step。
                loss.backward()
                with torch.no_grad():
                    gn = torch.norm(torch.stack([
                        p_.grad.norm() for p_ in model.parameters()
                        if p_.grad is not None]))
                    eps = {}
                    for n_, p_ in model.named_parameters():
                        if p_.grad is not None:
                            e_ = cfg["sam_rho"] * p_.grad / (gn + 1e-12)
                            p_.add_(e_); eps[n_] = e_
                opt.zero_grad()
                logits_s = model(**b)                       # fp32（autocast 已關）
                loss_s, _ = task_loss(logits_s, labels, cfg, cls_w)
                (loss_s / cfg["grad_accum"]).backward()
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        if n_ in eps:
                            p_.sub_(eps[n_])
                sam_done = True
            pcgrad_done = False
            if cfg.get("pcgrad") and pc_tp:
                # PCGrad（純 fp32）：取 4 任務加權 CE 的各自梯度（扁平成單向量以省 Python/GPU-sync
                # 開銷），兩兩衝突(dot<0)時把 g_i 投影到 g_j 法平面去負遷移；aux 梯度照常加。
                shared = [p for p in model.parameters() if p.requires_grad]
                shapes = [p.shape for p in shared]
                numels = [p.numel() for p in shared]

                def _flat(gt):
                    return torch.cat([(g if g is not None else torch.zeros(n, device=dev))
                                      .reshape(-1) for g, n in zip(gt, numels)])
                objs = [cfg["task_w"][TASKS.index(t)] * pc_tp[t]
                        for t in TASKS if t in pc_tp]
                task_total = sum(objs)
                aux_loss = loss - task_total      # 非任務 CE 的附加損失（共圖、可微）
                flats = [_flat(torch.autograd.grad(o, shared, retain_graph=True,
                                                   allow_unused=True)) for o in objs]
                ag = _flat(torch.autograd.grad(aux_loss, shared, retain_graph=False,
                                               allow_unused=True))
                sq = [f.dot(f) + 1e-12 for f in flats]
                order = list(range(len(flats)))
                final = ag.clone()
                for i in range(len(flats)):
                    pi = flats[i].clone()
                    random.shuffle(order)
                    for j in order:
                        if i == j:
                            continue
                        d = pi.dot(flats[j])
                        if d < 0:
                            pi = pi - (d / sq[j]) * flats[j]
                    final += pi
                off = 0
                for p_, sh, n in zip(shared, shapes, numels):
                    p_.grad = final[off:off + n].view(sh).clone()
                    off += n
                pcgrad_done = True
            if not sam_done and not pcgrad_done:
                if use_amp:
                    scaler.scale(loss / cfg["grad_accum"]).backward()
                else:
                    (loss / cfg["grad_accum"]).backward()
            if fgm is not None:
                fgm.attack()
                with torch.amp.autocast("cuda"):
                    adv_loss, _ = task_loss(model(**b), labels, cfg, cls_w)
                scaler.scale(adv_loss / cfg["grad_accum"]).backward()
                fgm.restore()
            if step + 1 == len(dl_tr) and K_soup > 0 and ep >= cfg["epochs"] - K_soup:
                soup.append({k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()})
            if (step + 1) % cfg["grad_accum"] == 0 or step + 1 == len(dl_tr):
                if use_amp:
                    if cfg.get("grad_clip", 0) > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg["grad_clip"])
                    scaler.step(opt); scaler.update()
                else:                                       # 純 fp32（SAM）路徑
                    if cfg.get("grad_clip", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg["grad_clip"])
                    opt.step()
                opt.zero_grad(); sched.step()
                if ema is not None:
                    d = cfg["ema"]
                    with torch.no_grad():
                        for n, p_ in model.named_parameters():
                            if n in ema:
                                ema[n].mul_(d).add_(p_.detach(), alpha=1 - d)
        if es_loader is not None:           # early stopping：逐 epoch 評 val' 存最佳
            model.eval()
            ep_probs = {t: [] for t in TASKS}
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for bb in es_loader:
                    bb = {k: v.to(dev) for k, v in bb.items()
                          if k != "labels" and not k.startswith("aux_")}
                    lg = model(**bb)
                    for t in TASKS:
                        ep_probs[t].append(F.softmax(lg[t].float(), -1).cpu())
            ep_probs = {t: torch.cat(v).numpy() for t, v in ep_probs.items()}
            sc = score_frames(es_df, cascade_decode(ep_probs, len(es_df)))["score"]
            if sc > best_es:
                best_es = sc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            model.train()

    if cfg.get("early_stop") and best_state is not None:
        model.load_state_dict(best_state)   # 還原最佳 epoch 權重
    # checkpoint soup：後 K 個 epoch 快照權重平均（攻 seed 方差，零額外訓練）
    if K_soup > 0 and len(soup) >= 1:
        avg_sd = {k: sum(sd[k].float() for sd in soup) / len(soup)
                  for k in soup[0]}
        model.load_state_dict(avg_sd)
    # chunk 推論：>max_len 文檔做 head/tail 雙視窗、機率取 elementwise max 再歸一
    # 推論：回傳各頭 softmax 機率（EMA 啟用時換上影子權重）
    if ema is not None:
        with torch.no_grad():
            for n, p_ in model.named_parameters():
                if n in ema:
                    p_.copy_(ema[n])
    model.eval()
    probs = {t: [] for t in TASKS}
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for b in dl_te:
            b = {k: v.to(dev) for k, v in b.items()
                 if k != "labels" and not k.startswith("aux_")}
            logits = model(**b)
            for t in TASKS:
                lg = logits[t].float()
                if t == "T4" and cfg.get("t4_ordinal"):
                    sgm = torch.cummin(torch.sigmoid(lg), dim=-1).values
                    p = torch.stack([1 - sgm[:, 0], sgm[:, 0] - sgm[:, 1],
                                     sgm[:, 1] - sgm[:, 2], sgm[:, 2]], dim=-1)
                    p = p.clamp(min=1e-6)
                    probs[t].append((p / p.sum(-1, keepdim=True)).cpu())
                else:
                    probs[t].append(F.softmax(lg, -1).cpu())
    out_probs = {t: torch.cat(v).numpy() for t, v in probs.items()}
    if cfg.get("tta_trunc"):
        cfg_tail = {**cfg, "trunc": "tail"}
        dl_t = DataLoader(TextDS(te_df, tok, cfg_tail), batch_size=64,
                          shuffle=False, collate_fn=lambda b: collate(b, pad))
        pb = {t: [] for t in TASKS}
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for b in dl_t:
                b = {k: v.to(dev) for k, v in b.items()
                     if k != "labels" and not k.startswith("aux_")}
                lg = model(**b)
                for t in TASKS:
                    pb[t].append(F.softmax(lg[t].float(), -1).cpu())
        for t in TASKS:
            out_probs[t] = (out_probs[t] + torch.cat(pb[t]).numpy()) / 2
    if cfg.get("tta_csv"):
        bt = pd.read_csv(cfg["tta_csv"], keep_default_na=False)
        btmap = dict(zip(bt["id"].astype(str), bt["data_bt"]))
        te_bt = te_df.copy()
        te_bt["data"] = [btmap.get(str(i), d) for i, d in
                         zip(te_bt["id"], te_bt["data"])]
        dl_bt = DataLoader(TextDS(te_bt, tok, cfg), batch_size=64, shuffle=False,
                           collate_fn=lambda b: collate(b, pad))
        probs_bt = {t: [] for t in TASKS}
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for b in dl_bt:
                b = {k: v.to(dev) for k, v in b.items()
                     if k != "labels" and not k.startswith("aux_")}
                lg = model(**b)
                for t in TASKS:
                    probs_bt[t].append(F.softmax(lg[t].float(), -1).cpu())
        for t in TASKS:
            out_probs[t] = (out_probs[t] + torch.cat(probs_bt[t]).numpy()) / 2
    if cfg.get("chunk_infer"):
        cfg_tail = {**cfg, "trunc": "tail"}
        dl_te2 = DataLoader(TextDS(te_df, tok, cfg_tail), batch_size=64,
                            shuffle=False, collate_fn=lambda b: collate(b, pad))
        probs2 = {t: [] for t in TASKS}
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for b in dl_te2:
                b = {k: v.to(dev) for k, v in b.items()
                     if k != "labels" and not k.startswith("aux_")}
                lg = model(**b)
                for t in TASKS:
                    probs2[t].append(F.softmax(lg[t].float(), -1).cpu())
        for t in TASKS:
            p2 = torch.cat(probs2[t]).numpy()
            pm = np.maximum(out_probs[t], p2)
            out_probs[t] = pm / pm.sum(axis=1, keepdims=True)
    if return_model:
        return out_probs, model, tok
    del model
    torch.cuda.empty_cache()
    return out_probs


def tent_adapt(model, df, tok, cfg, dev, lr=1e-3, epochs=1, w=(0.2, 0.3, 0.35, 0.15)):
    """TENT 測試時適應：凍結除 LayerNorm 仿射外的全部參數，在目標(oval/test)上最小化
    級聯預測熵（無監督）→ 適應 release 分布偏移。回傳適應後 model（再 predict 目標）。"""
    pad = tok.pad_token_id or 0
    ln_params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            ln_params += [m.weight, m.bias]
    ln_ids = set(id(p) for p in ln_params)
    for p in model.parameters():
        p.requires_grad_(id(p) in ln_ids)
    opt = torch.optim.Adam(ln_params, lr=lr)
    loader = DataLoader(TextDS(df, tok, cfg), batch_size=cfg["batch"], shuffle=True,
                        collate_fn=lambda b: collate(b, pad), drop_last=True)
    model.eval()                       # LN 無 running stat；eval 關 dropout → 熵確定性
    for _ in range(epochs):
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()
                 if k in ("input_ids", "attention_mask", "token_type_ids")}
            with torch.amp.autocast("cuda"):
                lg = model(**b)
                ent = 0.0
                for j, t in enumerate(TASKS):
                    p = F.softmax(lg[t].float(), -1)
                    ent = ent + w[j] * (-(p * torch.log(p.clamp(min=1e-8))).sum(-1).mean())
            opt.zero_grad(); ent.backward(); opt.step()
    return model


def cascade_decode(probs, n):
    """probs[t]: (n, n_head_classes) → 級聯預測 DataFrame。"""
    rows = []
    for i in range(n):
        t1 = HEAD_LABELS["T1"][int(probs["T1"][i].argmax())]
        if t1 == "No":
            rows.append({"promise_status": "No", "evidence_status": "N/A",
                         "evidence_quality": "N/A", "verification_timeline": "N/A"})
            continue
        t2 = HEAD_LABELS["T2"][int(probs["T2"][i].argmax())]
        t3 = HEAD_LABELS["T3"][int(probs["T3"][i].argmax())] if t2 == "Yes" else "N/A"
        t4 = HEAD_LABELS["T4"][int(probs["T4"][i].argmax())]
        rows.append({"promise_status": t1, "evidence_status": t2,
                     "evidence_quality": t3, "verification_timeline": t4})
    return pd.DataFrame(rows)


DEFAULTS = dict(
    model_name="hfl/chinese-macbert-base", max_len=384, pool="mean",
    lr=2e-5, head_lr_mult=1.0, epochs=6, batch=16, warmup_frac=0.1,
    weight_decay=0.01, seed=42, task_w=[1.0, 1.0, 1.0, 1.0],
    class_weight=None, focal_gamma=0.0, label_smooth=0.0, rdrop_alpha=0.0,
    fgm_eps=0.0, llrd=0.0, grad_accum=1, trunc="head", aug_csv=None,
    prompt=False, dropout=0.1, norm=False, cond_heads=False, t4_ordinal=False,
    aux_esg=0.0, aux_span="", aux_w=0.2, aux_attn=0.0, aux_attn_union=False,
    aux_attn_year=0.0, aux_attn_ev=0.0, t4_w2y_boost=0.0, ema=0.0, tta_trunc=False,
    emb_noise=0.0, token_drop=0.0, seq_head="", keep_hs=False,
    t4_gce=0.0, aux_attn_anneal=False, attn_rdrop=0.0, learned_taskw=False,
    company_prefix=False, t3_diff=False, chunk_infer=False,
    head_resid=False, resid_bilstm=False, tta_csv=None,
    hyper_mix=0, deq_iters=0, resid_attn=False,
    reinit_layers=0, mixup_alpha=0.0, sam_rho=0.0, cosine=False,
    ckpt_soup=0, grad_clip=0.0, t3_focal=0.0, t3_xattn=False,
    resid_bilstm_n=1, dyn_layer_k=0, msdrop_k=0, keep_layers=[],
    deq_solver=False, ode_steps=0, ode_solver="rk4", kan_head=0, gmlp=False,
    supcon=0.0, supcon_tasks=("T1",), supcon_tau=0.1,
    arcface=False, arcface_tasks=("T1",), arc_s=20.0, arc_m=0.2,
    dann=0.0, dann_layers=1,
    se_head=0, fixup=False, pcgrad=False, pcgrad_ckpt=None,
    fixmatch=0.0, fixmatch_tasks=("T1",), fm_tau=0.95, fm_mask=0.15,
    head_glu="", head_norm="", onecycle=False, optim="",
    graft_task="", graft_arch="", graft_mode="concat",
    early_stop=False, es_frac=0.15,
    lora_r=0, lora_alpha=0, train_added=False, head_act="", enc_dropout=0.0, aux_head=0.0, aux_head_layer=0,
)


def run_cv(user_cfg, train_path="/content/train.csv", n_folds=3,
           out_jsonl="/content/results.jsonl", oof_dir="/content/oof"):
    """3-fold GroupKFold(ticker) pooled-OOF。回傳 report dict 並落盤。"""
    cfg = {**DEFAULTS, **user_cfg}
    assert "name" in cfg
    os.makedirs(oof_dir, exist_ok=True)
    df = load_df(train_path)
    aug = load_df(cfg["aug_csv"]) if cfg["aug_csv"] else None
    gkf = GroupKFold(n_splits=n_folds)
    oof_probs = {t: np.zeros((len(df), len(HEAD_LABELS[t])), dtype=np.float32)
                 for t in TASKS}
    t0 = time.time()
    for fi, (tr, te) in enumerate(gkf.split(df, groups=df["ticker"])):
        fold_cache = f"{oof_dir}/{cfg['name']}_fold{fi}.npz"
        if os.path.exists(fold_cache):  # 折級斷點（VM 回收後還原）
            z = np.load(fold_cache)
            for t in TASKS:
                oof_probs[t][te] = z[t]
            print(f"[{cfg['name']}] fold{fi} cached", flush=True)
            continue
        tr_df = df.iloc[tr]
        if aug is not None:  # 增強列只進訓練折：id 任一 token 命中 OOF 原始 id 即排除
            # 格式涵蓋 "11000_E4_02"、"claude_syn_01"、"10432_x10348_claude_..."（雙源）
            te_ids = set(df.iloc[te]["id"].astype(str))
            toks = aug["id"].astype(str).str.replace("x", "", regex=False).str.split("_")
            keep = ~toks.apply(lambda ts: any(t in te_ids for t in ts))
            tr_df = pd.concat([tr_df, aug[keep]], ignore_index=True)
        fold_cfg = {**cfg, "seed": cfg["seed"] + fi}
        probs = train_one(tr_df, df.iloc[te], fold_cfg)
        for t in TASKS:
            oof_probs[t][te] = probs[t]
        np.savez(fold_cache, **probs)
        import shutil
        shutil.make_archive("/content/oof_backup", "gztar", oof_dir)
        print(f"[{cfg['name']}] fold{fi} done {time.time()-t0:.0f}s", flush=True)
    pred = cascade_decode(oof_probs, len(df))
    rep = score_frames(df, pred)
    rec = {"name": cfg["name"], "cfg": {k: v for k, v in cfg.items()
                                        if not k.startswith("_")
                                        and (v != DEFAULTS.get(k) or k == "name")},
           "score": rep["score"], "per_task": rep["per_task"],
           "per_class": rep["per_class"], "secs": round(time.time() - t0)}
    with open(out_jsonl, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pred.insert(0, "id", df["id"].values)
    pred.to_csv(f"{oof_dir}/{cfg['name']}.csv", index=False)
    np.savez(f"{oof_dir}/{cfg['name']}.npz", **oof_probs)
    print(f"[{cfg['name']}] CV={rep['score']:.4f} " +
          " ".join(f"{t}={rep['per_task'][t]:.3f}" for t in TASKS) +
          f" ({rec['secs']}s)", flush=True)
    return rec


# ---------------- 斷點續跑（VM 被回收後 replay 用） ----------------
def load_results(path="/content/results.jsonl"):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[r["name"]] = r
    return out


def get_or_run(name, cfg, **kw):
    """已有同名紀錄（含本機回傳的 seed）→ 直接用；否則跑。失敗回 score=-1。"""
    d = load_results(kw.get("out_jsonl", "/content/results.jsonl"))
    if name in d:
        print(f"SKIP {name} (cached {d[name]['score']:.4f})", flush=True)
        return d[name]
    try:
        return run_cv({"name": name, **cfg}, **kw)
    except Exception:
        import traceback
        traceback.print_exc()
        try:                       # 防 OOM 後 CUDA 記憶體洩漏到下個 config（PCGrad 曾連環 OOM）
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"{name}_FAILED", flush=True)
        return {"name": name, "score": -1.0, "per_task": {}}


def assemble_score(base_name, repl_name=None, repl_tasks=(),
                   train_path="/content/train.csv", oof_dir="/content/oof",
                   out_jsonl="/content/results.jsonl", record_as=None):
    """以 base 實驗的 OOF 機率為底，將 repl_tasks 換成另一實驗的機率後級聯計分。
    用於單任務專模的決策（專模自身的其他頭未訓練、原始分數無意義）。"""
    df = load_df(train_path)
    probs = {t: np.load(f"{oof_dir}/{base_name}.npz")[t] for t in TASKS}
    if repl_name:
        z = np.load(f"{oof_dir}/{repl_name}.npz")
        for t in repl_tasks:
            probs[t] = z[t]
    pred = cascade_decode(probs, len(df))
    rep = score_frames(df, pred)
    if record_as:
        rec = {"name": record_as, "score": rep["score"],
               "per_task": rep["per_task"], "assembled": True}
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{record_as or repl_name}] ASM CV={rep['score']:.4f} " +
          " ".join(f"{t}={rep['per_task'][t]:.3f}" for t in TASKS), flush=True)
    return rep


def run_full(user_cfg, seeds=(42,), train_path="/content/train.csv",
             eval_path="/content/vpesg4k_val_1000.csv",
             oof_dir="/content/oof", out_jsonl="/content/results.jsonl"):
    """oval gate 用：全 train 訓練（多 seed）、預測 eval、僅平均機率的級聯分數
    印出/落盤（個別 seed 分數不印不記 = gate 紀律）。每 seed eval 機率 npz 快取，
    VM 回收後續跑不重訓。"""
    cfg = {**DEFAULTS, **user_cfg}
    tr, te = load_df(train_path), load_df(eval_path)
    if cfg.get("aug_csv"):
        tr = pd.concat([tr, load_df(cfg["aug_csv"])], ignore_index=True)
    os.makedirs(oof_dir, exist_ok=True)
    acc = None
    for sd in seeds:
        fp = f"{oof_dir}/{cfg['name']}_s{sd}_full.npz"
        if os.path.exists(fp):
            probs = {t: np.load(fp)[t] for t in TASKS}
        else:
            probs = train_one(tr, te, {**cfg, "seed": sd})
            np.savez(fp, **probs)
            import shutil
            shutil.make_archive("/content/oof_backup", "gztar", oof_dir)
        print(f"[{cfg['name']}] seed{sd} full done", flush=True)
        acc = probs if acc is None else {t: acc[t] + probs[t] for t in TASKS}
    avg = {t: acc[t] / len(seeds) for t in TASKS}
    pred = cascade_decode(avg, len(te))
    rep = score_frames(te, pred)
    rec = {"name": cfg["name"] + "_oval", "score": rep["score"],
           "per_task": rep["per_task"], "per_class": rep["per_class"],
           "seeds": list(seeds), "full": True}
    with open(out_jsonl, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{rec['name']}] OVAL={rep['score']:.4f} " +
          " ".join(f"{t}={rep['per_task'][t]:.3f}" for t in TASKS), flush=True)
    return rec


def run_predict(user_cfg, seeds=(42, 142, 242), train_path="/content/train.csv",
                test_path="/content/vpesg4k_test_2000.csv",
                oof_dir="/content/oof"):
    """提交用：全 train 訓練（多 seed）、預測無標籤 test、存平均機率 npz 與級聯
    標籤 csv。每 seed npz 快取（{name}_s{sd}_test.npz），VM 回收續跑。"""
    cfg = {**DEFAULTS, **user_cfg}
    tr, te = load_df(train_path), load_df(test_path)
    os.makedirs(oof_dir, exist_ok=True)
    acc = None
    for sd in seeds:
        fp = f"{oof_dir}/{cfg['name']}_s{sd}_test.npz"
        if os.path.exists(fp):
            probs = {t: np.load(fp)[t] for t in TASKS}
        else:
            probs = train_one(tr, te, {**cfg, "seed": sd})
            np.savez(fp, **probs)
            import shutil
            shutil.make_archive("/content/oof_backup", "gztar", oof_dir)
        print(f"[{cfg['name']}] seed{sd} test done", flush=True)
        acc = probs if acc is None else {t: acc[t] + probs[t] for t in TASKS}
    avg = {t: acc[t] / len(seeds) for t in TASKS}
    np.savez(f"{oof_dir}/{cfg['name']}_avg_test.npz", **avg)
    pred = cascade_decode(avg, len(te))
    pred.insert(0, "id", te["id"].values)
    pred.to_csv(f"{oof_dir}/{cfg['name']}_test_pred.csv", index=False)
    import shutil
    shutil.make_archive("/content/oof_backup", "gztar", oof_dir)
    print(f"[{cfg['name']}] TEST_PRED saved ({len(te)} rows)", flush=True)
    return pred
