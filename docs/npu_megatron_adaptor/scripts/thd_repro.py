"""Reproduce verl thd packed forward vs simple forward to pinpoint the logprob bug.

A: simple [1,S] forward (known correct, ppl ~1.23)
B: thd forward, single sequence (preprocess_packed_seqs, 1 sample)
C: thd forward, two sequences packed (mimics GRPO multi-sample batch)

If B==A but C wrong -> cross-sequence alignment (position_ids/cu_seqlens).
If B also wrong -> thd attention / packed_seq_params itself.

Run: torchrun --standalone --nnodes=1 --nproc-per-node=1 thd_repro.py
"""
import os
import torch
import megatron_adaptor  # noqa: F401
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

MODEL = "/home/b84412626/models/Qwen3-0.6B"
PROMPT = "Question: What is 12 + 15? Answer:"
MAX_NEW = 32

dist.init_process_group(backend="hccl")
torch.npu.set_device(int(os.environ.get("LOCAL_RANK", 0)))
mpu.initialize_model_parallel(1, 1)
model_parallel_cuda_manual_seed(42)

# HF ground truth + response
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np

tok = AutoTokenizer.from_pretrained(MODEL)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids
gen = hf.generate(prompt_ids, max_new_tokens=MAX_NEW, do_sample=False)
full_ids = gen[0].tolist()
S = len(full_ids)
plen = prompt_ids.shape[1]
resp_ids = full_ids[plen:]
with torch.no_grad():
    hf_lp = torch.log_softmax(hf(torch.tensor([full_ids])).logits[0].float(), -1)
hf_resp = np.array([hf_lp[plen + i - 1, t].item() for i, t in enumerate(resp_ids)])
print(f"[HF] ppl={np.exp(-hf_resp.mean()):.3f}")

# megatron load
from mbridge import AutoBridge
bridge = AutoBridge.from_config(AutoConfig.from_pretrained(MODEL), dtype=torch.bfloat16)
bridge.config.bf16 = True
models = bridge.get_model(bf16=True, wrap_with_ddp=False)
bridge.load_weights(models, MODEL)
model = models[0]
model.eval()


def resp_logprobs_from_logits_SV(logits_SV):
    lp = torch.log_softmax(logits_SV.float().cpu(), -1)
    return np.array([lp[plen + i - 1, t].item() for i, t in enumerate(resp_ids)])


def ppl(a):
    return np.exp(-a.mean())


# ---------- A: simple [1,S] ----------
with torch.no_grad():
    out = model(input_ids=torch.tensor([full_ids], device="npu"),
                position_ids=torch.arange(S, device="npu").unsqueeze(0),
                attention_mask=None)
logits_A = out[0] if out.shape[0] == 1 else out[:, 0]
A = resp_logprobs_from_logits_SV(logits_A)
print(f"[A simple]      ppl={ppl(A):.3f}  corr_vs_HF={np.corrcoef(np.exp(A),np.exp(hf_resp))[0,1]:.4f}")

# ---------- B: thd single sequence ----------
from verl.models.mcore.util import preprocess_packed_seqs, postprocess_packed_seqs
ids = torch.tensor([full_ids], device="npu")
amask = torch.ones(1, S, dtype=torch.bool, device="npu")
rmpad, psp = preprocess_packed_seqs(ids, amask, pre_process=True)
print(f"[thd] rmpad shape={tuple(rmpad.shape)} cu_seqlens_q={psp.cu_seqlens_q.tolist()}")
# verl passes position_ids straight through (model_forward.py); replicate with padded [1,S]
pos_padded = torch.arange(S, device="npu").unsqueeze(0)
with torch.no_grad():
    out_b = model(input_ids=rmpad.unsqueeze(0) if rmpad.dim() == 1 else rmpad,
                  position_ids=pos_padded, attention_mask=None, packed_seq_params=psp)
ob = postprocess_packed_seqs(out_b, psp, amask, 1, S, post_process=True)  # [1,S,V]
B = resp_logprobs_from_logits_SV(ob[0])
print(f"[B thd 1-seq]   ppl={ppl(B):.3f}  corr_vs_HF={np.corrcoef(np.exp(B),np.exp(hf_resp))[0,1]:.4f}")

# ---------- C: thd two sequences packed (mimic GRPO batch) ----------
ids2 = torch.tensor([full_ids, full_ids], device="npu")
amask2 = torch.ones(2, S, dtype=torch.bool, device="npu")
rmpad2, psp2 = preprocess_packed_seqs(ids2, amask2, pre_process=True)
print(f"[thd-2] rmpad2 shape={tuple(rmpad2.shape)} cu_seqlens_q={psp2.cu_seqlens_q.tolist()}")
pos_padded2 = torch.arange(S, device="npu").unsqueeze(0).expand(2, S).contiguous()
with torch.no_grad():
    out_c = model(input_ids=rmpad2.unsqueeze(0) if rmpad2.dim() == 1 else rmpad2,
                  position_ids=pos_padded2, attention_mask=None, packed_seq_params=psp2)
oc = postprocess_packed_seqs(out_c, psp2, amask2, 2, S, post_process=True)  # [2,S,V]
C = resp_logprobs_from_logits_SV(oc[0])  # first sequence
print(f"[C thd 2-seq]   ppl={ppl(C):.3f}  corr_vs_HF={np.corrcoef(np.exp(C),np.exp(hf_resp))[0,1]:.4f}")

# ---------- B variants: position_ids handling in thd ----------
def thd_forward(pos):
    with torch.no_grad():
        o = model(input_ids=rmpad.unsqueeze(0) if rmpad.dim() == 1 else rmpad,
                  position_ids=pos, attention_mask=None, packed_seq_params=psp)
    ob_ = postprocess_packed_seqs(o, psp, amask, 1, S, post_process=True)
    return resp_logprobs_from_logits_SV(ob_[0])

for tag, pos in [("None", None),
                 ("packed_arange_1xS", torch.arange(S, device="npu").unsqueeze(0)),
                 ("packed_arange_flat", torch.arange(S, device="npu"))]:
    try:
        bv = thd_forward(pos)
        print(f"[B pos={tag:18s}] ppl={ppl(bv):.3f} corr_vs_HF={np.corrcoef(np.exp(bv),np.exp(hf_resp))[0,1]:.4f}")
    except Exception as e:
        print(f"[B pos={tag:18s}] ERROR {type(e).__name__}: {str(e)[:60]}")

# ---------- B': raw packed logits vs simple, alignment probe ----------
lt = (out_b[0] if out_b.dim() == 3 else out_b).float().cpu()   # thd packed logits [S,V] (1 seq, no pad)
ls = logits_A.float().cpu()                                     # simple logits [S,V]
am_t, am_s = lt.argmax(-1), ls.argmax(-1)
m0 = (am_t == am_s).float().mean().item()
m_tp1 = (am_t[:-1] == am_s[1:]).float().mean().item()   # thd[i]==simple[i+1] (thd shifted left)
m_tm1 = (am_t[1:] == am_s[:-1]).float().mean().item()   # thd[i]==simple[i-1] (thd shifted right)
dmax = (lt - ls).abs().max().item()
print(f"\n=== ALIGNMENT PROBE (thd packed vs simple, 1 seq) ===")
print(f"argmax match same-pos : {m0:.3f}")
print(f"argmax match thd[i]=simple[i+1] : {m_tp1:.3f}")
print(f"argmax match thd[i]=simple[i-1] : {m_tm1:.3f}")
print(f"max|logits diff| same-pos : {dmax:.3f}")

print("\n=== VERDICT ===")
print(f"A simple    ppl={ppl(A):.2f}")
print(f"B thd 1-seq ppl={ppl(B):.2f}")
print(f"C thd 2-seq ppl={ppl(C):.2f}   (GRPO actor ppl ~319)")
