"""Isolation test: is the high rollout_probs_diff caused by the A5 ND cache patch?

Compares per-token logprobs of vLLM (NPU, patched) vs HF transformers (CPU fp32,
mathematical ground truth). Both load the SAME HF checkpoint directly, so the
verl megatron->vllm weight-resync path is excluded — this isolates the numerical
correctness of the patch + vllm-on-NPU inference itself.

Verdict:
  high pearson corr (>0.99) + small |dprob| (<0.05)  -> ND patch is numerically safe;
                                                         GRPO diff is from weight-sync/config.
  low corr / large |dprob|                            -> patch or vllm-NPU inference is wrong.
"""
import torch
import numpy as np

MODEL = "/home/b84412626/models/Qwen3-0.6B"
PROMPTS = [
    "Question: What is 12 + 15? Answer:",
    "The capital of France is",
    "Question: A shop sells apples at 3 dollars each. How much for 4 apples? Answer:",
]
MAX_NEW = 48

# ---- 1. vLLM (NPU, patched) generate greedily + record chosen-token logprobs ----
from vllm import LLM, SamplingParams

import os
CHUNKED = os.environ.get("TEST_CHUNKED", "0") == "1"
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.5,
          max_model_len=1024, block_size=128,
          enable_chunked_prefill=CHUNKED, enable_prefix_caching=CHUNKED)
print(f"[config] enable_chunked_prefill={CHUNKED} enable_prefix_caching={CHUNKED}")
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, logprobs=0)
outs = llm.generate(PROMPTS, sp)

samples = []
for o in outs:
    comp = o.outputs[0]
    ids = list(comp.token_ids)
    # logprobs: list[dict{token_id: Logprob}], one per generated token
    lps = []
    for tid, lp_dict in zip(ids, comp.logprobs):
        lps.append(lp_dict[tid].logprob)
    samples.append({
        "prompt_ids": list(o.prompt_token_ids),
        "resp_ids": ids,
        "vllm_logprobs": lps,
    })
    print(f"[vllm] prompt={o.prompt[:30]!r} resp_len={len(ids)} text={comp.text[:50]!r}")

del llm
import gc; gc.collect()

# ---- 2. HF transformers (CPU fp32) teacher-forcing on the SAME token ids ----
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

all_vllm, all_hf = [], []
for s in samples:
    full = torch.tensor([s["prompt_ids"] + s["resp_ids"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(full).logits[0].float()  # [seq, vocab]
    logprobs = torch.log_softmax(logits, dim=-1)
    plen = len(s["prompt_ids"])
    hf_lps = []
    for i, tid in enumerate(s["resp_ids"]):
        # logits at position (plen+i-1) predict token at (plen+i)
        pos = plen + i - 1
        hf_lps.append(logprobs[pos, tid].item())
    all_vllm.extend(s["vllm_logprobs"])
    all_hf.extend(hf_lps)

v = np.array(all_vllm)
h = np.array(all_hf)
vp, hp = np.exp(v), np.exp(h)
corr_logprob = np.corrcoef(v, h)[0, 1]
corr_prob = np.corrcoef(vp, hp)[0, 1]
print("\n===== VERDICT =====")
print(f"tokens compared       : {len(v)}")
print(f"mean |dlogprob|       : {np.abs(v - h).mean():.4f}")
print(f"mean |dprob|          : {np.abs(vp - hp).mean():.4f}   (GRPO showed 0.55)")
print(f"pearson corr (logprob): {corr_logprob:.4f}")
print(f"pearson corr (prob)   : {corr_prob:.4f}   (GRPO showed 0.21)")
print(f"max  |dprob|          : {np.abs(vp - hp).max():.4f}")
