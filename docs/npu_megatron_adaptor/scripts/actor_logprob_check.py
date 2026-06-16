"""Verify actor-side logprob: megatron_adaptor forward vs HF fp32 ground truth.

GRPO showed training_ppl(actor)=319 vs rollout_ppl(vllm)=1.54 on vllm-generated
tokens -> the actor recompute is wrong (weights are correct per round-trip).
This isolates whether the megatron forward itself is numerically wrong (simple
[1,S] non-packed forward). If it matches HF, the bug is in verl's thd-packing
logprob path; if not, megatron_adaptor forward is wrong.

Run: torchrun --standalone --nnodes=1 --nproc-per-node=1 actor_logprob_check.py
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

# ---- HF (CPU fp32): generate response + ground-truth logprobs ----
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids
gen = hf.generate(prompt_ids, max_new_tokens=MAX_NEW, do_sample=False)
full_ids = gen[0].tolist()
plen = prompt_ids.shape[1]
resp_ids = full_ids[plen:]
print(f"[hf] resp_len={len(resp_ids)} text={tok.decode(resp_ids)[:60]!r}")

with torch.no_grad():
    hf_logits = hf(torch.tensor([full_ids])).logits[0].float()
hf_lp = torch.log_softmax(hf_logits, -1)
hf_resp_lp = [hf_lp[plen + i - 1, tid].item() for i, tid in enumerate(resp_ids)]

# ---- megatron (NPU bf16): same tokens, simple [1,S] forward ----
from mbridge import AutoBridge
from transformers import AutoConfig

hf_config = AutoConfig.from_pretrained(MODEL)
bridge = AutoBridge.from_config(hf_config, dtype=torch.bfloat16)
bridge.config.bf16 = True
models = bridge.get_model(bf16=True, wrap_with_ddp=False)
bridge.load_weights(models, MODEL)
model = models[0]
model.eval()

S = len(full_ids)
input_ids = torch.tensor([full_ids], device="npu")
position_ids = torch.arange(S, device="npu").unsqueeze(0)
with torch.no_grad():
    out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
print(f"[megatron] raw output shape={tuple(out.shape)}")
# normalize to [S, V]
if out.dim() == 3:
    logits = out[0] if out.shape[0] == 1 else out[:, 0]  # [b,s,v] or [s,b,v]
else:
    logits = out
logits = logits.float().cpu()
print(f"[megatron] logits[S,V] shape={tuple(logits.shape)}")
mg_lp = torch.log_softmax(logits, -1)
mg_resp_lp = [mg_lp[plen + i - 1, tid].item() for i, tid in enumerate(resp_ids)]

# ---- compare ----
h = np.array(hf_resp_lp)
m = np.array(mg_resp_lp)
corr = np.corrcoef(np.exp(h), np.exp(m))[0, 1]
print("\n===== ACTOR LOGPROB VERDICT =====")
print(f"tokens                : {len(h)}")
print(f"HF      mean logprob  : {h.mean():.4f}  ppl={np.exp(-h.mean()):.2f}")
print(f"megatron mean logprob : {m.mean():.4f}  ppl={np.exp(-m.mean()):.2f}  (GRPO actor ppl ~319)")
print(f"mean |dlogprob|       : {np.abs(h - m).mean():.4f}")
print(f"pearson corr (prob)   : {corr:.4f}")
print(f"first 6 HF  : {[round(x,2) for x in hf_resp_lp[:6]]}")
print(f"first 6 MG  : {[round(x,2) for x in mg_resp_lp[:6]]}")
