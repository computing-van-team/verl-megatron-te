from vllm import LLM, SamplingParams

llm = LLM(
    model="/home/b84412626/models/Qwen3-0.6B",
    max_model_len=2048,
    gpu_memory_utilization=0.6,
    enforce_eager=True,
    block_size=128,
)
prompts = ["1+1等于几?请直接回答。", "What is the capital of France?"]
outs = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0))
for o in outs:
    print("PROMPT:", o.prompt[:30], "-> OUTPUT:", o.outputs[0].text[:80].replace("\n", " "))
print("VLLM_ASCEND_SMOKE_OK")
