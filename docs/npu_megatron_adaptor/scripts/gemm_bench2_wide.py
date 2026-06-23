# MoE 专家 GroupedLinear mxfp8 vs bf16,权重缓存稳态,宽扫 tok/expert 找拐点
import time, torch, torch_npu  # noqa
import megatron_adaptor  # noqa
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
dev='npu'; dt=torch.bfloat16; recipe=MXFP8BlockScaling(); torch.manual_seed(0)
def bench(fn,w=8,it=25):
    for _ in range(w): fn()
    torch.npu.synchronize(); t0=time.time()
    for _ in range(it): fn()
    torch.npu.synchronize(); return (time.time()-t0)/it*1000.0
def grouped(name,K,N,ng,tok,fwdbwd):
    gl=te.GroupedLinear(ng,K,N,bias=False,params_dtype=dt,device=dev)
    M=ng*tok; x=torch.randn(M,K,device=dev,dtype=dt,requires_grad=fwdbwd)
    ms=[tok]*ng; fl=2*M*K*N*(3 if fwdbwd else 1)
    def fb():
        if fwdbwd: gl(x,ms).sum().backward()
        else:
            with torch.no_grad(): gl(x,ms)
    def mx(first):
        if fwdbwd:
            with te.fp8_autocast(enabled=True,fp8_recipe=recipe): y=gl(x,ms,is_first_microbatch=first)
            y.sum().backward()
        else:
            with torch.no_grad(), te.fp8_autocast(enabled=True,fp8_recipe=recipe): gl(x,ms,is_first_microbatch=first)
    for _ in range(3): mx(True)
    b=bench(fb); m=bench(lambda: mx(False))
    tag=name+('(fwd+bwd)' if fwdbwd else '(fwd)')
    print('%-20s tok/exp=%5d M=%7d | bf16 %8.3fms %6.1fTF | mxfp8 %8.3fms %6.1fTF | x%4.2f'%(tag,tok,M,b,fl/b/1e9,m,fl/m/1e9,b/m))
print('==== MoE expert GroupedLinear (128 experts), steady-state, sweep tok/expert ====')
print('--- forward only ---')
for tok in (256,512,1024,2048,4096,8192):
    grouped('fc1 2048->1536',2048,1536,128,tok,False)
print('--- fwd+bwd (training, FLOPs x3) ---')
for tok in (256,512,1024,2048,4096,8192):
    grouped('fc1 2048->1536',2048,1536,128,tok,True)
