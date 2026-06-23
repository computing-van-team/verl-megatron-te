# dense Linear mxfp8 vs bf16,权重缓存稳态,扫 M(=token数) 找拐点+加速比
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
def lin(name,K,N,M,fwdbwd):
    L=te.Linear(K,N,bias=False,params_dtype=dt).to(dev)
    x=torch.randn(M,K,device=dev,dtype=dt,requires_grad=fwdbwd)
    fl=2*M*K*N*(3 if fwdbwd else 1)
    def fb():
        if fwdbwd: L(x).sum().backward()
        else:
            with torch.no_grad(): L(x)
    def mx(first):
        if fwdbwd:
            with te.fp8_autocast(enabled=True,fp8_recipe=recipe): y=L(x,is_first_microbatch=first)
            y.sum().backward()
        else:
            with torch.no_grad(), te.fp8_autocast(enabled=True,fp8_recipe=recipe): L(x,is_first_microbatch=first)
    for _ in range(3): mx(True)
    b=bench(fb); m=bench(lambda: mx(False))
    tag=name+('(fwd+bwd)' if fwdbwd else '(fwd)')
    print('%-22s M=%6d | bf16 %8.3fms %6.1fTF | mxfp8 %8.3fms %6.1fTF | x%4.2f'%(tag,M,b,fl/b/1e9,m,fl/m/1e9,b/m))
print('==== dense Linear mxfp8 vs bf16, steady-state, sweep M (hidden=2048) ====')
print('--- forward ---')
for M in (256,512,1024,2048,4096,8192,16384,32768):
    lin('attn-proj 2048->2048',2048,2048,M,False)
print('--- fwd+bwd (training, FLOPs x3) ---')
for M in (256,512,1024,2048,4096,8192,16384,32768):
    lin('attn-proj 2048->2048',2048,2048,M,True)
