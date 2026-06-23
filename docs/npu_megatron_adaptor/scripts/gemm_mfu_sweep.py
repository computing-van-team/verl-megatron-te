# MFU vs M(token数) sweep: 固定 dense GEMM (K,N),扫 M,测实测 TFLOPs
import time, torch, torch_npu  # noqa
import megatron_adaptor  # noqa
import transformer_engine.pytorch as te
dev='npu'; dt=torch.bfloat16
def bench(fn,w=10,it=30):
    for _ in range(w): fn()
    torch.npu.synchronize(); t0=time.time()
    for _ in range(it): fn()
    torch.npu.synchronize(); return (time.time()-t0)/it
for (K,N,tag) in [(2048,2048,'attn-proj 2048x2048'),(2048,6144,'mlp up 2048x6144')]:
    print('==== GEMM %s (bf16 fwd) ===='%tag)
    print('%9s %10s %10s'%('M(tokens)','time_ms','TFLOPs'))
    res=[]
    for M in [64,128,256,512,1024,2048,4096,8192,16384,32768,65536]:
        lin=te.Linear(K,N,bias=False,params_dtype=dt).to(dev)
        x=torch.randn(M,K,device=dev,dtype=dt)
        def f():
            with torch.no_grad(): lin(x)
        t=bench(f); tf=2*M*K*N/t/1e12
        res.append((M,t*1000,tf)); print('%9d %10.3f %10.1f'%(M,t*1000,tf))
    peak=max(r[2] for r in res)
    print('--- util = 实测/达到的峰值(%.0f TFLOPs) ---'%peak)
    for M,ms,tf in res: print('M=%7d  TFLOPs=%7.1f  util=%5.1f%%'%(M,tf,100*tf/peak))
    print()
