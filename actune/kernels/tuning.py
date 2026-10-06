"""Bit-exact shape tuning of the native grouped integer kernels."""
from pathlib import Path
import statistics
import torch
import triton
TUNINGS={}
_EXTENSION=None

def extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        _EXTENSION=load(name='acttune_grouped_geometry_m_v1',sources=[str(Path(__file__).with_name('grouped_big_m.cu'))],extra_cuda_cflags=['-O3'],extra_cflags=['-O3'],verbose=False)
    return _EXTENSION

def select(key, launch, configurations, default):
    if key in TUNINGS: return TUNINGS[key]['selected']
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Kernel geometry must be warmed before graph capture')
    expected=launch(default); torch.cuda.synchronize()
    results=[]
    for config in configurations:
        result=launch(config); torch.cuda.synchronize()
        if not torch.equal(expected,result):
            results.append(dict(config=config,exact=False)); continue
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): launch(config)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10): launch(config)
        times=[]
        for _ in range(5):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graph.replay();end.record();end.synchronize();times.append(start.elapsed_time(end)/10)
        results.append(dict(config=config,exact=True,ms=statistics.median(times)))
        del graph
    valid=[r for r in results if r['exact']]
    baseline=next(r for r in valid if r['config']==default)
    best=min(valid,key=lambda r:r['ms'])
    selected=best['config'] if best['ms'] < baseline['ms']*.95 else default
    TUNINGS[key]=dict(selected=selected,candidates=results)
    print('geometry',key,'selected',selected,'baseline_ms',baseline['ms'],'best_ms',best['ms'],flush=True)
    return selected

def grouped_forward(self,x):
    from .grouped import _quantize_groups,_group_gemm
    flat=x.reshape(-1,self.in_features).contiguous();m,k=flat.shape;n=self.out_features
    packed=torch.empty((m,k*self.a//8),device=x.device,dtype=torch.uint8)
    scales=torch.empty((m,k//64),device=x.device,dtype=torch.float32)
    _quantize_groups[(triton.cdiv(m*k//64,16),)](flat,packed,scales,ELEMENTS=m*k,A=self.a,num_warps=4)
    if max(self.w,self.a)<=4:
        if x.dtype!=torch.bfloat16:return self._round2_original(x)
        ext=extension();b=torch.empty(0,device=x.device,dtype=torch.float32) if self.bias is None else self.bias.float()
        launch=lambda c:ext.linear(packed,scales,self.packed_weight,self.weight_scale,b,flat,self.w,self.a,c)
        configs=list(range(13));default=0
    else:
        def launch(c):
            bm,bn,warps,stages=c
            out=torch.empty((m,n),device=x.device,dtype=x.dtype)
            _group_gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](packed,self.packed_weight,scales,self.weight_scale,
                scales if self.bias is None else self.bias,out,m,N=n,K=k,W=self.w,A=self.a,BIAS=self.bias is not None,
                BM=bm,BN=bn,num_warps=warps,num_stages=stages)
            return out
        default=(16,32,4,2)
        configs=[default,(16,64,4,2),(16,128,4,2),(32,64,4,2),(32,128,4,2),(32,64,4,3)]
    key=('grouped',self.w,self.a,m,k,n,str(x.dtype),self.bias is not None)
    config=select(key,launch,configs,default)
    return launch(config).reshape(*x.shape[:-1],n)
