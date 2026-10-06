"""Packed W248/A248 with K=64 activation scales and integer Tensor Cores.

W2/W4 x A2/A4 use native S4 MMA, sign-extending S2 operands in registers.
Pairs containing an 8-bit operand use S8 MMA. Weights retain the calibrated
per-output-row quantizer; activations store their declared number of bits.
Scale metadata is separate from the existing logical W/A code-bit budget.
"""
from pathlib import Path
from types import MethodType
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .quantizers import weight_codes
from actune.precision import active_precision_profile


def _inference_mlp(self, x):
    if torch.is_grad_enabled():
        return self._group64_original_forward(x)
    # The gate activation is a private temporary. Reuse it for the product
    # instead of keeping three full intermediate-width tensors live together.
    gate=self.act_fn(self.gate_proj(x))
    gate.mul_(self.up_proj(x))
    return self.down_proj(gate)


def reuse_mlp_temporary(model):
    from transformers.models.gemma.modeling_gemma import GemmaMLP
    count=0
    for module in model.modules():
        if type(module) is GemmaMLP and not hasattr(module,'_group64_original_forward'):
            module._group64_original_forward=module.forward
            module.forward=MethodType(_inference_mlp,module)
            count+=1
    return count


@triton.jit
def _quantize_group(x, packed, scales, K:tl.constexpr, A:tl.constexpr, GROUP:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1)
    i=tl.arange(0,GROUP);value=tl.load(x+row*K+group*GROUP+i).to(tl.float32)
    positive:tl.constexpr=(1<<(A-1))-1
    scale=tl.maximum(tl.max(tl.abs(value)),1e-8)/positive
    tl.store(scales+row*(K//GROUP)+group,scale)
    code=libdevice.rint(libdevice.div_rn(value,scale))
    code=tl.minimum(tl.maximum(code,-positive-1),positive).to(tl.int32)
    P:tl.constexpr=8//A
    fields=tl.reshape(code & ((1<<A)-1),(GROUP//P,P))
    bits=fields << (tl.arange(0,P)[None,:]*A)
    byte=tl.sum(bits,1).to(tl.uint8)
    tl.store(packed+row*(K//P)+group*(GROUP//P)+tl.arange(0,GROUP//P),byte)


@triton.jit
def _quantize_groups(x, packed, scales, ELEMENTS:tl.constexpr, A:tl.constexpr,
                     GROUP:tl.constexpr=64, GROUPS:tl.constexpr=16):
    # Several independent groups per CTA avoid one tiny block per 64 values.
    groups=tl.program_id(0)*GROUPS+tl.arange(0,GROUPS)
    i=tl.arange(0,GROUP)
    value=tl.load(x+groups[:,None]*GROUP+i[None,:],groups[:,None]<ELEMENTS//GROUP,0).to(tl.float32)
    positive:tl.constexpr=(1<<(A-1))-1
    scale=tl.maximum(tl.max(tl.abs(value),1),1e-8)/positive
    tl.store(scales+groups,scale,groups<ELEMENTS//GROUP)
    code=libdevice.rint(libdevice.div_rn(value,scale[:,None]))
    code=tl.minimum(tl.maximum(code,-positive-1),positive).to(tl.int32)
    P:tl.constexpr=8//A
    fields=tl.reshape(code&((1<<A)-1),(GROUPS,GROUP//P,P))
    byte=tl.sum(fields << (tl.arange(0,P)[None,None,:]*A),2).to(tl.uint8)
    tl.store(packed+groups[:,None]*(GROUP//P)+tl.arange(0,GROUP//P)[None,:],byte,
             groups[:,None]<ELEMENTS//GROUP)


@triton.jit
def _group_gemm(ap,wp,asc,wsc,bias,out,M,N:tl.constexpr,K:tl.constexpr,
                W:tl.constexpr,A:tl.constexpr,BIAS:tl.constexpr,
                BM:tl.constexpr=16,BN:tl.constexpr=32,GROUP:tl.constexpr=64):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,GROUP);acc=tl.zeros((BM,BN),tl.float32)
    for group in range(K//GROUP):
        ki=group*GROUP+k
        ab=tl.load(ap+m[:,None]*(K*A//8)+(ki[None,:]*A//8),m[:,None]<M,0).to(tl.int32)
        av=(ab >> ((ki[None,:]%(8//A))*A)) & ((1<<A)-1)
        av=((av^(1<<(A-1)))-(1<<(A-1))).to(tl.int8)
        wb=tl.load(wp+n[None,:]*(K*W//8)+(ki[:,None]*W//8),n[None,:]<N,0).to(tl.int32)
        wv=(wb >> ((ki[:,None]%(8//W))*W)) & ((1<<W)-1)
        wv=((wv^(1<<(W-1)))-(1<<(W-1))).to(tl.int8)
        dots=tl.dot(av,wv,out_dtype=tl.int32)
        scale=tl.load(asc+m*(K//GROUP)+group,m<M,0)
        acc+=dots.to(tl.float32)*scale[:,None]
    result=acc*tl.load(wsc+n,n<N,0)[None,:]
    if BIAS:result+=tl.load(bias+n,n<N,0)[None,:]
    tl.store(out+m[:,None]*N+n[None,:],result,(m[:,None]<M)&(n[None,:]<N))


_EXTENSION=None
def extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        _EXTENSION=load(name='pi05_grouped_s4_sm80_v2',
            sources=[str(Path(__file__).with_name('grouped_native.cu'))],
            extra_cuda_cflags=['-O3'],extra_cflags=['-O3'],verbose=False)
    return _EXTENSION


def pack_rows(q,bits):
    fields=q.to(torch.int16).reshape(q.shape[0],-1,8//bits)&((1<<bits)-1)
    shifts=torch.arange(8//bits,device=q.device)*bits
    return (fields<<shifts).sum(-1).to(torch.uint8).contiguous()


class GroupedLinear(torch.nn.Module):
    def __init__(self,linear,w,a,ratios):
        super().__init__();self.in_features=linear.in_features;self.out_features=linear.out_features
        self.w,self.a=w,a
        if self.in_features%64:raise ValueError('Grouped backend requires K divisible by 64')
        if w not in (2,4,8) or a not in (2,4,8):raise ValueError('Unsupported integer width')
        q,s=weight_codes(linear.weight,w,a,clip_ratios=ratios)
        self.register_buffer('packed_weight',pack_rows(q,w))
        self.register_buffer('weight_scale',s.contiguous())
        b=None if linear.bias is None else linear.bias.detach().clone()
        if b is not None and a==4:b=b.float()
        self.register_buffer('bias',b)

    def forward(self,x):
        if not x.is_cuda:raise ValueError('Grouped integer backend requires CUDA')
        flat=x.reshape(-1,self.in_features).contiguous();m,k=flat.shape
        packed=torch.empty((m,k*self.a//8),device=x.device,dtype=torch.uint8)
        scales=torch.empty((m,k//64),device=x.device,dtype=torch.float32)
        _quantize_groups[(triton.cdiv(m*k//64,16),)](
            flat,packed,scales,ELEMENTS=m*k,A=self.a,num_warps=4)
        if max(self.w,self.a)<=4:
            b=torch.empty(0,device=x.device,dtype=torch.float32) if self.bias is None else self.bias.float()
            out=extension().linear(packed,scales,self.packed_weight,self.weight_scale,b,flat,self.w,self.a)
        else:
            out=torch.empty((m,self.out_features),device=x.device,dtype=x.dtype)
            _group_gemm[(triton.cdiv(m,16),triton.cdiv(self.out_features,32))](
                packed,self.packed_weight,scales,self.weight_scale,
                scales if self.bias is None else self.bias,out,m,N=self.out_features,K=k,
                W=self.w,A=self.a,BIAS=self.bias is not None,num_warps=4,num_stages=2)
        return out.reshape(*x.shape[:-1],self.out_features)


class GroupedPaddedLinear(torch.nn.Module):
    def __init__(self,linear,assignments,initial,clips):
        super().__init__();logical=linear.in_features;padded=(logical+127)//128*128
        if padded!=logical:
            fc=torch.nn.Linear(padded,linear.out_features,bias=linear.bias is not None,
                               device=linear.weight.device,dtype=linear.weight.dtype)
            with torch.no_grad():
                fc.weight.zero_();fc.weight[:,:logical].copy_(linear.weight)
                if fc.bias is not None:fc.bias.copy_(linear.bias)
            linear=fc
        self.in_features=logical;self.out_features=linear.out_features;self.padding=padded-logical
        self.assignments=dict(assignments);self.default_profile=initial
        self.backends=torch.nn.ModuleDict({mode:GroupedLinear(linear,int(mode[1]),int(mode[3]),clips[mode])
                                           for mode in sorted(set(assignments.values()))})
        self.register_buffer('_weight_dtype',torch.empty(0,device=linear.weight.device,dtype=linear.weight.dtype))
        self.calls=self.input_elements=0
    @property
    def weight(self):return self._weight_dtype
    def forward(self,x):
        self.calls+=1;self.input_elements+=x.numel()
        if self.padding:x=torch.nn.functional.pad(x,(0,self.padding))
        return self.backends[self.assignments[active_precision_profile() or self.default_profile]](x)
