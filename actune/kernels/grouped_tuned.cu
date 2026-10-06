#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cstdint>

__device__ __forceinline__ uint32_t expand2(uint16_t raw) {
  uint32_t result=0;
  #pragma unroll
  for(int i=0;i<8;++i){uint32_t v=(raw>>(i*2))&3;result|=(((v^2)-2)&15)<<(i*4);}
  return result;
}
__device__ __forceinline__ uint32_t read8(const uint8_t* p,int bits,int offset) {
  return bits==4?*reinterpret_cast<const uint32_t*>(p+offset/2)
                :expand2(*reinterpret_cast<const uint16_t*>(p+offset/4));
}

// Four warps reuse a 16x256 activation tile and stage 64 output rows with
// cp.async. Group dequantization remains after each exact K=64 integer dot.
template<int W,int A,int TN,int NW,int SK>
__device__ __forceinline__ void stage(uint8_t* dst,const uint8_t* a,const uint8_t* w,
                                    int tm,int tn,int k,int M,int N,int K) {
  constexpr int ak=SK*A/8,wk=SK*W/8,ac=16*ak/16,wc=TN*wk/16;
  for(int copy=threadIdx.x;copy<ac+wc;copy+=NW*32){
    bool activation=copy<ac;
    int item=activation?copy:copy-ac;
    int stride=activation?ak:wk;
    int bits=activation?A:W;
    int row=item/(stride/16),chunk=item%(stride/16);
    bool valid=(activation?tm+row<M:tn+row<N)&&k+chunk*128/bits<K;
    const uint8_t* source=activation?a+(tm+row)*(K*A/8)+k*A/8+chunk*16
                                    :w+(tn+row)*(K*W/8)+k*W/8+chunk*16;
    uint8_t* dest=dst+(activation?0:16*ak)+row*stride+chunk*16;
    uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dest));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"(address),"l"(valid?source:a),"r"(valid?16:0));
  }
  asm volatile("cp.async.commit_group;\n"::);
}

template<typename T,int W,int A,int TN,int NW,int SK>
__global__ void grouped(const uint8_t* a,const float* as,const uint8_t* w,
                        const float* ws,const float* bias,T* out,
                        int M,int N,int K,bool has_bias) {
  extern __shared__ __align__(16) uint8_t tiles[];
  constexpr int ak=SK*A/8,wk=SK*W/8,bytes=16*ak+TN*wk;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32,g=lane/4,t=lane%4;
  const int tm=blockIdx.y*16,tn=blockIdx.x*TN+warp*(TN/NW);
  const int r0=tm+g,r1=r0+8;
  float f[TN/NW/8][4]={};
  stage<W,A,TN,NW,SK>(tiles,a,w,tm,blockIdx.x*TN,0,M,N,K);
  for(int start=0;start<K;start+=SK){
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    uint8_t* current=tiles+((start/SK)&1)*bytes;
    if(start+SK<K)stage<W,A,TN,NW,SK>(tiles+(((start/SK)+1)&1)*bytes,a,w,tm,blockIdx.x*TN,start+SK,M,N,K);
    #pragma unroll
    for(int inner=0;inner<SK;inner+=64){
    if(start+inner>=K)break;
    const uint8_t* p0=current+g*ak+inner*A/8;
    const uint8_t* p1=current+(g+8)*ak+inner*A/8;
    uint32_t av[4]={read8(p0,A,t*8),read8(p1,A,t*8),read8(p0,A,32+t*8),read8(p1,A,32+t*8)};
    float s0=r0<M?as[r0*(K/64)+(start+inner)/64]:0;
    float s1=r1<M?as[r1*(K/64)+(start+inner)/64]:0;
    #pragma unroll
    for(int sub=0;sub<TN/NW/8;++sub){
    const uint8_t* pw=current+16*ak+(warp*(TN/NW)+sub*8+g)*wk+inner*W/8;
    uint32_t wv[2]={read8(pw,W,t*8),read8(pw,W,32+t*8)};
    int d0=0,d1=0,d2=0,d3=0;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
       : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
       : "r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3]),"r"(wv[0]),"r"(wv[1]));
    f[sub][0]+=d0*s0;f[sub][1]+=d1*s0;f[sub][2]+=d2*s1;f[sub][3]+=d3*s1;
    }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int sub=0;sub<TN/NW/8;++sub){
  #pragma unroll
  for(int i=0;i<4;++i){
    int r=tm+g+(i>=2?8:0),c=tn+sub*8+t*2+(i&1);
    if(r<M&&c<N)out[r*N+c]=static_cast<T>(f[sub][i]*ws[c]+(has_bias?bias[c]:0));
  }
  }
}

torch::Tensor linear(torch::Tensor a,torch::Tensor as,torch::Tensor w,torch::Tensor ws,
                     torch::Tensor bias,torch::Tensor source,int64_t W,int64_t A,int64_t config){
  TORCH_CHECK(a.is_cuda()&&as.is_cuda()&&w.is_cuda()&&ws.is_cuda()&&bias.is_cuda()&&source.is_cuda(),"CUDA required");
  c10::cuda::CUDAGuard guard(source.device());
  TORCH_CHECK(a.device()==source.device()&&as.device()==source.device()&&w.device()==source.device()&&ws.device()==source.device()&&bias.device()==source.device(),"device mismatch");
  TORCH_CHECK((W==2||W==4)&&(A==2||A==4),"S2/S4 operands required");
  TORCH_CHECK(a.scalar_type()==at::kByte&&w.scalar_type()==at::kByte,"packed uint8 required");
  TORCH_CHECK(as.scalar_type()==at::kFloat&&ws.scalar_type()==at::kFloat&&bias.scalar_type()==at::kFloat,"float scales/bias required");
  TORCH_CHECK(a.is_contiguous()&&as.is_contiguous()&&w.is_contiguous()&&ws.is_contiguous()&&bias.is_contiguous(),"contiguous operands required");
  TORCH_CHECK(a.dim()==2&&w.dim()==2&&source.dim()==2&&as.dim()==2,"rank mismatch");
  int M=source.size(0),K=source.size(1),N=w.size(0);
  TORCH_CHECK(M>0&&N>0&&K>0&&K%64==0,"invalid dimensions");
  TORCH_CHECK(a.size(0)==M&&a.size(1)==K*A/8&&w.size(1)==K*W/8,"packed shape mismatch");
  TORCH_CHECK(as.size(0)==M&&as.size(1)==K/64&&ws.numel()==N,"scale shape mismatch");
  TORCH_CHECK(bias.numel()==0||bias.numel()==N,"bias shape mismatch");
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major>=8,"Ampere or newer required");
  auto out=torch::empty({M,N},source.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  #define LAUNCH(WB,AB,TN,NW,SK) grouped<scalar_t,WB,AB,TN,NW,SK><<<dim3((N+TN-1)/TN,(M+15)/16),NW*32,2*(16*SK*AB/8+TN*SK*WB/8),stream>>>(a.data_ptr<uint8_t>(),as.data_ptr<float>(),w.data_ptr<uint8_t>(),ws.data_ptr<float>(),bias.numel()?bias.data_ptr<float>():nullptr,out.data_ptr<scalar_t>(),M,N,K,bias.numel()!=0)
  #define MODES(TN,NW,SK) if(W==4&&A==4){LAUNCH(4,4,TN,NW,SK);}else if(W==2&&A==4){LAUNCH(2,4,TN,NW,SK);}else if(W==4&&A==2){LAUNCH(4,2,TN,NW,SK);}else{LAUNCH(2,2,TN,NW,SK);}
  TORCH_CHECK(out.scalar_type()==at::kBFloat16,"BF16 output required");
  using scalar_t=at::BFloat16;
  {
    switch(config){
      case 0: {MODES(64,4,256);break;}
      case 1: {MODES(32,4,256);break;}
      case 2: {MODES(128,4,256);break;}
      case 3: {MODES(64,4,128);break;}
      case 4: {MODES(128,4,128);break;}
      case 5: {MODES(64,8,256);break;}
      case 6: {MODES(128,8,256);break;}
      default: TORCH_CHECK(false,"invalid configuration");
    }
  }
  #undef MODES
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("linear",&linear);}
