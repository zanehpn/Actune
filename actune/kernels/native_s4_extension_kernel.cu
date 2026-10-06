#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t expand_signed_s2x8_to_s4x8(uint16_t raw) {
  uint32_t result = 0u;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    const uint32_t code = (raw >> (2 * index)) & 3u;
    const uint32_t nibble = static_cast<uint32_t>(
        static_cast<int32_t>((code ^ 2u) - 2u)) & 15u;
    result |= nibble << (4 * index);
  }
  return result;
}

__device__ __forceinline__ void copy_async_16(
    void* shared_destination,
    const void* global_source,
    int source_bytes) {
  const uint32_t shared_address = static_cast<uint32_t>(
      __cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :
      : "r"(shared_address), "l"(global_source), "r"(source_bytes));
}

__device__ __forceinline__ void commit_async_copies() {
  asm volatile("cp.async.commit_group;\n" : :);
}

__device__ __forceinline__ void wait_for_async_copies() {
  asm volatile("cp.async.wait_group 0;\n" : :);
}

__device__ __forceinline__ void issue_s4_stage(
    uint8_t* shared_stage,
    const uint8_t* activation,
    const uint8_t* weight,
    int thread,
    int tile_m,
    int tile_n,
    int start_k,
    int rows,
    int out_features,
    int in_features,
    int activation_stride,
    int weight_stride) {
  // One K=256 stage contains 128 A copies followed by 512 B copies. Having
  // every CTA thread issue five aligned 16-byte transfers keeps the copy work
  // balanced while amortizing one barrier over eight S4 MMA instructions.
  constexpr int packedK = 128;
  constexpr int activationCopies = 16 * packedK / 16;
  constexpr int weightCopies = 64 * packedK / 16;
#pragma unroll
  for (int copy = thread; copy < activationCopies + weightCopies; copy += 128) {
    if (copy < activationCopies) {
      const int row = copy >> 3;
      const int chunk = copy & 7;
      const bool valid =
          tile_m + row < rows && start_k + chunk * 32 < in_features;
      const uint8_t* source = valid
          ? activation + (tile_m + row) * activation_stride + (start_k >> 1)
              + chunk * 16
          : activation;
      copy_async_16(
          shared_stage + row * packedK + chunk * 16,
          source,
          valid ? 16 : 0);
    } else {
      const int weight_copy = copy - activationCopies;
      const int output = weight_copy >> 3;
      const int chunk = weight_copy & 7;
      const bool valid =
          tile_n + output < out_features
          && start_k + chunk * 32 < in_features;
      const uint8_t* source = valid
          ? weight + (tile_n + output) * weight_stride + (start_k >> 1)
              + chunk * 16
          : weight;
      copy_async_16(
          shared_stage + 16 * packedK + output * packedK + chunk * 16,
          source,
          valid ? 16 : 0);
    }
  }
  commit_async_copies();
}

__device__ __forceinline__ void issue_s2_stage(
    uint8_t* shared_stage,
    const uint8_t* activation,
    const uint8_t* weight,
    int thread,
    int tile_m,
    int tile_n,
    int start_k,
    int rows,
    int out_features,
    int in_features,
    int activation_stride,
    int weight_stride) {
  // A uses 128 packed bytes per row; genuine W2 storage uses only 64 bytes
  // per output row for a K=256 stage. Expansion to signed S4 fragments stays
  // in registers immediately before MMA, so no persistent 4-bit copy exists.
  constexpr int packedActivationK = 128;
  constexpr int packedWeightK = 64;
  constexpr int activationCopies = 16 * packedActivationK / 16;
  constexpr int weightCopies = 64 * packedWeightK / 16;
#pragma unroll
  for (int copy = thread; copy < activationCopies + weightCopies; copy += 128) {
    if (copy < activationCopies) {
      const int row = copy >> 3;
      const int chunk = copy & 7;
      const bool valid =
          tile_m + row < rows && start_k + chunk * 32 < in_features;
      const uint8_t* source = valid
          ? activation + (tile_m + row) * activation_stride + (start_k >> 1)
              + chunk * 16
          : activation;
      copy_async_16(
          shared_stage + row * packedActivationK + chunk * 16,
          source,
          valid ? 16 : 0);
    } else {
      const int weight_copy = copy - activationCopies;
      const int output = weight_copy >> 2;
      const int chunk = weight_copy & 3;
      const bool valid =
          tile_n + output < out_features
          && start_k + chunk * 64 < in_features;
      const uint8_t* source = valid
          ? weight + (tile_n + output) * weight_stride + (start_k >> 2)
              + chunk * 16
          : weight;
      copy_async_16(
          shared_stage + 16 * packedActivationK
              + output * packedWeightK + chunk * 16,
          source,
          valid ? 16 : 0);
    }
  }
  commit_async_copies();
}

template <typename output_t>
__global__ void native_s4s4_linear_kernel(
    const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    output_t* __restrict__ output,
    int rows,
    int out_features,
    int in_features,
    int weight_bits,
    bool has_bias) {
  // Exactly one warp computes one 16x8xK output tile.  This reference kernel
  // prioritizes an auditable native S4 MMA path; tile/autotuning work is gated
  // separately before any latency claim.
  const int lane = threadIdx.x;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 8;
  const int activation_stride = in_features >> 1;
  const int weight_stride = (in_features * weight_bits) >> 3;
  int d0 = 0;
  int d1 = 0;
  int d2 = 0;
  int d3 = 0;

  for (int start_k = 0; start_k < in_features; start_k += 64) {
    uint32_t a[4] = {0u, 0u, 0u, 0u};
    const int row0 = tile_m + group;
    const int row1 = row0 + 8;
    if (row0 < rows) {
      const uint8_t* row =
          activation + row0 * activation_stride + (start_k >> 1);
      a[0] = *reinterpret_cast<const uint32_t*>(row + thread_in_group * 4);
      a[2] = *reinterpret_cast<const uint32_t*>(row + 16 + thread_in_group * 4);
    }
    if (row1 < rows) {
      const uint8_t* row =
          activation + row1 * activation_stride + (start_k >> 1);
      a[1] = *reinterpret_cast<const uint32_t*>(row + thread_in_group * 4);
      a[3] = *reinterpret_cast<const uint32_t*>(row + 16 + thread_in_group * 4);
    }
    uint32_t b[2] = {0u, 0u};
    const int col = tile_n + group;
    if (col < out_features) {
      const uint8_t* row =
          weight + col * weight_stride + (start_k * weight_bits >> 3);
      if (weight_bits == 2) {
        const uint16_t raw0 = *reinterpret_cast<const uint16_t*>(
            row + thread_in_group * 2);
        const uint16_t raw1 = *reinterpret_cast<const uint16_t*>(
            row + 8 + thread_in_group * 2);
        b[0] = expand_signed_s2x8_to_s4x8(raw0);
        b[1] = expand_signed_s2x8_to_s4x8(raw1);
      } else {
        b[0] = *reinterpret_cast<const uint32_t*>(row + thread_in_group * 4);
        b[1] = *reinterpret_cast<const uint32_t*>(row + 16 + thread_in_group * 4);
      }
    }
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
  }

  const int accumulators[4] = {d0, d1, d2, d3};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = tile_m + group + (i >= 2 ? 8 : 0);
    const int col = tile_n + thread_in_group * 2 + (i & 1);
    if (row < rows && col < out_features) {
      float value = static_cast<float>(accumulators[i]);
      value *= activation_scale[row] * weight_scale[col];
      if (has_bias) {
        value += bias[col];
      }
      output[row * out_features + col] = static_cast<output_t>(value);
    }
  }
}

template <typename output_t>
__global__ void native_s4s4_w4_linear_optimized_kernel(
    const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    output_t* __restrict__ output,
    int rows,
    int out_features,
    int in_features,
    bool has_bias) {
  // Four warps share one 16x64 output tile. Each warp computes two adjacent
  // N=8 MMA fragments and reuses the same activation registers. N=64 provides
  // materially better small-M SM occupancy than the N=128 alternative.
  // Ampere cp.async double-buffers both packed A and B tiles. Shared uint32
  // loads then map directly to PTX S4 fragments with no per-nibble work.
  extern __shared__ __align__(16) uint8_t activation_tile[];
  constexpr int stageK = 256;
  constexpr int packedStageK = stageK / 2;
  constexpr int stageBytes = (16 + 64) * packedStageK;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 64 + warp * 16;
  const int activation_stride = in_features >> 1;
  const int weight_stride = in_features >> 1;
  int d[2][4] = {};

  issue_s4_stage(
      activation_tile,
      activation,
      weight,
      threadIdx.x,
      tile_m,
      blockIdx.x * 64,
      0,
      rows,
      out_features,
      in_features,
      activation_stride,
      weight_stride);
  for (int start_k = 0; start_k < in_features; start_k += stageK) {
    wait_for_async_copies();
    __syncthreads();
    uint8_t* current_stage =
        activation_tile + ((start_k / stageK) & 1) * stageBytes;
    if (start_k + stageK < in_features) {
      uint8_t* next_stage =
          activation_tile + (((start_k / stageK) + 1) & 1) * stageBytes;
      issue_s4_stage(
          next_stage,
          activation,
          weight,
          threadIdx.x,
          tile_m,
          blockIdx.x * 64,
          start_k + stageK,
          rows,
          out_features,
          in_features,
          activation_stride,
          weight_stride);
    }
    const uint8_t* shared_weight = current_stage + 16 * packedStageK;
#pragma unroll
    for (int inner_k = 0; inner_k < stageK; inner_k += 64) {
      const int packed_k = inner_k >> 1;
      const uint8_t* a_row0 = current_stage + group * packedStageK + packed_k;
      const uint8_t* a_row1 =
          current_stage + (group + 8) * packedStageK + packed_k;
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
          a_row0 + thread_in_group * 4);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
          a_row1 + thread_in_group * 4);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
          a_row0 + 16 + thread_in_group * 4);
      const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
          a_row1 + 16 + thread_in_group * 4);
#pragma unroll
      for (int subtile = 0; subtile < 2; ++subtile) {
        const int local_col = warp * 16 + subtile * 8 + group;
        const uint8_t* b_row =
            shared_weight + local_col * packedStageK + packed_k;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
            b_row + thread_in_group * 4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
            b_row + 16 + thread_in_group * 4);
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+r"(d[subtile][0]), "+r"(d[subtile][1]),
              "+r"(d[subtile][2]), "+r"(d[subtile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int subtile = 0; subtile < 2; ++subtile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = tile_m + group + (i >= 2 ? 8 : 0);
      const int col =
          tile_n + subtile * 8 + thread_in_group * 2 + (i & 1);
      if (row < rows && col < out_features) {
        float value = static_cast<float>(d[subtile][i]);
        value *= activation_scale[row] * weight_scale[col];
        if (has_bias) {
          value += bias[col];
        }
        output[row * out_features + col] = static_cast<output_t>(value);
      }
    }
  }
}

template <typename output_t>
__global__ void native_s4s4_w2_linear_optimized_kernel(
    const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    output_t* __restrict__ output,
    int rows,
    int out_features,
    int in_features,
    bool has_bias) {
  extern __shared__ __align__(16) uint8_t activation_tile[];
  constexpr int stageK = 256;
  constexpr int packedActivationK = stageK / 2;
  constexpr int packedWeightK = stageK / 4;
  constexpr int stageBytes =
      16 * packedActivationK + 64 * packedWeightK;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 64 + warp * 16;
  const int activation_stride = in_features >> 1;
  const int weight_stride = in_features >> 2;
  int d[2][4] = {};

  issue_s2_stage(
      activation_tile,
      activation,
      weight,
      threadIdx.x,
      tile_m,
      blockIdx.x * 64,
      0,
      rows,
      out_features,
      in_features,
      activation_stride,
      weight_stride);
  for (int start_k = 0; start_k < in_features; start_k += stageK) {
    wait_for_async_copies();
    __syncthreads();
    uint8_t* current_stage =
        activation_tile + ((start_k / stageK) & 1) * stageBytes;
    if (start_k + stageK < in_features) {
      uint8_t* next_stage =
          activation_tile + (((start_k / stageK) + 1) & 1) * stageBytes;
      issue_s2_stage(
          next_stage,
          activation,
          weight,
          threadIdx.x,
          tile_m,
          blockIdx.x * 64,
          start_k + stageK,
          rows,
          out_features,
          in_features,
          activation_stride,
          weight_stride);
    }
    const uint8_t* shared_weight =
        current_stage + 16 * packedActivationK;
#pragma unroll
    for (int inner_k = 0; inner_k < stageK; inner_k += 64) {
      const int packed_a_k = inner_k >> 1;
      const int packed_w_k = inner_k >> 2;
      const uint8_t* a_row0 =
          current_stage + group * packedActivationK + packed_a_k;
      const uint8_t* a_row1 =
          current_stage + (group + 8) * packedActivationK + packed_a_k;
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(
          a_row0 + thread_in_group * 4);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
          a_row1 + thread_in_group * 4);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
          a_row0 + 16 + thread_in_group * 4);
      const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
          a_row1 + 16 + thread_in_group * 4);
#pragma unroll
      for (int subtile = 0; subtile < 2; ++subtile) {
        const int local_col = warp * 16 + subtile * 8 + group;
        const uint8_t* b_row =
            shared_weight + local_col * packedWeightK + packed_w_k;
        const uint16_t raw0 = *reinterpret_cast<const uint16_t*>(
            b_row + thread_in_group * 2);
        const uint16_t raw1 = *reinterpret_cast<const uint16_t*>(
            b_row + 8 + thread_in_group * 2);
        const uint32_t b0 = expand_signed_s2x8_to_s4x8(raw0);
        const uint32_t b1 = expand_signed_s2x8_to_s4x8(raw1);
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+r"(d[subtile][0]), "+r"(d[subtile][1]),
              "+r"(d[subtile][2]), "+r"(d[subtile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int subtile = 0; subtile < 2; ++subtile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = tile_m + group + (i >= 2 ? 8 : 0);
      const int col =
          tile_n + subtile * 8 + thread_in_group * 2 + (i & 1);
      if (row < rows && col < out_features) {
        float value = static_cast<float>(d[subtile][i]);
        value *= activation_scale[row] * weight_scale[col];
        if (has_bias) {
          value += bias[col];
        }
        output[row * out_features + col] = static_cast<output_t>(value);
      }
    }
  }
}

}  // namespace

torch::Tensor native_s4s4_linear_cuda(
    torch::Tensor packed_activation,
    torch::Tensor activation_scale,
    torch::Tensor packed_weight,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    torch::Tensor output_template,
    int64_t weight_bits) {
  TORCH_CHECK(packed_activation.is_cuda(), "packed activation must be CUDA");
  TORCH_CHECK(packed_weight.is_cuda(), "packed weight must be CUDA");
  TORCH_CHECK(activation_scale.is_cuda(), "activation scale must be CUDA");
  TORCH_CHECK(weight_scale.is_cuda(), "weight scale must be CUDA");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  TORCH_CHECK(output_template.is_cuda(), "output template must be CUDA");
  const auto device = packed_activation.device();
  TORCH_CHECK(packed_weight.device() == device, "packed weight device mismatch");
  TORCH_CHECK(activation_scale.device() == device, "activation scale device mismatch");
  TORCH_CHECK(weight_scale.device() == device, "weight scale device mismatch");
  TORCH_CHECK(bias.device() == device, "bias device mismatch");
  TORCH_CHECK(output_template.device() == device, "output template device mismatch");
  const c10::cuda::CUDAGuard device_guard(device);
  TORCH_CHECK(packed_activation.scalar_type() == at::kByte, "activation must be uint8");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte, "weight must be uint8");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat, "activation scale must be float32");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat, "weight scale must be float32");
  TORCH_CHECK(weight_bits == 2 || weight_bits == 4, "weight_bits must be 2 or 4");
  TORCH_CHECK(packed_activation.dim() == 2, "packed activation must be 2-D");
  TORCH_CHECK(packed_weight.dim() == 2, "packed weight must be 2-D");
  TORCH_CHECK(packed_activation.is_contiguous(), "packed activation must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed weight must be contiguous");
  const int rows = static_cast<int>(packed_activation.size(0));
  const int in_features = static_cast<int>(packed_activation.size(1) * 2);
  const int out_features = static_cast<int>(packed_weight.size(0));
  TORCH_CHECK(in_features % 64 == 0, "S4 MMA requires K divisible by 64");
  TORCH_CHECK(rows > 0 && out_features > 0, "S4 MMA requires nonempty M and N");
  TORCH_CHECK(
      packed_weight.size(1) * 8 == static_cast<int64_t>(in_features) * weight_bits,
      "packed weight K does not match packed activation K");
  TORCH_CHECK(activation_scale.numel() == rows, "activation scale length mismatch");
  TORCH_CHECK(weight_scale.numel() == out_features, "weight scale length mismatch");
  TORCH_CHECK(bias.numel() == 0 || bias.numel() == out_features, "bias length mismatch");
  TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == at::kFloat, "bias must be float32");
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major >= 8, "native S4 MMA requires compute capability 8.0+");

  auto output = torch::empty(
      {rows, out_features}, output_template.options().memory_format(torch::MemoryFormat::Contiguous));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      output.scalar_type(),
      "native_s4s4_linear_cuda",
      [&] {
        if (weight_bits == 4) {
          const dim3 block(128);
          const dim3 grid((out_features + 63) / 64, (rows + 15) / 16);
          native_s4s4_w4_linear_optimized_kernel<scalar_t>
              <<<grid, block, 2 * (16 + 64) * (256 / 2), stream>>>(
                  packed_activation.data_ptr<uint8_t>(),
                  activation_scale.data_ptr<float>(),
                  packed_weight.data_ptr<uint8_t>(),
                  weight_scale.data_ptr<float>(),
                  bias.numel() ? bias.data_ptr<float>() : nullptr,
                  output.data_ptr<scalar_t>(),
                  rows,
                  out_features,
                  in_features,
                  bias.numel() != 0);
        } else {
          const dim3 block(128);
          const dim3 grid((out_features + 63) / 64, (rows + 15) / 16);
          native_s4s4_w2_linear_optimized_kernel<scalar_t>
              <<<grid, block, 2 * (16 * (256 / 2) + 64 * (256 / 4)), stream>>>(
              packed_activation.data_ptr<uint8_t>(),
              activation_scale.data_ptr<float>(),
              packed_weight.data_ptr<uint8_t>(),
              weight_scale.data_ptr<float>(),
              bias.numel() ? bias.data_ptr<float>() : nullptr,
              output.data_ptr<scalar_t>(),
              rows,
              out_features,
              in_features,
              bias.numel() != 0);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
