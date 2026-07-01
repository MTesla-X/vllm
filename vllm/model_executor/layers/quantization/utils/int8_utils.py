# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/blob/4cb53ecd0cffceb6dee5c011a58f65997a86f151/python/sglang/srt/layers/quantization/int8_kernel.py
import logging
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.block_scaled_mm import (  # noqa: E501
    get_w8a8_block_configs,
    w8a8_block_matmul,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)


def apply_w8a8_block_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    output = w8a8_block_int8_matmul(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input.dtype
    )

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def input_to_int8(
    x: torch.Tensor, dtype: torch.dtype = torch.int8
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to int8 values with
    tensor-wise quantization."""
    iinfo = torch.iinfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    int8_min, int8_max = iinfo.min, iinfo.max
    scale = int8_max / amax
    x_scl_sat = (x * scale).clamp(min=int8_min, max=int8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """This function conducts block-wise dequantization.
    The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are dequantized tensor.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ] *= x_s[j][i]

    return x_dq_block


if current_platform.is_rocm():

    @triton.jit
    def round_int8(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int8)


elif current_platform.is_xpu():

    @triton.jit
    def round_int8(x):
        return tl.extra.intel.libdevice.round(x).to(tl.int8)

else:

    @triton.jit
    def round_int8(x):
        return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = round_int8(x_q)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, original_shape[-1])
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    x = x.contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    x_q = x_q.view(*original_shape)
    scales = scales.view(*original_shape[:-1], 1)
    return x_q, scales


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8`
            is supported for now.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
            scaling factor for quantization.
    """
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    # prefer CUDA kernel if available
    if current_platform.is_cuda():
        torch.ops._C.per_token_group_quant_int8(
            x, x_q, x_s, group_size, eps, float(int8_min), float(int8_max)
        )
        return x_q, x_s

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def get_w8a8_block_int8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    """Return optimized configurations for the W8A8 block INT8 kernel."""
    return get_w8a8_block_configs(N, K, block_n, block_k, "int8_w8a8")


def w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Block-scaled INT8 W8A8 matrix multiplication.

    Delegates to the shared :func:`w8a8_block_matmul` implementation.
    """
    return w8a8_block_matmul(
        A,
        B,
        As,
        Bs,
        block_size,
        output_dtype,
        dtype_label="int8_w8a8",
        default_num_stages=3,
    )
