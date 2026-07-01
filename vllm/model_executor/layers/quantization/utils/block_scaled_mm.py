# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared Triton kernel, config loader, and matmul wrapper for
block-scaled W8A8 (FP8 and INT8) matrix multiplication.

``fp8_utils`` and ``int8_utils`` had near-identical copies of three
components:

* a Triton ``@jit`` matmul kernel,
* a JSON config-file loader, and
* a Python wrapper that validates shapes, picks a config, and launches
  the kernel.

This module de-duplicates them into dtype-parameterised helpers.
"""

import functools
import json
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# ── Triton kernel ──────────────────────────────────────────────────────
@triton.jit
def _w8a8_block_scaled_mm_kernel(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors ``A`` and ``B`` with block-wise quantization,
    and store the result in output tensor ``C``.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        # .to(tl.float32) is a no-op for FP8 (already float32 accumulator)
        # but required for INT8 dot products.
        accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ── Config loader ─────────────────────────────────────────────────────
@functools.lru_cache
def get_w8a8_block_configs(
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    dtype_label: str,
) -> dict[int, Any] | None:
    """Return optimized configurations for a W8A8 block-scaled kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the kernel.  To evaluate the kernel
    on a given batch size *bs*, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the
    kernel.

    Args:
        N: Number of output features.
        K: Number of input features.
        block_n: Block size along the N dimension.
        block_k: Block size along the K dimension.
        dtype_label: Label used in the JSON filename, e.g.
            ``"fp8_w8a8"`` or ``"int8_w8a8"``.
    """

    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},"
        f"dtype={dtype_label},"
        f"block_shape=[{block_n},{block_k}].json"
    )

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block %s kernel.",
                config_file_path,
                dtype_label.upper(),
            )
            return {int(key): val for key, val in json.load(f).items()}

    logger.warning(
        "Using default W8A8 Block %s kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        dtype_label.upper(),
        config_file_path,
    )
    return None


# ── Matmul wrapper ─────────────────────────────────────────────────────
def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
    *,
    dtype_label: str,
    default_num_stages: int = 2,
) -> torch.Tensor:
    """Block-scaled W8A8 matrix multiplication.

    This is the shared implementation behind
    ``w8a8_triton_block_scaled_mm`` (FP8) and
    ``w8a8_block_int8_matmul`` (INT8).

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for ``A``.
        Bs: The per-block quantization scale for ``B``.
        block_size: The block size for per-block quantization.  It
            should be 2-dim, e.g., ``[128, 128]``.
        output_dtype: The dtype of the returned tensor.
        dtype_label: Label for config lookup (``"fp8_w8a8"`` or
            ``"int8_w8a8"``).
        default_num_stages: Default ``num_stages`` when no tuned config
            is available (2 for FP8, 3 for INT8).

    Returns:
        The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = get_w8a8_block_configs(N, K, block_size[0], block_size[1], dtype_label)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": default_num_stages,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_scaled_mm_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C
