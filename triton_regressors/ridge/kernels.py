import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64, "BLOCK_B": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_B": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 256, "BLOCK_B": 512}, num_warps=8),
    ],
    key=["B", "D"],
)
@triton.jit
def xty_kernel(
    X_ptr,
    y_ptr,
    out_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    stride_xb,
    stride_xd,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for b0 in range(0, B, BLOCK_B):
        b = b0 + tl.arange(0, BLOCK_B)

        y = tl.load(y_ptr + b, mask=b < B, other=0.0).to(tl.float32)
        x = tl.load(
            X_ptr + b[:, None] * stride_xb + d[None, :] * stride_xd,
            mask=(b[:, None] < B) & (d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(x * y[:, None], axis=0)

    tl.store(out_ptr + d, acc, mask=d < D)


@triton.autotune(
    configs=[
        triton.Config({"BM": 32, "BN": 32, "BK": 256}, num_warps=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 256}, num_warps=8),
        triton.Config({"BM": 64, "BN": 64, "BK": 512}, num_warps=8),
    ],
    key=["B", "D"],
)
@triton.jit
def xtx_kernel(
    X_ptr,
    out_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    stride_xb,
    stride_xd,
    stride_ob,
    stride_od,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)  
    pid_n = tl.program_id(1)  

    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for b0 in range(0, B, BK):
        b = b0 + tl.arange(0, BK)

        A = tl.load(
            X_ptr + b[:, None] * stride_xb + m[None, :] * stride_xd,
            mask=(b[:, None] < B) & (m[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        Bm = tl.load(
            X_ptr + b[:, None] * stride_xb + n[None, :] * stride_xd,
            mask=(b[:, None] < B) & (n[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(tl.trans(A), Bm)

    tl.store(
        out_ptr + m[:, None] * stride_ob + n[None, :] * stride_od,
        acc,
        mask=(m[:, None] < D) & (n[None, :] < D),
    )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def diag_add_kernel(
    A_ptr,
    alpha,
    D: tl.constexpr,
    stride0,
    stride1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < D

    diag_ptrs = A_ptr + i * (stride0 + stride1)

    a = tl.load(diag_ptrs, mask=mask, other=0.0).to(tl.float32)

    a += tl.full([], alpha, tl.float32)

    tl.store(diag_ptrs, a, mask=mask)
