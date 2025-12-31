import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4),
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def logistic_predict_kernel(
    X_ptr,
    W_ptr,
    b_ptr,
    Y_ptr,
    B,
    D,
    stride_xb,
    stride_xd,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for d in range(0, D, BLOCK):
        x = tl.load(
            X_ptr + pid * stride_xb + (d + offs) * stride_xd,
            mask=(d + offs) < D,
            other=0.0,
        )
        w = tl.load(
            W_ptr + d + offs,
            mask=(d + offs) < D,
            other=0.0,
        )
        acc += x * w

    z = tl.sum(acc, axis=0)
    b = tl.load(b_ptr)
    z += b

    y = 1.0 / (1.0 + tl.exp(-z))

    tl.store(Y_ptr + pid, y)
