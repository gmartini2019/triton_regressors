# Performance Benchmarks
Performance numbers are only meaningful when interpreted correctly.

This document explains how benchmarks in `triton-regressors` are structured
and how to reason about the results they produce.

## 1. What is being benchmarked
Benchmarks in this repository measure **inference performance only**.

This is intentional.

Training:
- is performed infrequently
- may use different solvers
- may prioritize stability over speed

Inference:
- is executed repeatedly
- is latency-sensitive
- benefits most from GPU acceleration

By isolating inference, benchmarks reflect the true performance
characteristics of the deployed model.

## 2. What is not being benchmarked
Benchmarks do **not** measure:
- training time
- solver convergence speed
- preprocessing overhead
- data loading costs

These aspects are studied separately in convergence experiments
or design discussions.

Conflating them with inference would obscure interpretation.

## 3. CPU vs GPU comparisons
Some benchmarks compare:
- scikit-learn inference on CPU
- Triton-based inference on GPU

These comparisons are illustrative, not competitive.
They demonstrate:
- how inference scales with batch size
- how GPU overhead amortizes
- where GPU execution becomes advantageous

They are not intended as leaderboard claims.

## 4. Interpreting batch size effects
GPU inference performance depends heavily on batch size.

Typical observations include:
- higher overhead at very small batch sizes
- rapid improvement as batch size increases
- eventual saturation of memory bandwidth or compute

These trends are expected and reflect kernel launch and memory behavior.

Benchmarks are structured to make these effects visible.

## 5. Why Triton kernels are benchmarked separately
Inference kernels are benchmarked independently of training
to isolate kernel behavior.

This allows:
- direct comparison with Torch-based inference
- identification of launch overhead
- analysis of memory access patterns

The goal is understanding, not absolute throughput.

## 6. Reproducibility
All benchmarks:
- fix random seeds
- synchronize CUDA explicitly
- run multiple iterations
- report averaged timings

This reduces noise and improves interpretability.

## Summary
Performance benchmarks in this library are designed to be:
- focused
- interpretable
- honest

They complement convergence experiments and help connect
solver behavior to real-world inference performance.