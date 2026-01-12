# Iterative Solvers
When closed-form solutions become impractical on GPU,
iterative optimization methods take over.

This library implements iterative solvers not as a fallback,
but as **first-class training backends** whose behavior is worth studying.

This document explains why iterative solvers matter and how they are used.

## 1. Why iterative solvers are unavoidable on GPU
As feature dimension grows, closed-form solvers suffer from:

- quadratic memory growth
- expensive global reductions
- increased numerical sensitivity

Iterative solvers avoid these issues by:
- operating directly on \(X\) rather than \(X^T X\)
- using matrix–vector products instead of matrix–matrix products
- converging progressively rather than solving exactly

On GPU, these properties often lead to better scalability.

## 2. The objective being optimized
For many regressors in this library, training reduces to minimizing:

\[
\frac{1}{2B} \|Xw - y\|^2 + R(w)
\]

where \(R(w)\) is a regularization term.

Examples:
- Lasso: \(R(w) = \alpha \|w\|_1\)
- ElasticNet:
  \[
  R(w) = \alpha \cdot l1\_ratio \|w\|_1
       + \frac{1}{2}\alpha(1 - l1\_ratio)\|w\|_2^2
  \]

These objectives are convex but not always smooth.

## 3. Proximal gradient methods
For objectives involving non-smooth terms (e.g. L1 regularization),
standard gradient descent is insufficient.

This library uses **proximal gradient methods**:

1. Take a gradient step on the smooth part
2. Apply a proximal operator for the non-smooth term

For L1 regularization, this proximal operator is **soft thresholding**.

This approach:
- preserves convexity guarantees
- allows efficient GPU implementation
- makes sparsity explicit

## 4. ISTA and FISTA
Two closely related solvers are implemented:

### ISTA (Iterative Shrinkage-Thresholding Algorithm)
- simple and stable
- guaranteed convergence
- relatively slow in practice

### FISTA (Fast ISTA)
- adds Nesterov-style acceleration
- often converges much faster
- can exhibit oscillatory behavior early on

Both solvers are exposed explicitly.
Acceleration is not hidden behind heuristics.

## 5. Lipschitz constants and step size
Proximal gradient methods require a step size based on
the Lipschitz constant of the gradient.

In this library:
- the Lipschitz constant is estimated via power iteration
- step size is derived explicitly
- no magic defaults are used

This makes solver behavior:
- reproducible
- inspectable
- easier to reason about

## 6. Stopping criteria and convergence
Iterative solvers stop when:
- relative parameter updates fall below a tolerance
- or a maximum iteration count is reached

Models may expose:
- `n_iter_`
- `converged_`
- `objective_` (if enabled)

These signals are intentionally exposed rather than hidden.

## 7. Why iterative solvers are implemented in Torch
Although inference uses Triton, iterative solvers are implemented in Torch.

Reasons include:
- dynamic control flow
- adaptive stopping conditions
- easier numerical debugging
- lower implementation complexity

The goal is to study optimization behavior, not to minimize kernel count.

Once training completes, inference takes over.

## Summary
Iterative solvers are not a compromise.
They are often the most appropriate choice on GPU.

By exposing their mechanics and convergence behavior,
this library treats optimization as a first-class concern.