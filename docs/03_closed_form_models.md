# Closed-Form Models on GPU
Closed-form solutions are often viewed as the “best” way to solve
classical regression problems.

On CPU, this intuition is frequently correct.
On GPU, it is more complicated.

This document explains when closed-form solutions make sense on GPU,
when they do not, and why this library treats them carefully.

## 1. What “closed-form” means in practice
For linear and ridge regression, closed-form solutions typically involve
solving a system of the form:

\[
(X^T X + \alpha I) w = X^T y
\]

This requires:
- forming \(X^T X\)
- forming \(X^T y\)
- solving a dense linear system

These operations have well-understood numerical properties
and strong theoretical guarantees.

## 2. Why closed-form is attractive
Closed-form solvers offer several benefits:

- deterministic solutions
- no hyperparameters related to optimization
- clear convergence guarantees
- simple correctness reasoning

For small to moderate problem sizes, they are often the fastest and
most reliable choice.

## 3. Why closed-form can be problematic on GPU
On GPU, closed-form methods introduce several challenges:

### Memory cost
- \(X^T X\) is a dense \(D \times D\) matrix
- memory scales quadratically with feature dimension

### Compute patterns
- forming \(X^T X\) requires global reductions
- synchronization costs can dominate runtime

### Numerical considerations
- conditioning of \(X^T X\) can be worse than that of \(X\)
- higher precision may be required for stability

As \(D\) grows, these issues quickly outweigh the theoretical elegance
of the closed-form solution.

## 4. When closed-form still makes sense
Closed-form solvers remain useful when:
- feature dimension is moderate
- data is reasonably well-conditioned
- deterministic solutions are required
- training is performed infrequently

In these regimes, closed-form solvers provide strong baselines
and useful reference behavior.

## 5. How this library handles closed-form solvers
`triton-regressors` includes closed-form solvers where they are appropriate,
but treats them as **one option among many**, not the default answer.

For example:
- linear regression uses a closed-form solution
- ridge regression exposes both closed-form and iterative solvers

This allows users to:
- compare solver behavior directly
- understand performance tradeoffs
- choose the right approach for their constraints

## 6. Closed-form vs iterative: a comparison
| Aspect            | Closed-form        | Iterative solvers      |
||--||
| Memory usage     | High (O(D²))       | Lower (O(B·D))         |
| Determinism      | Yes                | Approximate            |
| Convergence      | Exact              | Tolerance-based        |
| GPU scalability  | Limited            | Often better           |
| Flexibility      | Low                | High                   |

Neither approach dominates in all regimes.

## 7. Why closed-form is still included
Closed-form solvers serve an important role:

- correctness reference
- baseline performance comparison
- educational contrast with iterative methods

Including them makes tradeoffs visible rather than implicit.

## Summary
Closed-form solutions are elegant but not universally optimal on GPU.
By including them explicitly — and not treating them as default —
this library aims to make their strengths and limitations clear. Understanding when *not* to use a closed-form solution
is as important as knowing how to compute one.