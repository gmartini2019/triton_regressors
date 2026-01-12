# Convergence Experiments
Understanding how optimization behaves is as important as
measuring final performance.

This library includes explicit convergence experiments to study:
- solver dynamics
- acceleration effects
- conditioning
- regularization geometry

This document explains how to interpret those experiments.

## 1. Why convergence matters
Two solvers may produce the same final solution but behave very differently:

- one may converge smoothly
- another may oscillate
- one may make rapid early progress
- another may stall

Ignoring convergence hides these differences.

## 2. What is measured
Convergence experiments track the **objective value** over iterations:

\[
\frac{1}{2B} \|Xw - y\|^2 + R(w)
\]

This scalar summarizes both:
- data fit
- regularization effects

Tracking the objective provides a clear, model-agnostic signal.

## 3. Lasso convergence
For Lasso, convergence plots typically show:

- rapid initial decrease
- diminishing returns near the optimum
- sharper kinks due to the L1 penalty

Comparing ISTA and FISTA reveals:
- FISTA’s faster early progress
- possible oscillations near convergence
- similar final objective values

These behaviors are expected and informative.

## 4. ElasticNet convergence
ElasticNet introduces an L2 component, which smooths the objective.

As a result:
- convergence is often smoother than Lasso
- acceleration is more effective
- oscillations are reduced

This demonstrates how problem geometry influences solver behavior.

## 5. Conditioning effects
Ill-conditioned data amplifies differences between solvers.

In such cases:
- step size estimation becomes critical
- acceleration may help or hurt
- convergence may slow significantly

These effects are visible only when convergence is measured explicitly.

## 6. Why plots are preferred over tables
Single numbers hide behavior.

Convergence plots reveal:
- transient dynamics
- stability issues
- solver sensitivity

They provide intuition that performance benchmarks alone cannot.

## 7. Reproducibility
All convergence experiments in this repository:
- fix random seeds
- use explicit solver parameters
- run on GPU

This ensures that observed behavior reflects solver dynamics,
not experimental noise.

## Summary
Convergence experiments turn optimization from a black box
into an observable process.
They complement performance benchmarks and provide insight
into why certain solvers behave the way they do.