# Design Tradeoffs and Limitations
Every system reflects a set of tradeoffs.
This document makes the limitations and intentional choices
of `triton-regressors` explicit.

## 1. Why this library is opinionated
This library favors:
- explicit behavior over abstraction
- clarity over convenience
- inspectability over completeness

As a result, it is not optimized for:
- beginner usability
- API compatibility
- feature breadth

This is a deliberate choice.

## 2. GPU-first does not mean GPU-only
Although inference is GPU-only, training may:
- use Torch solvers
- rely on CPU-backed libraries
- trade performance for stability

This reflects real-world workflows where training and inference
have different constraints.

## 3. Limited model scope
The library focuses exclusively on classical regressors:
- linear
- ridge
- lasso
- elasticnet

It does not include:
- logistic regression (intentionally excluded)
- tree-based models
- neural networks

Expanding beyond this scope would dilute the core focus.

## 4. No automatic hyperparameter tuning
Hyperparameters such as:
- regularization strength
- solver choice
- tolerance thresholds

are intentionally exposed.

The library does not attempt to automate these decisions,
as doing so would obscure optimization behavior.

## 5. Numerical precision choices
Training may use:
- higher precision arithmetic
- more stable solvers

Inference uses:
- float32
- fixed kernel behavior

This asymmetry reflects practical deployment concerns.

## 6. Triton kernel specialization
Inference kernels are specialized and explicit.

This improves:
- performance predictability
- clarity of behavior

But it reduces:
- flexibility
- ease of extension

Kernel specialization is a conscious tradeoff.

## 7. Educational value over generality
Many design decisions prioritize:
- learning
- experimentation
- reasoning about systems

This makes the library well-suited for:
- research
- teaching
- systems exploration

It may be less suitable for production use without adaptation.

## Summary
`triton-regressors` is not designed to be everything.

It is designed to be **clear**, **honest**, and **educational**.

By making tradeoffs explicit, the library invites understanding
rather than hiding complexity.