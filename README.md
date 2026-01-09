# triton-regressors
A GPU-first library for classical regression models that explicitly separates
training algorithms from high-performance Triton inference.

This project focuses on:
- linear, ridge, lasso, and elasticnet regression
- explicit optimization behavior (closed-form vs iterative solvers)
- GPU-resident inference via Triton
- reproducible performance and convergence experiments

This is **not** a drop-in replacement for scikit-learn.
It is an educational and systems-oriented library for understanding
how classical models behave on modern GPU hardware.

## Key ideas
- Training and inference are different problems
- Closed-form solutions do not scale the same way on GPU
- Iterative solvers expose optimization dynamics
- Triton enables predictable, low-overhead inference

## Repository structure
- `triton_regressors/` — library code
- `training/` — optimization backends
- `linear/`, `ridge/`, `lasso/`, `elasticnet/` — model implementations
- `experiments/` — convergence and optimization studies
- `benchmarks/` — performance comparisons
- `docs/` — guided explanations

## Status
This project is research-grade and intentionally opinionated.
The API may change as design tradeoffs are explored.
If you have another regressor to add or can improve upon the math usability, please do, everyone is welcomed to contribute.
