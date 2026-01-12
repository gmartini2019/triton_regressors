# Library Design

This document explains the architectural decisions behind the `triton-regressors` repository.

The goal of this library is not to provide a convenient API AND to make the behavior of classical regression models on GPUs **explicit, inspectable, and reproducible**.

To achieve that, the library is organized around a small number of
non-negotiable design principles.

## 1. A minimal, explicit model contract
All models in this library implement the same base contract:

- `fit(X, y)`
- `predict(X)`

After `fit`, every model **must** expose:

- `coef_` — learned weights (CUDA tensor)
- `intercept_` — bias term (CUDA tensor)
- `n_features_in_` — input dimensionality

This contract is enforced via a shared base class.

There are **no implicit CPU fallbacks**.
All learned parameters live on the GPU, even if parts of training happen on CPU.
This keeps inference behavior consistent and predictable.

Optional attributes such as:
- `objective_`
- `n_iter_`
- `converged_`

are exposed only by iterative solvers, where they are meaningful.

## 2. Training and inference are separate concerns
A core assumption of this library is that **training and inference are
fundamentally different problems**.

Training:
- may involve large intermediate state
- may benefit from higher-level linear algebra
- may use different numerical tradeoffs
- may run once or infrequently

Inference:
- must be fast
- must be predictable
- must minimize overhead
- is run repeatedly

For this reason, training logic and inference logic are explicitly separated.

### Training backends
Training lives in `triton_regressors/training/` and is allowed to use:
- closed-form solvers
- iterative optimization
- Torch autograd
- (optionally) external libraries for bootstrapping

Each training backend returns **only learned parameters**.
It does not perform inference.

### Inference kernels
Inference is always performed on the GPU using Triton kernels.
Inference kernels:
- do not know how the model was trained
- only consume `(X, coef_, intercept_)`
- are optimized independently of training

This separation makes performance behavior easier to reason about
and avoids conflating numerical concerns with kernel design.

## 3. Model files are thin by design
Each model implementation (e.g. linear, ridge, lasso, elasticnet)
acts as an **orchestrator**, not a solver.

Model files:
- choose a training backend
- store learned parameters
- enforce the base contract
- dispatch to the appropriate inference kernel

They do **not**:
- implement optimization loops
- contain solver-specific math
- duplicate training logic

If a model file grows large, that is considered a design smell.

## 4. Kernels are co-located with models
Triton kernels live next to the model they serve.

For example:
- `linear/model.py`
- `linear/kernels.py`

This is an intentional choice.

Inference kernels are tightly coupled to:
- parameter layout
- expected input shapes
- numerical assumptions of the model

Treating kernels as globally shared utilities often leads to
leaky abstractions and hidden coupling.

Co-location makes ownership explicit and evolution safer.

## 5. Multiple training backends are first-class
Some models expose more than one training backend.

For example, ridge regression supports:
- a closed-form solver using explicit `XᵀX` construction
- an iterative LBFGS solver that avoids forming `XᵀX`

Both are valid.
Both are useful under different constraints.

The model API makes these choices explicit via a `solver` argument.
There are no silent fallbacks.

This allows:
- controlled performance comparisons
- clearer numerical reasoning
- better documentation of tradeoffs

## 6. Explicit non-goals
This library intentionally does **not** aim to be:

- a drop-in replacement for scikit-learn
- a general-purpose ML framework
- a collection of every possible regression model
- a benchmarking leaderboard

It focuses on a small set of classical regressors and explores them deeply.

Logistic regression, tree-based models, and deep learning are
intentionally out of scope.

## 7. Why scikit-learn wrappers exist (and why they are minimal)
Scikit-learn wrappers, if present, exist only as **explicit escape hatches**.

They may be used to:
- bootstrap coefficients
- provide reference solutions
- validate correctness

They are never used implicitly and are not considered core training backends.

The long-term goal is not to depend on scikit-learn,
but to make comparisons with it clear and honest.

## Summary
This library is designed to make tradeoffs visible.

Rather than hiding optimization details behind a uniform API,
it exposes them in a controlled, structured way.

The result is a codebase that favors:
- clarity over convenience
- explicitness over abstraction
- systems understanding over feature breadth