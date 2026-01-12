# Training vs Inference
A foundational assumption in `triton-regressors` is that **training and inference
are different problems** and should be treated as such.

This distinction is often blurred in machine learning libraries.
On modern GPU systems, doing so hides important tradeoffs and leads to
confusing performance behavior.

This document explains why the library makes this separation explicit.

## 1. Different objectives, different constraints
Training and inference optimize for different goals.

**Training** prioritizes:
- numerical stability
- convergence guarantees
- flexibility in solver choice
- tolerance for higher overhead

**Inference** prioritizes:
- predictable latency
- minimal kernel launch overhead
- stable memory access patterns
- repeated execution at scale

Attempting to optimize both with the same abstractions usually results
in unnecessary compromises.

## 2. Training characteristics
Training classical regression models on GPU can involve:

- explicit matrix construction (e.g. normal equations)
- iterative optimization (e.g. ISTA, FISTA, LBFGS)
- adaptive stopping criteria
- higher precision arithmetic

These operations often:
- allocate large intermediate tensors
- involve global synchronization
- exhibit irregular control flow
- depend on problem conditioning

Because training is typically performed infrequently, the library allows
training backends to make tradeoffs that would be unacceptable for inference.

## 3. Inference characteristics
Inference has much stricter requirements.

For all regressors in this library, inference reduces to a simple form:

\[
y = Xw + b
\]

This operation:
- is embarrassingly parallel
- has regular memory access
- does not require autograd
- benefits from specialization

Inference kernels in this library:
- are written in Triton
- assume GPU-resident parameters
- avoid unnecessary abstraction layers
- perform no dynamic allocation

## 4. Why inference always uses Triton
Although Torch provides highly optimized matrix multiplication,
this library uses Triton for inference by design.

Reasons include:

1. **Predictability**  
   Triton kernels expose memory access patterns and launch behavior directly.

2. **Specialization**  
   Kernels can be tailored to the exact operation required, not a general GEMM.

3. **Isolation**  
   Inference kernels are decoupled from training and autograd.

4. **Transparency**  
   Performance behavior is explicit rather than opaque.

This choice favors control and clarity over convenience.

## 5. Why training does not use Triton
Training solvers are intentionally implemented using Torch rather than Triton.

Training often requires:
- complex control flow
- adaptive iteration
- solver-specific heuristics
- easier numerical debugging

Implementing these solvers at the kernel level would increase complexity
without improving insight.

Torch provides a stable environment for experimentation and validation.
Once training produces parameters, inference takes over.


## 6. GPU-resident parameters as a contract
After training, all learned parameters are moved to the GPU and remain there.

This is a hard contract enforced by the library.

Benefits include:
- consistent inference behavior
- no hidden CPU↔GPU transfers
- simpler performance analysis
- easier debugging

Attempting to run inference with CPU-resident parameters results in an error,
making misuse explicit.

## 7. Practical implications
This separation allows the library to:
- benchmark inference independently of training
- study convergence behavior without conflating it with kernel performance
- evolve training backends without affecting inference correctness

The result is a clearer mental model of how classical regressors behave on GPU.


## Summary
Separating training and inference is not an implementation detail.
It is a deliberate design decision that shapes the entire library.
This separation enables clarity, reproducibility, and disciplined GPU design.