# triton_regressors

**GPU-accelerated implementations of classical regression models using Triton.**

This repo is an **educational + systems-focused project** that reimplements the **inference path** of classic scikit-learn regression models using **Triton GPU kernels**, with a strong emphasis on:

- performance
- correctness
- memory access patterns
- real-world inference tradeoffs

This is **not** a drop-in replacement for scikit-learn.

---

## 🧠 Motivation

Most production ML systems still rely heavily on **classical models** (e.g. linear regression, ridge, elastic net), but these models are almost always run on CPU.

Meanwhile:
- GPUs sit idle
- inference latency matters
- batching patterns vary wildly

This project explores:
> **When does it actually make sense to run classical ML inference on GPUs?**

The long-term goal is to build intuition and tooling relevant to:
- NVIDIA Triton
- Forest Inference Library (FIL)
- GPU inference systems beyond neural networks

---

## 🎯 Project Scope

### What this repo **does**
- Reimplements **inference-only** versions of classic regression models
- Uses **Triton** for GPU kernels
- Compares:
  - output correctness vs scikit-learn
  - latency and throughput vs CPU
- Provides benchmarks across batch sizes

### What this repo **does NOT do**
- ❌ Training on GPU
- ❌ Replace scikit-learn
- ❌ Support sparse inputs
- ❌ Handle NaNs or missing values
- ❌ Support float64 (v1)

These are **explicit non-goals**.

---

## 📦 Models (Planned)

| Model | Status |
|-----|------|
| LinearRegression | 🚧 In progress |
| Ridge | ⏳ Planned |
| Lasso (inference) | ⏳ Planned |
| ElasticNet | ⏳ Planned |
| Polynomial Regression | ⏳ Planned |

---

## 🔬 Design Philosophy

### 1. Inference-first
Training is done using **scikit-learn on CPU**.  
Only trained parameters (`coef_`, `intercept_`) are passed to Triton.

### 2. Math-driven, not sklearn-driven
We do **not** mirror scikit-learn internals.

Instead:
- define the math contract
- implement it from scratch
- use scikit-learn only as a **numerical oracle**

### 3. Explicit constraints
- float32 only
- dense inputs only
- deterministic outputs

This keeps the focus on **systems-level clarity**.

---

## 🗂️ Repository Structure

```text
triton_regressors/
├── triton_regressors/
│   ├── linear/
│   │   ├── kernels.py
│   │   ├── model.py
│   │   └── reference.py
│   │
│   ├── training/
│   │   └── sklearn_train.py
│   │
│   └── utils/
│       ├── validation.py
│       └── timers.py
│
├── benchmarks/
│   ├── linear_vs_sklearn.py
│   └── batch_sweep.py
│
├── tests/
│   ├── test_linear_correctness.py
│   └── test_shapes.py
│
└── README.md
```

---

## 🔁 Typical Workflow

1. **Train on CPU (scikit-learn)**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
```

2. **Extract weights**
```python
W = model.coef_
b = model.intercept_
```

3. **Move to GPU**
```python
W_t = torch.tensor(W, device="cuda", dtype=torch.float32)
b_t = torch.tensor(b, device="cuda", dtype=torch.float32)
X_t = torch.tensor(X, device="cuda", dtype=torch.float32)
```

4. **Run Triton inference**
```python
from triton_regressors.linear import TritonLinearRegression

triton_model = TritonLinearRegression(W_t, b_t)
y_pred = triton_model.predict(X_t)
```

5. **Validate correctness**
```python
np.allclose(y_pred.cpu().numpy(), model.predict(X), atol=1e-5)
```

---

## 📊 Benchmarks

Benchmarks focus on:
- latency (not just throughput)
- batch size crossover points
- GPU launch overhead vs compute

Example:
```bash
python benchmarks/linear_vs_sklearn.py
```

---

## 🧪 Testing

Run:
```bash
pytest tests/
```

---

## 🚧 Status

This repo is under **active development**.  
APIs may change as models and kernels evolve.

---

## 📜 License
MIT
