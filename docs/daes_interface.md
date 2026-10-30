# Original DAES interface

## Provenance

The original DAES implementation was read from the supplied snapshot, not modified, and is
**not** vendored into this repository.

- Archive: `C:\Users\chaya\Downloads\drift-aware-eigen-scaling-lab-main.zip`
- Archive SHA-256: `C3A6ADF517888BE3BC7457B9AB7950F4559E99A1922C705424A02E7672229FE4`
- Extracted file: `drift-aware-eigen-scaling-lab-main/drift_aware_eigen_scaling.py`
- File SHA-256: `6B9AF5B20C2468A90A8B56977070B6FA96B6E072571D02C23C47D5BB58E6F85A`
- The snapshot was extracted to a temporary reference directory outside this repository.

The original README describes the project as "an educational simulation for a maths project:
gradient descent on a strongly convex quadratic whose Hessian changes over time." It is not a
radiation-specific method. The paper in the separate final package calls the method a
"drift-aware eigenvalue step-size heuristic" and explicitly declines to claim a general
optimizer or a non-stationary convergence theorem.

## Exact original equations

Objective:

```
f_t(x) = 0.5 (x − x*_t)^T A_t (x − x*_t)
∇f_t(x) = A_t (x − x*_t)
x_{t+1} = x_t − η_t ∇f_t(x_t)
```

Step size:

```
η_t = γ / ( (1 + s) λ̂_t )
```

Spectral estimate (`SpectralTracker.update`):

```
for k iterations:
    v ← A_t v / ||A_t v||
r_t = v^T A_t v
d_t = |r_t − r_{t−1}| / max(|r_{t−1}|, 1e−12)
α_t = clip( α_0 + κ d_t, α_0, α_max )
λ̂_t = (1 − α_t) λ̂_{t−1} + α_t r_t
if r_t > λ̂_{t−1}:  λ̂_t = max(λ̂_t, r_t)
```

Original defaults: `γ = 1.78`, `s = 0.10`, power iterations `7`, `α_0 = 0.16`,
`α_max = 0.82`, `κ = 7.0`. The initial vector in the original demo is
`(0.83, 0.41) / ||(0.83, 0.41)||`.

## Interface used by Rad-DAES

Rad-DAES re-implements the tracker behind a small, documented interface. The equation form
is preserved; only the matrix supplied to it changes.

```python
class DAESController:
    def step(self, curvature: np.ndarray) -> DAESStep:
        """Return (eta, raw_rayleigh, filtered_lambda, drift_score, alpha)."""

    def reset(self, curvature: np.ndarray) -> None: ...
```

- **Input:** a symmetric positive-definite curvature matrix `H_t`. Rad-DAES supplies
  `H_t = Σ_{t−1}^{−1}`.
- **Output:** a scalar step `η_t` and the tracker diagnostics.
- **State:** power-iteration vector, previous raw Rayleigh quotient, filtered eigenvalue.
- **No matrix update is performed by DAES.** The caller applies the scalar step.

The original demo is 2-D. Rad-DAES uses `D = 6`; the power iteration and Rayleigh quotient
are dimension-agnostic. The initial vector is a fixed unit vector of length 6, specified in
`configs/model.yaml`. This is the only interface-level change.

## What is not part of original DAES

The supplied Gemini planning PDFs contain a different, multi-direction rule:

```
H_t = β H_{t−1} + (1 − β) ∇ℓ_t ∇ℓ_t^T
H_t = V_t Λ_t V_t^T
Φ_{ii} = η_0 / ( sqrt(λ_i) + δ_0 exp(−κ ρ_t) )
θ_t = θ_{t−1} + V_t Φ_t V_t^T r_t
```

This is **not** the original DAES implementation. It uses a different curvature recursion,
a different damping law, a matrix-valued gain, and a different update direction. It is
recorded in `docs/decisions.md` as an unverified proposal and is not implemented as DAES.
The first draft does not test it. A future study could test it as a separate method, clearly
labelled.

## Conformance check

`tests/test_daes.py` checks the implementation against hand-computed values and against the
original 2-D schedule. The test does not import the external snapshot; it reproduces the
original equations independently. The external snapshot is retained only for provenance.
