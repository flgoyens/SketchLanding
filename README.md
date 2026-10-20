# SketchLanding

A Julia implementation of the **sketched landing method** — a randomised, infeasible algorithm for optimisation under orthogonality constraints [Arxiv](https://arxiv.org/abs/2605.31505). In notes, find an addendum of the paper (to appear in Controlo 2026, https://controlo2026.apca.pt/).

## Problem

The algorithm solves

$$\min_{X \in \mathbb{R}^{n \times p}} f(X) \quad \text{subject to} \quad X^\top X = I_p,$$

where $f$ is smooth (possibly non-convex) and the feasible set is the **Stiefel manifold** $\mathrm{St}(n,p)$.

## Algorithm

### Infeasibility measure

Let $h(X) = \tfrac{1}{2}(X^\top X - I_p)$. Infeasibility is measured by $\psi(X) = \tfrac{1}{2}\|h(X)\|_F^2$, whose Euclidean gradient is

$$\nabla\psi(X) = X(X^\top X - I_p).$$

### Landing update

Each iterate is updated by

$$X_{k+1} = X_k - \alpha_k \bigl(\omega_T \, d_k^T + \omega_N \, d_k^N\bigr),$$

where $\omega_T, \omega_N > 0$ control the relative weight of the two components.

**Normal component** (drives feasibility):

$$d_k^N = B_{\zeta'} X_k \bigl(X_k^\top B_\zeta X_k - I_p\bigr).$$

**Tangent component** (decreases $f$ along the infeasibility level set):

$$d_k^T = \nabla f_\xi(X_k)\, X_k^\top B_\zeta B_{\zeta'} X_k \;-\; B_\zeta X_k \,\bigl(\nabla f_\xi(X_k)\bigr)^\top B_{\zeta'} X_k.$$

Here $B_\zeta = S_\zeta S_\zeta^\top$ is a **random sketch** (see below) satisfying $\mathbb{E}[B_\zeta] = I_n$. With $B_\zeta = I_n$ (identity sketch) this recovers the deterministic landing method.

### Variance reduction (SVRG)

The SVRG estimator refreshes a snapshot point $Y$ every $T$ iterations and corrects the sketched direction:

$$d_k = d^S(X_k) - d^S(Y) + d(Y),$$

where $d^S(\cdot)$ is the sketched direction and $d(\cdot)$ is the full (unsketched) direction at the snapshot. This eliminates the noise floor of the plain sketched method while keeping most iterations cheap.

---

## Features

### Sketch families (`sketch` keyword)

| Symbol | Description | Cost per iteration |
|--------|-------------|-------------------|
| `:subsample` | Selects $k$ rows of $I_n$ uniformly at random, scales by $\sqrt{n/k}$ | $O(npk)$ |
| `:gaussian` | Dense random matrix $S \in \mathbb{R}^{n \times k}$ with i.i.d. $\mathcal{N}(0, k^{-1})$ entries | $O(npk)$ |
| `:identity` | $B = I_n$ — deterministic landing, no sketching | $O(np^2)$ |

The sketch dimension $k$ is set by the `k` keyword. Smaller $k$ means cheaper iterations but more variance.

### Variance reduction (`estimator` keyword)

| Symbol | Description |
|--------|-------------|
| `:plain` | Raw sketched direction, no correction |
| `:svrg` | SVRG-style correction; snapshot period set by `T` keyword |

### Step-size modes (`stepsize_mode` keyword)

All modes take `stepsize` as the base learning rate $\alpha$. 

| Symbol | Update rule | Extra keywords |
|--------|-------------|----------------|
| `:constant` | $\alpha_k = \alpha$ | — |
| `:normalized` | $\alpha_k = \alpha / \max(1, \|d_k\|_F)$ | — |
| `:decreasing` | $\alpha_k = \alpha / \sqrt{k}$ | — |
| `:vector` | $\alpha_k = \texttt{stepsize}[k]$ (user-supplied schedule) | — |
| `:function` | $\alpha_k = \texttt{stepsize}(k, \texttt{state})$ (user-supplied function) | — |
| `:adagrad` | $v \mathrel{+}= d \odot d$; $X \mathrel{-}= \alpha \cdot d / (\sqrt{v} + \varepsilon)$ | `adagrad_ε` |
| `:adam` | Bias-corrected first and second moment estimates; $X \mathrel{-}= \alpha \cdot \hat{m} / (\sqrt{\hat{v}} + \varepsilon)$ | `adam_β1`, `adam_β2`, `adam_ε` |
| `:bb` | Barzilai–Borwein (BB1, long step) on the full direction: $\alpha_k = \langle s, s\rangle / \langle s, y\rangle$ with $s = X_k - X_{k-1}$, $y = D_k - D_{k-1}$ | `bb_tol` |
| `:bb_split` | BB1 applied separately to the tangent and normal components, giving $\alpha_T, \alpha_N$ used as $X \mathrel{-}= \alpha_T\,\omega_T T + \alpha_N\,\omega_N N$ | `bb_tol` |

> **Note:** The `:adagrad`, `:adam`, `:bb` and `:bb_split` step-size modes are experimental (beta). They have no convergence guarantees for the sketched landing method and may need tuning. Use them with care.

For `:bb` / `:bb_split`, iteration 1 uses the scalar `stepsize` as $\alpha_0$. If a BB denominator is non-finite or has magnitude $\le$ `bb_tol`, the previously accepted step size is reused ($\alpha_0$ if none accepted yet).

---

## Usage

### Defining a problem

Implement `grad!(G, X)` (in-place Euclidean gradient) and wrap it in a `FunctionProblem`:

```julia
using SketchLanding

n, p = 500, 50

# Example: Rayleigh quotient trace minimisation  f(X) = -tr(X'*A*X)
A = Symmetric(randn(n, n))
gradf! = (G, X) -> (mul!(G, A, X); rmul!(G, -2.0); nothing)

prob = FunctionProblem(n, p, gradf!)
X0   = Matrix(qr(randn(n, p)).Q)   # feasible starting point (not required)
```

Optionally provide a cost function and a stochastic gradient:

```julia
prob = FunctionProblem(n, p, gradf!;
    cost_fun        = X -> -tr(X' * A * X),
    stoch_grad!_fun = (G, X, i) -> (mul!(G, A[:, i:i], X[i:i, :]'); rmul!(G, -2.0); nothing),
)
```

### Basic run

```julia
out = sketchLanding(prob, X0;
    sketch       = :subsample,
    k            = 20,
    stepsize     = 1e-2,
    maxiter      = 2000,
)
```

### Identity sketch (deterministic landing)

```julia
out = sketchLanding(prob, X0;
    sketch   = :identity,
    stepsize = 1e-2,
    maxiter  = 1000,
)
```

### Gaussian sketch

```julia
out = sketchLanding(prob, X0;
    sketch   = :gaussian,
    k        = 30,
    stepsize = 1e-3,
    maxiter  = 5000,
)
```

### SVRG variance reduction

```julia
out = sketchLanding(prob, X0;
    sketch    = :subsample,
    k         = 20,
    estimator = :svrg,
    T         = 50,          # snapshot refresh period
    stepsize  = 5e-3,
    maxiter   = 3000,
)
```

### Adaptive step sizes

> The Adagrad, Adam and BB modes are experimental (beta). See [Step-size modes](#step-size-modes-stepsize_mode-keyword).

```julia
# Adagrad
out = sketchLanding(prob, X0;
    sketch        = :subsample,
    k             = 20,
    stepsize_mode = :adagrad,
    stepsize      = 1e-2,
    adagrad_ε     = 1e-8,
    maxiter       = 3000,
)

# Adam
out = sketchLanding(prob, X0;
    sketch        = :subsample,
    k             = 20,
    stepsize_mode = :adam,
    stepsize      = 1e-3,
    adam_β1       = 0.9,
    adam_β2       = 0.999,
    adam_ε        = 1e-8,
    maxiter       = 3000,
)

# Decreasing schedule
out = sketchLanding(prob, X0;
    stepsize_mode = :decreasing,
    stepsize      = 1e-1,    # α_k = 0.1 / sqrt(k)
    maxiter       = 5000,
)

# User-supplied schedule vector
schedule = [1e-2 / sqrt(k) for k in 1:5000]
out = sketchLanding(prob, X0;
    stepsize_mode = :vector,
    stepsize      = schedule,
    maxiter       = 5000,
)

# User-supplied function
out = sketchLanding(prob, X0;
    stepsize_mode = :function,
    stepsize      = (it, state) -> 1e-2 / (1 + 1e-3 * it),
    maxiter       = 5000,
)
```

### Controlling the normal/tangent weights

The parameters $\omega_T$ and $\omega_N$ balance descent speed against feasibility recovery:

```julia
out = sketchLanding(prob, X0;
    sketch = :subsample,
    k      = 20,
    ωT     = 1.0,
    ωN     = 2.0,   # stronger pull towards feasibility
    maxiter = 2000,
)
```

### Recording output

```julia
using SketchLanding

rec = create_recorder(history=1)   # scalar traces: cost, infeasibility, gradnorm, time
# history=0 : nothing recorded (only final iterate)
# history=1 : scalar traces
# history=2 : full iterates and step vectors

out = sketchLanding(prob, X0; record=rec, maxiter=2000)

out.costs           # Vector of f(X_k)
out.infeasibilities # Vector of ||X_k'X_k - I||_F
out.gradnorms       # Vector of ||grad_T f(X_k)||_F
out.times           # Wall-clock times per iteration
out.x_final         # Final iterate
```

### Verbosity and time limit

```julia
out = sketchLanding(prob, X0;
    maxiter    = 10_000,
    maxtime    = 60.0,       # stop after 60 seconds
    verbosity  = 2,          # print per-iteration stats
    print_every = 100,       # print every 100 iterations
)
```

### Custom problem type

For performance-critical problems, subtype `AbstractProblem` directly:

```julia
struct RayleighProblem <: AbstractProblem
    A::Symmetric{Float64, Matrix{Float64}}
end

SketchLanding.n(prob::RayleighProblem) = size(prob.A, 1)
SketchLanding.p(prob::RayleighProblem) = size(prob.A, 1) ÷ 2

function SketchLanding.grad!(G, prob::RayleighProblem, X)
    mul!(G, prob.A, X)
    rmul!(G, -2.0)
end

SketchLanding.cost(prob::RayleighProblem, X) = -tr(X' * prob.A * X)
```
