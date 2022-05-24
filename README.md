# Dito

[![CI](https://github.com/lassepe/Dito.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/lassepe/Dito.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lassepe/Dito.jl/branch/main/graph/badge.svg?token=i1g7Vf5xOY)](https://codecov.io/gh/lassepe/Dito.jl)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Dito.jl is a package for **Di**fferentiable **T**rajetory **O**ptimization in Julia. It supports both forward and reverse mode differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) and therefore integrates seamlessly with machine learning frameworks such as [Flux.jl](https://github.com/FluxML/Flux.jl).

---

A substantial part of machine learning (ML) algorithms relies upon the ability to propagate gradient signals through the entire learning pipeline.
Traditionally, such models have been mostly limited to artificial neural networks and "simple" analytic functions.
Recent work has focused on extending the class of admissible models for gradient-based learning by making all sorts of procedures differentiable.
These efforts range from [differentiable physics engines](https://arxiv.org/pdf/2103.16021.pdf) over [differentiable rendering](https://arxiv.org/pdf/2006.12057.pdf?ref=https://githubhelp.com) to [differentiable optimzation](https://arxiv.org/pdf/1703.00443.pdf).

Dito.jl focus on a special case of the latter category, differentiable trajectory optimization.
As such, Dito algorithmically provides a (local) answer to the question:

> *"How does the optimal solution of an inequality constrained trajectory optimization problem change if the problem changes?"*.

This implementation was initially developed as part of our research on [Learning Mixed Strategies in Trajectory Games](https://arxiv.org/pdf/2205.00291.pdf).
There, Dito allowed us to efficiently train a neural network pipeline that rapidly generates feasible equilibrium trajectories in multi-player non-cooperative dynamic games.
Since this component has proven to be very useful in that context, we have since decided to factor it out into a stand-alone package.

## Installation

To install Dito, imply add it via Julia's package manage from the REPL:

```julia
# hit `]` to enter "pkg"-mode of the REPL
pkg> add Dito
```
## Usage

Below we construct a parametric optimization problem for a 2D integrator with 2 states, 2 inputs
over a horizon of 10 stages with box constraints on states and inputs.

Please consult the documentation for each of the types below for further information. For example, just type `?ParametricTrajectoryOptimizationProblem` to learn more about the problem setup.
You can also consult the [tests](test/runtests.jl) as an additional source of implicit documentation.

### 1. Problem Setup

The entry-point for getting started with this package is to set up you problem of choice as an `ParametricTrajectoryOptimizationProblem`.


```julia
horizon = 10
state_dim = 2
control_dim = 2
cost = (xs, us, params) -> sum(sum((x - params).^2) + sum(u.^2) for (x, u) in zip(xs, us))
dynamics = (x, u, t) -> x + u
inequality_constraints = let
    state_constraints = state -> [state .+ 0.1; -state .+ 0.1]
    control_constraints = control -> [control .+ 0.1; -control .+ 0.1]
    (xs, us) -> [
        mapreduce(state_constraints, vcat, xs)
        mapreduce(control_constraints, vcat, us)
    ]
end

problem = ParametricTrajectoryOptimizationProblem(
    cost,
    dynamics,
    inequality_constraints,
    state_dim,
    control_dim,
    horizon
)

```

### 2. Optimizer Setup

Given an instance of the `ParametricTrajectoryOptimizationProblem`, you can construct an `Optimizer` for the problem.

```julia
backend = QPSolver()
optimizer = Optimizer(problem, backend)
```

### 3. Solving the Problem

Given an optimizer, we can solve a problem instance for a given initial state `x0` and parameter values `params`.

```julia
x0 = zeros(state_dim)
optimizer(x0, params)
```

## Background

Dito achieves differentiable trajectory optimization by augmenting existing optimization routines with custom derivative rules that apply the [implicit function theorem (IFT)](https://en.wikipedia.org/wiki/Implicit_function_theorem) to the resulting KKT-system.
Through this formulation, Dito avoids differentiation of the entire (potentially iterative) algorithm, leading to substantially accelerated derivative computation and facilitating differentiation of optimization back-ends that are not written in pure Julia.

The following body of work provides more information about this IFT-based differentiation approach:

- [Ralph, Daniel, and Stephan Dempe. "Directional derivatives of the solution of a parametric nonlinear program." Mathematical programming 70.1 (1995): 159-172.](https://link.springer.com/content/pdf/10.1007/BF01585934.pdf)

- [Amos, Brandon, and J. Zico Kolter. "Optnet: Differentiable optimization as a layer in neural networks." International Conference on Machine Learning. PMLR, 2017.](https://arxiv.org/pdf/1703.00443.pdf)
