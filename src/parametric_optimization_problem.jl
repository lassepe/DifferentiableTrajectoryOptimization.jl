"""
    ParametricTrajectoryOptimizationProblem(
        cost,
        dynamics,
        inequality_constraints,
        state_dim,
        control_dim,
        parameter_dim,
        horizon,
    )

Constructs a `ParametricTrajectoryOptimizationProblem` from the given problem data:

- `cost` is callable as `cost(xs, us, params) -> c` to compute objective value for a given \
sequence of states `xs` and control inputs `us` for a parameter vector `params`.

- `dynamics` is callable as `dynamics(x, u, t [, params]) -> xp` to generate the next state `xp` \
from the previous state `x`, control `u`, time `t` and optional parameters `params`.  See \
`parameterize_dynamics` for toggling the optional parameter vector.

- `inequality_constraints` is callable as `inequality_constraints(xs, us, params) -> gs` to \
generate a vector of constraints `gs` from states `xs` and `us` where the layout and types of `xs` \
and `us` are the same as for the `cost`. Constraints specified in this form will be enforced as \
`0 <= gs`; i.e., feasible trajectories evalute to non-negative constraints. If your prolbem has \
no inequality constraints, set `inequality_constraints = (xs, us, params) -> Symbolics.Num[]`.

- `state_dim::Integer` is the stagewise dimension of the state.

- `control_dim::Integer` is the stagewise dimension of the control input.

- `horizon::Integer` is the horizon of the problem

- `parameterize_dynamics` controls the optional `params` argument handed to dynamics. This flag is \
disabled by default. When set to `true`, `dynamics` are called as `dynamics(x, u, t, params)`
instead of `dynamics(x, u, t)`. Note that *all* parameters are handed to the dynamics call

# Note

This function uses `Syombolics.jl` to compile all of the functions, gradients, jacobians, and
hessians needed to solve a parametric trajectory optimization problem. Therfore, all callables above
must be sufficiently generic to accept `Syombolics.Num`-valued arguments.

Since the setup procedure involves code-generation, calls to this contructor are rather expensive
and shold be avoided in tight inner loops. By contrast, repeated solver invokations on the same
`ParametricTrajectoryOptimizationProblem` for varying parameter values are very fast. Therefore, it
is a good idea to choose a parameterization that avoids re-construction.

Furthermore, note that the *entire* parameter vector is handed to `costs`, `dyanmics`, and
`inequality_constraints`. This allows parameters to be shared between multiple calls. For example,
a parameter that controlls the collision avoidance radius may apear both in the cost and
constraints. It's the users responsibility to correctly index into the `params` vector to extract
the desired parameters for each call.

# Example

Below we construct a parametric optimization problem for a 2D integrator with 2 states, 2 inputs
over a hrizon of 10 stages.
Additionally, this problem features ±0.1 box constraints on states and inputs.

```@example running_example
horizon = 10
state_dim = 2
control_dim = 2
cost = (xs, us, params) -> sum(sum((x - params).^2) + sum(u.^2) for (x, u) in zip(xs, us))
dynamics = (x, u, t) -> x + u
inequality_constraints = let
    state_constraints = state -> [state .+ 0.1; -state .+ 0.1]
    control_constraints = control -> [control .+ 0.1; -control .+ 0.1]
    (xs, us, params) -> [
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
"""
Base.@kwdef struct ParametricTrajectoryOptimizationProblem{T1,T2,T3,T4,T5,T6,T7,T8}
    # https://github.com/JuliaLang/julia/issues/31231
    horizon::Int
    n::Int
    state_dim::Int
    control_dim::Int
    parameter_dim::Int
    num_equality::Int
    num_inequality::Int
    parametric_cost::T1
    parametric_cost_grad::T2
    parametric_cost_jac::T3
    parametric_cons::T4
    jac_primals::T5
    jac_params::T6
    cost_hess::T7
    lag_hess_primals::T8
end

function ParametricTrajectoryOptimizationProblem(
    cost,
    dynamics,
    inequality_constraints,
    state_dim,
    control_dim,
    parameter_dim,
    horizon;
    parameterize_dynamics = false
)
    n = horizon * (state_dim + control_dim)
    num_equality = nx = horizon * state_dim

    x0, z, p = let
        @variables(x0[1:state_dim], z[1:n], p[1:parameter_dim]) .|> scalarize
    end

    xs = hcat(x0, reshape(z[1:nx], state_dim, horizon)) |> eachcol |> collect
    us = reshape(z[(nx + 1):n], control_dim, horizon) |> eachcol |> collect

    cost_val = cost(xs[2:end], us, p)
    cost_grad = Symbolics.gradient(cost_val, z)
    cost_jac_param = Symbolics.sparsejacobian(cost_grad, p)
    (cost_jac_rows, cost_jac_cols, cost_jac_vals) = findnz(cost_jac_param)

    constraints_val = Symbolics.Num[]
    # NOTE: The dynamics constraints **must** always be first since the backward pass exploits this
    # structure to more easily identify active constraints.
    dynamic_extra_args = parameterize_dynamics ? tuple(p) : tuple()
    for t in eachindex(us)
        append!(constraints_val, dynamics(xs[t], us[t], t) .- xs[t + 1], dynamic_extra_args...)
    end
    append!(constraints_val, inequality_constraints(xs[2:end], us, p))

    num_inequality = length(constraints_val) - num_equality

    con_jac = Symbolics.sparsejacobian(constraints_val, z)
    (jac_rows, jac_cols, jac_vals) = findnz(con_jac)

    con_jac_p = Symbolics.sparsejacobian(constraints_val, p)
    (jac_p_rows, jac_p_cols, jac_p_vals) = findnz(con_jac_p)

    num_constraints = length(constraints_val)

    λ, cost_scaling, constraint_penalty_scaling = let
        @variables(λ[1:num_constraints], cost_scaling, constraint_penalty_scaling) .|> scalarize
    end
    lag = cost_scaling * cost_val - constraint_penalty_scaling * λ' * constraints_val

    lag_hess = Symbolics.sparsejacobian(Symbolics.gradient(lag, z), z)
    expression = Val{false}
    (lag_hess_rows, lag_hess_cols, hess_vals) = findnz(lag_hess)

    parametric_cost = let
        cost_fn = Symbolics.build_function(cost_val, [p; z]; expression)
        (params, primals) -> cost_fn(vcat(params, primals))
    end

    parametric_cost_grad = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [p; z]; expression)[2]
        (grad, params, primals) -> cost_grad_fn!(grad, vcat(params, primals))
    end

    cost_hess = let
        cost_hess_sym = Symbolics.sparsejacobian(cost_grad, z)
        (cost_hess_rows, cost_hess_cols, cost_hess_vals) = findnz(cost_hess_sym)
        cost_hess_fn! = Symbolics.build_function(cost_hess_vals, [p; z]; expression)[2]
        parametric_cost_hess_vals =
            (hess, params, primals) -> cost_hess_fn!(hess, vcat(params, primals))
        (; cost_hess_rows, cost_hess_cols, parametric_cost_hess_vals)
    end

    parametric_cost_jac_vals = let
        cost_jac_param_fn! = Symbolics.build_function(cost_jac_vals, [p; z]; expression)[2]
        (vals, params, primals) -> cost_jac_param_fn!(vals, vcat(params, primals))
    end

    parametric_cons = let
        con_fn! = Symbolics.build_function(constraints_val, [x0; p; z]; expression)[2]
        (cons, x0, params, primals) -> con_fn!(cons, vcat(x0, params, primals))
    end

    parametric_jac_vals = let
        jac_vals_fn! = Symbolics.build_function(jac_vals, [x0; p; z]; expression)[2]
        (vals, x0, params, primals) -> jac_vals_fn!(vals, vcat(x0, params, primals))
    end

    parametric_jac_p_vals = let
        jac_p_vals_fn! = Symbolics.build_function(jac_p_vals, [x0; p; z]; expression)[2]
        (vals, x0, params, primals) -> jac_p_vals_fn!(vals, vcat(x0, params, primals))
    end

    parametric_lag_hess_vals = let
        hess_vals_fn! = Symbolics.build_function(
            hess_vals,
            [x0; p; z; λ; cost_scaling; constraint_penalty_scaling];
            expression,
        )[2]
        (vals, x0, params, primals, duals, cost_scaling, constraint_penalty_scaling) ->
            hess_vals_fn!(
                vals,
                vcat(x0, params, primals, duals, cost_scaling, constraint_penalty_scaling),
            )
    end

    parametric_cost_jac = (; cost_jac_rows, cost_jac_cols, parametric_cost_jac_vals)
    jac_primals = (; jac_rows, jac_cols, parametric_jac_vals)
    jac_params = (; jac_p_rows, jac_p_cols, parametric_jac_p_vals)
    lag_hess_primals = (; lag_hess_rows, lag_hess_cols, parametric_lag_hess_vals)

    ParametricTrajectoryOptimizationProblem(;
        horizon,
        n,
        state_dim,
        control_dim,
        parameter_dim,
        num_equality,
        num_inequality,
        parametric_cost,
        parametric_cost_grad,
        parametric_cost_jac,
        parametric_cons,
        jac_primals,
        jac_params,
        cost_hess,
        lag_hess_primals,
    )
end

function parameter_dimension(problem::ParametricTrajectoryOptimizationProblem)
    problem.parameter_dim
end
