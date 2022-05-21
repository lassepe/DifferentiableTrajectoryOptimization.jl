# https://github.com/JuliaLang/julia/issues/31231
Base.@kwdef struct ParametricTrajectoryOptimizationProblem{T1,T2,T3,T4,T5,T6,T7,T8,T9}
    parameterization::T1
    T::Int
    n::Int
    state_dim::Int
    control_dim::Int
    num_equality::Int
    num_inequality::Int
    parametric_cost::T2
    parametric_cost_grad::T3
    parametric_cost_jac::T4
    parametric_cons::T5
    jac_primals::T6
    jac_params::T7
    cost_hess::T8
    lag_hess_primals::T9
end

function parameter_dimension(p::ParametricTrajectoryOptimizationProblem)
    (; parameterization, T, state_dim, control_dim) = p
    parameter_dimension(parameterization; state_dim, control_dim, T)
end

"""
Compiles all of the functions, gradients, jacobians, and hessians needed to solve a parametric
trajectory optimization problem (will be parameterized by the initial state of the system,
and a set of additional parameters which have meaning dictated by 'parameterization')
"""
function ParametricTrajectoryOptimizationProblem(
    parameterization,
    dynamics,
    state_dim,
    control_dim,
    T,
    inequality_constraints = (xs, us) -> Symbolics.Num[],
)
    n = T * (state_dim + control_dim)
    num_equality = nx = T * state_dim

    x0, z, p = let
        pdim = parameter_dimension(parameterization; T, state_dim, control_dim)
        @variables(x0[1:state_dim], z[1:n], p[1:pdim]) .|> scalarize
    end

    xs = hcat(x0, reshape(z[1:nx], state_dim, T)) |> eachcol |> collect
    us = reshape(z[(nx + 1):n], control_dim, T) |> eachcol |> collect

    cost = setup_cost(parameterization, xs, us, p)
    cost_grad = Symbolics.gradient(cost, z)
    cost_jac_param = Symbolics.sparsejacobian(cost_grad, p)
    (cost_jac_rows, cost_jac_cols, cost_jac_vals) = findnz(cost_jac_param)

    constraints = Symbolics.Num[]
    # NOTE: The dynamics constraints **must** always be first since the backward pass exploits this
    # structure to more easily identify active constraints.
    for t in eachindex(us)
        append!(constraints, dynamics(xs[t], us[t], t) .- xs[t + 1])
    end
    append!(constraints, inequality_constraints(xs[2:end], us))

    num_inequality = length(constraints) - num_equality

    con_jac = Symbolics.sparsejacobian(constraints, z)
    (jac_rows, jac_cols, jac_vals) = findnz(con_jac)

    con_jac_p = Symbolics.sparsejacobian(constraints, p)
    (jac_p_rows, jac_p_cols, jac_p_vals) = findnz(con_jac_p)

    num_constraints = length(constraints)

    λ, cost_scaling, constraint_penalty_scaling = let
        @variables(λ[1:num_constraints], cost_scaling, constraint_penalty_scaling) .|> scalarize
    end
    lag = cost_scaling * cost + constraint_penalty_scaling - λ' * constraints

    lag_hess = Symbolics.sparsejacobian(Symbolics.gradient(lag, z), z)
    expression = Val{false}
    (lag_hess_rows, lag_hess_cols, hess_vals) = findnz(lag_hess)

    parametric_cost = let
        cost_fn = Symbolics.build_function(cost, [p; z]; expression)
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
        con_fn! = Symbolics.build_function(constraints, [x0; p; z]; expression)[2]
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
        parameterization,
        T,
        n,
        state_dim,
        control_dim,
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
