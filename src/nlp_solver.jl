"""
Solves the trajectory optimization problem as NLP using Ipopt.

# Note

This solver is mostely here for historic reasons to provide a fully open-source backend for NLPs.
For many problems the [`MCPSolver`](@ref) backend using PATH is *much* faster.
"""
struct NLPSolver end
is_thread_safe(::NLPSolver) = true

function solve(solver::NLPSolver, problem, x0, params::AbstractVector{<:AbstractFloat})
    (;
        horizon,
        n,
        parametric_cost,
        parametric_cost_grad,
        parametric_cons,
        jac_primals,
        lag_hess_primals,
    ) = problem
    (; jac_rows, jac_cols, parametric_jac_vals) = jac_primals
    (; lag_hess_rows, lag_hess_cols, parametric_lag_hess_vals) = lag_hess_primals

    wrapper_cost = function (primals)
        parametric_cost(params, primals)
    end

    wrapper_cons = function (primals, cons)
        parametric_cons(cons, x0, params, primals)
    end

    wrapper_cost_grad = function (primals, grad)
        parametric_cost_grad(grad, params, primals)
        nothing
    end

    wrapper_con_jac = function (primals, rows, cols, values)
        if isnothing(values)
            rows .= jac_rows
            cols .= jac_cols
        else
            parametric_jac_vals(values, x0, params, primals)
        end
        nothing
    end

    wrapper_lag_hess = function (primals, rows, cols, α, λ, values)
        if isnothing(values)
            rows .= lag_hess_rows
            cols .= lag_hess_cols
        else
            parametric_lag_hess_vals(
                values,
                x0,
                params,
                primals,
                λ,
                α,
                # IPOPT has a flipped internal sign convention
                -1.0,
            )
        end
        nothing
    end

    lb = zeros(problem.num_equality + problem.num_inequality)
    ub = fill(Inf, length(lb))
    ub[1:(problem.num_equality)] .= lb[1:(problem.num_equality)]

    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(-Inf, n),
        fill(Inf, n),
        size(lb, 1),
        lb,
        ub,
        size(jac_rows, 1),
        size(lag_hess_rows, 1),
        wrapper_cost,
        wrapper_cons,
        wrapper_cost_grad,
        wrapper_con_jac,
        wrapper_lag_hess,
    )

    xinit = zeros(n)
    let xinit = reshape(xinit, length(x0), :)
        for t in 1:horizon
            xinit[:, t] = x0
        end
    end
    prob.x = xinit

    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    status = Ipopt.IpoptSolve(prob)

    if status != 0
        @warn "MCP not cleanly solved. IPOPT status is $(status)."
    end

    (;
        primals = prob.x,
        equality_duals = -prob.mult_g[1:(problem.num_equality)],
        inequality_duals = -prob.mult_g[(problem.num_equality + 1):end],
    )
end
