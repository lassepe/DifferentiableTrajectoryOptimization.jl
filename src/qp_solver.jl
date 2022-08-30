"""
A solver backend that treats the problem as a quadratic program (QP)

    QP(y) := argmin_x 0.5 x'Qx + x'(Ry+q)
             s.t.  lb <= Ax + By <= ub

# Note

Here, the problem data tuple `(Q, R, q A, B, lb, ub)` is derived from the provided
`ParametricTrajectoryOptimizationProblem` via linearization of constraints and quadraticization of
the objective. Therefore, if the problem is not a QP then this solution is not exact!

"""
struct QPSolver end
is_thread_safe(::QPSolver) = true

"""
Solves quadratic program:
QP(y) := argmin_x 0.5 x'Qx + x'(Ry+q)
         s.t.  lb <= Ax + By <= ub
Additionally provides gradients ∇_y QP(y)

Q, R, A, and B should be sparse matrices of type SparseMatrixCSC.
q, a, and y should be of type Vector{Float64}.
"""
function solve(::QPSolver, problem, x0, params::AbstractVector{<:AbstractFloat}; initial_guess = nothing)
    (; cost_hess_rows, cost_hess_cols, parametric_cost_hess_vals) = problem.cost_hess
    (; jac_rows, jac_cols, parametric_jac_vals) = problem.jac_primals

    n = problem.n
    m = problem.num_equality + problem.num_inequality

    primals = zeros(n)
    duals = zeros(m)

    Qvals = zeros(size(cost_hess_rows, 1))
    parametric_cost_hess_vals(Qvals, params, primals)
    Q = sparse(cost_hess_rows, cost_hess_cols, Qvals, n, n)

    q = zeros(n)
    problem.parametric_cost_grad(q, params, primals)

    Avals = zeros(size(jac_rows, 1))
    parametric_jac_vals(Avals, x0, params, primals)
    A = sparse(jac_rows, jac_cols, Avals, m, n)

    cons = zeros(m)
    problem.parametric_cons(cons, x0, params, primals)

    lb = -cons
    ub = fill(Inf, length(lb))
    ub[1:(problem.num_equality)] .= lb[1:(problem.num_equality)]

    m = OSQP.Model()
    OSQP.setup!(m; P = sparse(Q), q = q, A = A, l = lb, u = ub, verbose = false, polish = true)
    
    if !isnothing(initial_guess)
        OSQP.warm_start!(m; x = initial_guess.x, y = initial_guess.y)
    end

    results = OSQP.solve!(m)
    if (results.info.status_val != 1)
        @warn "QP not cleanly solved. OSQP status is $(results.info.status_val)"
    end

    (;
        primals = results.x,
        equality_duals = -results.y[1:(problem.num_equality)],
        inequality_duals = -results.y[(problem.num_equality + 1):end],
        info = (; raw_solution = results),
    )
end
