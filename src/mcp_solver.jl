struct MCPSolver end
is_thread_safe(::MCPSolver) = false

"""
Solves the (potentially nonlinear and non-convex) trajectory optimization problem by casting the KKT
system as as a mixed complementarity problem (MCP).
"""
function solve(solver::MCPSolver, problem, x0, params::AbstractVector{<:AbstractFloat})
    (; n, parametric_cost, parametric_cost_grad, parametric_cons, jac_primals, lag_hess_primals) =
        problem
    (; jac_rows, jac_cols, parametric_jac_vals) = jac_primals
    (; lag_hess_rows, lag_hess_cols, parametric_lag_hess_vals) = lag_hess_primals

    function F(n, z, f)
        primals = z[1:(problem.n)]
        duals = z[(problem.n + 1):end]

        ∇l = zeros(problem.n)
        parametric_cost_grad(∇l, params, primals)

        ∇g = let
            jac_vals = zeros(length(jac_rows))
            parametric_jac_vals(jac_vals, x0, params, primals)
            sparse(jac_rows, jac_cols, jac_vals)
        end

        f[1:(problem.n)] .= ∇l - ∇g' * duals
        f[(problem.n + 1):end] .= let
            g = zeros(problem.num_equality + problem.num_inequality)
            parametric_cons(g, x0, params, primals)
            g
        end
        Cint(0)
    end

    """
    J = [
        Q  -A'
        A  0
    ]

    nnz: number of non-zeros of the sparse J = nnz(Q) + 2nnz(A)
    z:   [primals; duals]
    (col, len, row, data): coo format sparse array representation
    """
    function J(n, nnz, z, col, len, row, data)
        primals = z[1:(problem.n)]
        duals = z[(problem.n + 1):end]
        # Hessian of the Lagrangian
        Q = let
            lag_hess_vals = zeros(length(lag_hess_rows))
            parametric_lag_hess_vals(lag_hess_vals, x0, params, primals, duals, 1, 1)
            sparse(lag_hess_rows, lag_hess_cols, lag_hess_vals)
        end

        # Jacobian of the constraints
        A = let
            jac_vals = zeros(length(jac_rows))
            parametric_jac_vals(jac_vals, x0, params, primals)
            sparse(jac_rows, jac_cols, jac_vals)
        end

        J = [
            Q -A'
            A 0I
        ]

        coo_from_sparse!(col, len, row, data, J)
        Cint(0)
    end

    lb = [
        fill(-Inf, problem.n + problem.num_equality)
        zeros(problem.num_inequality)
    ]
    ub = fill(Inf, length(lb))
    # TODO: initial guess
    z = zero(lb)
    # structual zeros: nnz(J)) = nnz(Q) + 2*nnz(A)
    nnz = length(lag_hess_rows) + 2 * length(jac_rows)

    """
    PATH solves an MCP of the form

    find   z
    s.t.   lᵢ == zᵢ       Fᵢ(z) >= 0
           lᵢ <  zᵢ <  u, Fᵢ(z) == 0
                 zᵢ == u, Fᵢ(z) <= 0

    Here, `J` is the functional representation of the jacobian to `F`.
    """
    status, variables, info = PATHSolver.solve_mcp(F, J, lb, ub, z; silent = true, nnz)

    (;
        primals = variables[1:(problem.n)],
        equality_duals = variables[((problem.n + 1):(problem.n + problem.num_equality))],
        inequality_duals = variables[(problem.n + problem.num_equality + 1):end],
    )
end

"""
Convert a Julia sparse array `M` into the \
[COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) format required by PATH.

This implementation has been extracted from \
[here](https://github.com/chkwon/PATHSolver.jl/blob/8e63723e51833cdbab58c39b6646f8cdf79d74a2/src/C_API.jl#L646)
"""
function coo_from_sparse!(col, len, row, data, M)
    @assert length(col) == length(len) == size(M, 1)
    @assert length(row) == length(data)
    n = length(col)
    for i in 1:n
        col[i] = M.colptr[i]
        len[i] = M.colptr[i + 1] - M.colptr[i]
    end
    for (i, v) in enumerate(SparseArrays.rowvals(M))
        row[i] = v
    end
    for (i, v) in enumerate(SparseArrays.nonzeros(M))
        data[i] = v
    end
end
