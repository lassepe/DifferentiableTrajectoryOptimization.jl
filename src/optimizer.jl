struct Optimizer{TP<:ParametricTrajectoryOptimizationProblem,TS}
    problem::TP
    solver::TS
end

is_thread_safe(optimizer::Optimizer) = is_thread_safe(optimizer.solver)
parameter_dimension(optimizer::Optimizer) = parameter_dimension(optimizer.problem)

"""

    optimizer(x0, params)

Generates an optimal trajectory starting from `x0` according to the optimization problem
parameterized by `params`. This call is differentaible in `params`.
"""
function (optimizer::Optimizer)(x0, params)
    sol = solve(optimizer.solver, optimizer.problem, x0, params)
    (; T, state_dim, control_dim) = optimizer.problem
    xs = [[x0]; collect.(eachcol(reshape(sol.primals[1:(T * state_dim)], state_dim, :)))]
    us = collect.(eachcol(reshape(sol.primals[((T * state_dim) + 1):end], control_dim, :)))
    (; xs, us, λs = sol.inequality_duals)
end
