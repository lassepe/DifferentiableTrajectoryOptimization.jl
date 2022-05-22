"""
    Optimizer(problem, solver)

Constructs an `Optimizer` for the given `problem` using the specificed `solver` backend.

# Example

```@example running_example
solver = QPSolver()
optimizer = Optimizer(problem, solver)
```
"""
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

# Example

``` @example running_example
x0 = zeros(4)
params = zeros(20)
solution = optimizer(x0, params)
```
"""
function (optimizer::Optimizer)(x0, params)
    @assert length(x0) == optimizer.problem.state_dim
    sol = solve(optimizer.solver, optimizer.problem, x0, params)
    (; horizon, state_dim, control_dim) = optimizer.problem
    xs = [[x0]; collect.(eachcol(reshape(sol.primals[1:(horizon * state_dim)], state_dim, :)))]
    us = collect.(eachcol(reshape(sol.primals[((horizon * state_dim) + 1):end], control_dim, :)))
    (; xs, us, λs = sol.inequality_duals)
end
