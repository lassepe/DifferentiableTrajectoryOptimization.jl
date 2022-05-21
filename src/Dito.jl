__precompile__(false)

module Dito

using ChainRulesCore: ChainRulesCore, NoTangent, ProjectTo, @thunk
using Symbolics: Symbolics, @variables, scalarize
using Ipopt: Ipopt
using OSQP: OSQP
using SparseArrays: SparseArrays, findnz, sparse, spzeros
using LinearAlgebra: ColumnNorm, qr, I
using ForwardDiff: ForwardDiff, Dual
using PATHSolver: PATHSolver
using PublicAPI: @public

include("utils.jl")
include("parameterizations.jl")
include("parametric_optimization_problem.jl")

include("qp_solver.jl")
include("nlp_solver.jl")
include("mcp_solver.jl")
include("autodiff.jl")
include("optimizer.jl")

# Public API
@public(
    Optimizer,
    ParametricTrajectoryOptimizationProblem,
    InputReferenceParameterization,
    GoalReferenceParameterization,
    parameter_dimension,
    get_constraints_from_box_bounds,
    QPSolver,
    MCPSolver,
    NLPSolver,
    solve,
    is_thread_safe,
)

end
