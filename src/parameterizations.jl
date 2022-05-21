"""
Computes the number of scalar parameters that this parameterization consists of when applied to
a problem of horizon `horizon` and stage-wise dimensions of states, `state_dim`, and controls
`control_dim`.
"""
parameter_dimension(parameterization; horizon::Integer, state_dim::Integer, control_dim::Integer)

"""
Computes the cost (optimization objective) for a given sequence of state `xs` and `us` and parameter
vector `params`
"""
setup_cost(
    parameterization,
    xs::AbstractVector{<:AbstractVector},
    us::AbstractVector{<:AbstractVector},
    params::AbstractVector,
)

#== InputReferenceParameterization ==#

Base.@kwdef struct InputReferenceParameterization
    α::Float64
end

function parameter_dimension(::InputReferenceParameterization; horizon, state_dim, control_dim)
    horizon * control_dim
end

function setup_cost(parameterization::InputReferenceParameterization, xs, us, params)
    horizon = length(us)
    ps = reshape(params, :, horizon) |> eachcol
    sum(zip(Iterators.drop(xs, 1), us, ps)) do (x, u, param)
        sum(0.5 .* parameterization.α .* u .^ 2 .- u .* param)
    end
end

#== GoalReferenceParameterization==#

Base.@kwdef struct GoalReferenceParameterization
    α::Float64
end

function parameter_dimension(::GoalReferenceParameterization; horizon, state_dim, control_dim)
    2
end

function setup_cost(parameterization::GoalReferenceParameterization, xs, us, params)
    sum(zip(Iterators.drop(xs, 1), us)) do (x, u)
        sum((x[1:2] - params) .^ 2) + parameterization.α * sum(u .^ 2)
    end
end
