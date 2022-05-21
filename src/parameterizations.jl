#== InputReferenceParameterization ==#

Base.@kwdef struct InputReferenceParameterization
    α::Float64
end

function parameter_dimension(::InputReferenceParameterization; T, state_dim, control_dim)
    T * control_dim
end

function setup_cost(p::InputReferenceParameterization, xs, us, params)
    T = length(us)
    ps = reshape(params, :, T) |> eachcol
    sum(zip(Iterators.drop(xs, 1), us, ps)) do (x, u, param)
        sum(0.5 .* p.α .* u .^ 2 .- u .* param)
    end
end

#== GoalReferenceParameterization==#

Base.@kwdef struct GoalReferenceParameterization
    α::Float64
end

function parameter_dimension(::GoalReferenceParameterization; T, state_dim, control_dim)
    2
end

function setup_cost(p::GoalReferenceParameterization, xs, us, params)
    sum(zip(Iterators.drop(xs, 1), us)) do (x, u)
        sum((x[1:2] - params) .^ 2) + p.α * sum(u .^ 2)
    end
end
