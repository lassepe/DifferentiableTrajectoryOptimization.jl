using DifferentiableTrajectoryOptimization:
    Optimizer,
    ParametricTrajectoryOptimizationProblem,
    NLPSolver,
    QPSolver,
    MCPSolver,
    parameter_dimension
using Test: @testset, @test
using Zygote: Zygote
using Random: MersenneTwister
using FiniteDiff: FiniteDiff

@testset "DifferentiableTrajectoryOptimization.jl" begin
    δt = 0.01
    x0 = zeros(2)
    horizon = 10
    state_dim = 2
    control_dim = 2
    dynamics = function (x, u, t, params = 0.01)
        local δt = last(params)
        x + δt * u
    end
    inequality_constraints = let
        state_constraints = state -> [state .+ 0.1; -state .+ 0.1]
        control_constraints = control -> [control .+ 0.1; -control .+ 0.1]
        (xs, us, params) -> [
            mapreduce(state_constraints, vcat, xs)
            mapreduce(control_constraints, vcat, us)
        ]
    end

    function goal_reference_cost(xs, us, params)
        goal = params[1:2]
        regularization = 10
        sum(zip(xs, us)) do (x, u)
            sum((x[1:2] - goal) .^ 2) + regularization * sum(u .^ 2)
        end
    end

    function input_reference_cost(xs, us, params)
        regularization = 10
        input_reference = reshape(params[1:(2 * length(us))], 2, :) |> eachcol
        sum(zip(us, input_reference)) do (u, r)
            sum(0.5 .* regularization .* u .^ 2 .- u .* r)
        end
    end

    for solver in [NLPSolver(), QPSolver(), MCPSolver()]
        @testset "$solver" begin
            for (cost, parameter_dim) in
                [(goal_reference_cost, 3), (input_reference_cost, (2 * horizon + 1))]
                trivial_params = [zeros(parameter_dim - 1); δt]

                @testset "$cost" begin
                    optimizer = let
                        problem = ParametricTrajectoryOptimizationProblem(
                            cost,
                            dynamics,
                            inequality_constraints,
                            state_dim,
                            control_dim,
                            parameter_dim,
                            horizon;
                            parameterize_dynamics = true,
                        )
                        Optimizer(problem, solver)
                    end

                    @testset "forward" begin
                        @testset "trivial trajectory qp" begin
                            # In this trivial example, the goal equals the initial position (at the origin).
                            # Thus, we expect the trajectory to be all zeros
                            xs, us, λs = optimizer(x0, trivial_params)
                            @test all(all(isapprox.(x, 0, atol = 1e-3)) for x in xs)
                            @test all(all(isapprox.(u, 0, atol = 1e-3)) for u in us)
                            @test all(>=(-1e-9), λs)
                        end
                    end

                    @testset "ad" begin
                        function objective(params)
                            xs, us, λs = optimizer(x0, params)
                            sum(sum(x .^ 2) for x in xs) + sum(sum(λ .^ 2) for λ in λs)
                        end,
                        for (mode, f) in [
                            ("reverse mode", objective),
                            ("forward mode", params -> Zygote.forwarddiff(objective, params)),
                        ]
                            @testset "$mode" begin
                                @testset "trivial" begin
                                    # The start and goal are the same. Thus, we expect the gradient of the objective
                                    # that penalizes deviation from the origin to be zero.
                                    @test all(
                                        isapprox.(
                                            only(Zygote.gradient(f, trivial_params)),
                                            0,
                                            atol = 1e-9,
                                        ),
                                    )
                                end

                                @testset "random" begin
                                    rng = MersenneTwister(0)
                                    for _ in 1:10
                                        @test let
                                            params = [10 * randn(rng, parameter_dim - 1); δt]
                                            ∇ = Zygote.gradient(f, params) |> only
                                            ∇_fd = FiniteDiff.finite_difference_gradient(f, params)
                                            isapprox(∇, ∇_fd; atol = 1e-3, rtol = 1e-2)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
