using Dito:
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

@testset "Dito.jl" begin
    x0 = zeros(2)
    horizon = 10
    state_dim = 2
    control_dim = 2
    dynamics = (x, u, t) -> x + 0.01 * u
    inequality_constraints = let
        state_constraints = state -> [state .+ 0.1; -state .+ 0.1]
        control_constraints = control -> [control .+ 0.1; -control .+ 0.1]
        (xs, us) -> [
            mapreduce(state_constraints, vcat, xs)
            mapreduce(control_constraints, vcat, us)
        ]
    end

    function goal_reference_cost(xs, us, params)
        regularization = 10
        sum(zip(xs, us)) do (x, u)
            sum((x[1:2] - params) .^ 2) + regularization * sum(u .^ 2)
        end
    end

    function input_reference_cost(xs, us, params)
        regularization = 10
        ps = reshape(params, 2, :) |> eachcol
        sum(zip(us, ps)) do (u, p)
            sum(0.5 .* regularization .* u .^ 2 .- u .* p)
        end
    end

    for solver in [
        NLPSolver(),
        QPSolver(),
        MCPSolver(),
    ]
        @testset "$solver" begin
            for (cost, parameter_dim) in
                [(goal_reference_cost, 2), (input_reference_cost, 2 * horizon)]
                @testset "$cost" begin
                    optimizer = let
                        problem = ParametricTrajectoryOptimizationProblem(
                            cost,
                            dynamics,
                            inequality_constraints,
                            state_dim,
                            control_dim,
                            parameter_dim,
                            horizon,
                        )
                        Optimizer(problem, solver)
                    end

                    @testset "forward" begin
                        @testset "trivial trajectory qp" begin
                            # In this trivial example, the goal equals the initial position (at the origin).
                            # Thus, we expect the trajectory to be all zeros
                            params = zeros(parameter_dim)
                            xs, us, λs = optimizer(x0, params)
                            @test all(all(isapprox.(x, 0, atol = 1e-9)) for x in xs)
                            @test all(all(isapprox.(u, 0, atol = 1e-9)) for u in us)
                            @test all(>=(-1e-9), λs)
                        end
                    end

                    @testset "ad" begin
                        function objective(params)
                            xs, us, λs = optimizer(x0, params)
                            Zygote.ignore() do
                                @test all(>=(-1e-9), λs)
                            end
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
                                            only(Zygote.gradient(f, zeros(parameter_dim))),
                                            0,
                                            atol = 1e-9,
                                        ),
                                    )
                                end

                                @testset "random" begin
                                    rng = MersenneTwister(0)
                                    for _ in 1:100
                                        @test let
                                            params = 10 * randn(rng, parameter_dim)
                                            ∇ = Zygote.gradient(f, params) |> only
                                            ∇_fd = FiniteDiff.finite_difference_gradient(f, params)
                                            isapprox(∇, ∇_fd; atol = 1e-3)
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
