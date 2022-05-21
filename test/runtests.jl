using Dito:
    Optimizer,
    GoalReferenceParameterization,
    InputReferenceParameterization,
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
    x0 = zeros(4)
    T = 10
    state_dim = 4
    control_dim = 2
    dynamics = let
        dt = 0.1
        dt2 = dt * dt

        A = [
            1.0 0.0 dt 0.0
            0.0 1.0 0.0 dt
            0.0 0.0 1.0 0.0
            0.0 0.0 0.0 1.0
        ]
        B = [
            dt2 0.0
            0.0 dt2
            dt 0.0
            0.0 dt
        ]

        (x, u, t) -> A * x + B * u
    end
    inequality_constraints = let
        state_constraints = state -> [state .+ 0.1; -state .+ 0.1]
        control_constraints = control -> [control .+ 0.1; -control .+ 0.1]
        (xs, us) -> [
            mapreduce(state_constraints, vcat, xs)
            mapreduce(control_constraints, vcat, us)
        ]
    end

    for solver in [NLPSolver(), QPSolver(), MCPSolver()]
        @testset "$solver" begin
            for (parameterization_name, parameterization) in (
                ("GoalReferenceParameterization", GoalReferenceParameterization(; α = 10.0)),
                ("InputReferenceParameterization", InputReferenceParameterization(; α = 10.0)),
            )
                @testset "$parameterization_name" begin
                    pdim = parameter_dimension(parameterization; T, state_dim, control_dim)
                    optimizer = let
                        problem = ParametricTrajectoryOptimizationProblem(
                            parameterization,
                            dynamics,
                            inequality_constraints,
                            state_dim,
                            control_dim,
                            T,
                        )
                        Optimizer(problem, solver)
                    end

                    @testset "forward" begin
                        @testset "trivial trajectory qp" begin
                            # In this trivial example, the goal equals the initial position (at the origin).
                            # Thus, we expect the trajectory to be all zeros
                            params = zeros(pdim)
                            xs, us, λs = optimizer(x0, params)
                            @test all(all(isapprox.(x, 0, atol = 1e-9)) for x in xs)
                            @test all(all(isapprox.(u, 0, atol = 1e-9)) for u in us)
                            @test all(>=(0), λs)
                        end
                    end

                    @testset "ad" begin
                        function objective(params)
                            xs, us, λs = optimizer(x0, params)
                            Zygote.ignore() do
                                @test all(>=(0), λs)
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
                                            only(Zygote.gradient(f, zeros(pdim))),
                                            0,
                                            atol = 1e-9,
                                        ),
                                    )
                                end

                                @testset "random" begin
                                    rng = MersenneTwister(0)
                                    for _ in 1:100
                                        @test let
                                            params = 10 * randn(rng, pdim)
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
