using FSimZoo: LiftCruiseVTOL2D, angle2rotmatrix2d
using DifferentialEquations
using FlightSims
using ComponentArrays
using Plots
using FSimPlots
using LaTeXStrings
using ForwardDiff
using JuMP, Ipopt
using Interpolations
using LinearAlgebra
using JLD2, FileIO
using Dates
using Printf
using ProgressMeter
using DiffResults


function solve_opt(plant::LiftCruiseVTOL2D, ṗ, p̈_ref)
    (; m, g, δ_e_min, δ_e_max, θ_min, θ_max) = plant
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "print_level", 0)
    @variable(model, 0 <= T_r)
    @variable(model, 0 <= T_p)
    @variable(model, δ_e_min <= δ_e <= δ_e_max)
    @variable(model, θ_min <= θ <= θ_max)
    @objective(model, Min, T_r+T_p)
    R = FSimZoo.angle2rotmatrix2d(θ)
    F_des = R * m*(p̈_ref - g*[0, 1])
    L, D1, D2 = FSimZoo.aerodynamic_forces(plant, θ, ṗ; δ_e)
    @constraint(model, [T_p, T_r] == [1 0; 0 -1] * (F_des + [D1, L-D2]))
    optimize!(model)
    T_r = value(T_r)
    T_p = value(T_p)
    δ_e = value(δ_e)
    θ = value(θ)
    return ComponentArray(; T_r, T_p, δ_e, θ)
end


struct TVOptSimplified2DVTOL <: AbstractEnv
    plant::LiftCruiseVTOL2D
end


function FSimBase.State(env::TVOptSimplified2DVTOL)
    return function (p=zeros(2), θ=0, ṗ=zeros(2), θ̇=0, θ_ref=0, U=zeros(2), θ̇_ref_est=zeros(1))
        return ComponentArray(; p, θ, ṗ, θ̇, θ_ref, U, θ̇_ref_est)
    end
end


function Φ(
    U::AbstractVector, p::AbstractVector, p_ref::AbstractVector, ṗ::AbstractVector, ṗ_ref::AbstractVector, p̈_ref::AbstractVector;
    plant::LiftCruiseVTOL2D,
    eps::Float64=1e-3,
)
    (; m, g, δ_e_min, δ_e_max, θ_min, θ_max) = plant
    θ_ref, δ_e = U
    L_ref, D1_ref, D2_ref = FSimZoo.aerodynamic_forces(plant, θ_ref, ṗ; δ_e)
    η = desired_acceleration(p, p_ref, ṗ, ṗ_ref, p̈_ref)
    R = FSimZoo.angle2rotmatrix2d(θ_ref)
    F_des = R * m*(η - g*[0, 1])

    T_p = F_des[1] + D1_ref
    T_r = -F_des[2] - (L_ref - D2_ref)
    J_0 = T_r + T_p
    J_aug = J_0
    J_aug = J_aug - (1/1e3)*log(T_r+eps)
    J_aug = J_aug - (1/1e3)*log(T_p+eps)
    J_aug = J_aug - (1/1e3)*log(θ_ref - θ_min+eps)
    J_aug = J_aug - (1/1e3)*log(θ_max - θ_ref+eps)
    J_aug = J_aug - (1/1e3)*log(δ_e - δ_e_min+eps)
    J_aug = J_aug - (1/1e3)*log(δ_e_max - δ_e+eps)
    return J_aug
end


function update_law(U::AbstractVector, p::AbstractVector, p_ref::AbstractVector, ṗ::AbstractVector, ṗ_ref::AbstractVector, p̈::AbstractVector, p̈_ref::AbstractVector, p⃛_ref::AbstractVector;
                    plant::LiftCruiseVTOL2D,
                    P::AbstractMatrix=diagm([1, 1]),
                    )
    ∇Φ_t = U -> (
        ForwardDiff.gradient(p -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), p)' * ṗ
        + ForwardDiff.gradient(p_ref -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), p_ref)' * ṗ_ref
        + ForwardDiff.gradient(ṗ -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), ṗ)' * p̈
        + ForwardDiff.gradient(ṗ_ref -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), ṗ_ref)' * p̈_ref
        + ForwardDiff.gradient(p̈_ref -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), p̈_ref)' * p⃛_ref
    )
    ∇Φ_t_U = ForwardDiff.gradient(∇Φ_t, U)
    # ∇Φ_U = U -> ForwardDiff.gradient(U -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), U)
    # ∇Φ_U_val = ∇Φ_U(U)
    # ∇Φ_U_U = ForwardDiff.jacobian(∇Φ_U, U)
    result = DiffResults.HessianResult(U)
    result = ForwardDiff.hessian!(result, U -> Φ(U, p, p_ref, ṗ, ṗ_ref, p̈_ref; plant), U)
    ∇Φ_U_val = DiffResults.gradient(result)
    ∇Φ_U_U = DiffResults.hessian(result)  # quite faster
    # U_dot = -inv(∇Φ_U_U) * (P*∇Φ_U_val + ∇Φ_t_U)
    U_dot = -∇Φ_U_U \ (P*∇Φ_U_val + ∇Φ_t_U)  # a little faster
    return U_dot
end


function desired_acceleration(p::AbstractVector, p_ref::AbstractVector, ṗ::AbstractVector, ṗ_ref::AbstractVector, p̈_ref::AbstractVector; k_P=20, k_D=20)
    η = p̈_ref - k_D*(ṗ - ṗ_ref) - k_P*(p - p_ref)
    return η
end


function desired_angular_acceleration(θ::Float64, θ_ref::Float64, θ̇::Float64, θ̇_ref::Float64, θ̈_ref::Float64; k_P=10, k_D=10)
    return θ̈_ref - k_D*(θ̇ - θ̇_ref) - k_P*(θ - θ_ref)
end


function FSimBase.Dynamics!(env::TVOptSimplified2DVTOL)
    (; plant) = env
    (; m, g, J, CL_0, CL_θ, CL_δ_e, CD1_0, CD1_k, CD2_0) = plant
    @Loggable function dynamics!(dX, X, param, t; p_ref::AbstractVector, ṗ_ref::AbstractVector, p̈_ref::AbstractVector, p⃛_ref::AbstractVector, τ::Float64=1e-2)
        e_2 = [0, 1]
        (; p, θ, ṗ, θ̇, U, θ̇_ref_est) = X
        @log p
        @log ṗ
        @log θ
        @log θ̇
        R = FSimZoo.angle2rotmatrix2d(θ)
        θ_ref, δ_e_cmd = U
        @log δ_e_cmd
        δ_e = δ_e_cmd
        # this is faster than min max; not applied for tvopt as it considers constraints
        # if plant.δ_e_max < δ_e
        #     δ_e = plant.δ_e_max
        # end
        # if plant.δ_e_min > δ_e
        #     δ_e = plant.δ_e_min
        # end
        @log δ_e
        # L_ref, D1_ref, D2_ref = FSimZoo.aerodynamic_forces(plant, θ_ref, ṗ; δ_e)
        L, D1, D2 = FSimZoo.aerodynamic_forces(plant, θ, ṗ; δ_e)
        # applied force
        # η = desired_acceleration(p, p_ref, ṗ, ṗ_ref, p̈_ref)
        # T_p_cmd, T_r_cmd = [1 0; 0 -1]*([0, -m*g] + η + [D1_ref, L_ref-D2_ref])
        η = desired_acceleration(p, p_ref, ṗ, ṗ_ref, p̈_ref)
        R = FSimZoo.angle2rotmatrix2d(θ)
        F_des = R * m*(η - g*[0, 1])
        T_p_cmd, T_r_cmd = [1 0; 0 -1] * (F_des + [D1, L-D2])
        @log T_r_cmd
        @log T_p_cmd
        T_r = T_r_cmd
        T_p = T_p_cmd
        # this is faster than min max; not applied for tvopt as it considers constraints
        # if T_r_cmd < 0
        #     T_r = 0
        # end
        # if T_p_cmd < 0
        #     T_p = 0
        # end
        @log T_r
        @log T_p
        F = [
            T_p-D1,
            -(T_r+L-D2),
        ]
        p̈ = (1/m) * (R'*F + m*g*e_2)
        @log p̈
        U_dot = update_law(U, p, p_ref, ṗ, ṗ_ref, p̈, p̈_ref, p⃛_ref; plant)
        θ̇_ref = U_dot[1]
        θ̃ = θ - θ_ref
        θ̇̃ = θ̇ - U_dot[1]
        θ̈_ref_est = (θ̇_ref - θ̇_ref_est[1]) / τ
        M = J * desired_angular_acceleration(θ, θ_ref, θ̇, θ̇_ref, θ̈_ref_est)

        dX.p .= ṗ
        dX.θ = θ̇
        dX.ṗ .= p̈
        dX.θ̇ = M/J
        dX.U .= U_dot
        dX.θ̇_ref_est .= θ̈_ref_est
        @log M
        @log θ_ref
    end
end


function solve_opt_feedback(env::TVOptSimplified2DVTOL, p, p_ref, ṗ, ṗ_ref, p̈_ref; mode=:eqconst)
    (; plant) = env
    (; m, g, δ_e_min, δ_e_max, θ_min, θ_max) = plant
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    if mode == :eqconst
        @variable(model, T_r >= 0)
        @variable(model, T_p >= 0)
        @variable(model, δ_e_min <= δ_e <= δ_e_max)
        @variable(model, θ_min <= θ <= θ_max)
        @objective(model, Min, T_r+T_p)
        R = FSimZoo.angle2rotmatrix2d(θ)
        η = desired_acceleration(p, p_ref, ṗ, ṗ_ref, p̈_ref)
        F_des = R * m*(η - g*[0, 1])
        L, D1, D2 = FSimZoo.aerodynamic_forces(plant, θ, ṗ; δ_e)
        @constraint(model, [T_p, T_r] == [1 0; 0 -1] * (F_des + [D1, L-D2]))
        optimize!(model)
    elseif mode == :substitution
        @variable(model, δ_e_min <= δ_e <= δ_e_max)
        @variable(model, θ_min <= θ <= θ_max)
        R = FSimZoo.angle2rotmatrix2d(θ)
        F_des = R * m*(η - g*[0, 1])
        L, D1, D2 = FSimZoo.aerodynamic_forces(plant, θ, ṗ; δ_e)
        T_p, T_r = [1 0; 0 -1] * (F_des + [D1, L-D2])
        @constraint(model, T_r >= 0)
        @constraint(model, T_p >= 0)
        @objective(model, Min, T_r+T_p)
        optimize!(model)
    else
        error("Invalid mode: $(mode)")
    end
    T_r = value(T_r)
    T_p = value(T_p)
    δ_e = value(δ_e)
    θ = value(θ)
    return ComponentArray(; T_r, T_p, δ_e, θ)
end


function bezier(t::Number; t0::Float64, tf::Float64, Ps::AbstractVector)
    _t = (t-t0)/(tf-t0)
    output = 0.0
    n = length(Ps)-1
    for (i, P_i) in enumerate(Ps)
        output = output + binomial(n, i-1)*(1-_t)^(n-(i-1))*_t^(i-1)*P_i
    end
    return output
end


function generate_reference_position_trajectory(scenario)
     if scenario == :h2c
        p_ref = function (t)
            tf1 = 50.0
            tf2 = 100.0
            tf3 = 125.0
            if t <= tf1
                x_ref =  bezier(t; t0=0.0, tf=tf1, Ps=zeros(10))
                z_ref =  bezier(t; t0=0.0, tf=tf1, Ps=[zeros(5)..., -50*ones(5)...])
            elseif t <= tf2
                x_ref =  bezier(t; t0=tf1, tf=tf2, Ps=[zeros(5)..., 18.0556, 54.1667, 90.2778, 126.3889, 162.5000])
                z_ref =  bezier(t; t0=tf1, tf=tf2, Ps=-50*ones(10))
            elseif t <= tf3
                x_ref =  bezier(t; t0=tf2, tf=tf3, Ps=[162.5000, 180.5556, 198.6111, 216.6667, 234.7222, 252.7778, 270.8333, 288.8889, 306.9444, 325.0000])
                z_ref =  bezier(t; t0=tf2, tf=tf3, Ps=-50*ones(10))
            end
            return [x_ref, z_ref]
        end
    elseif scenario == :c2h
        p_ref = function (t)
            tf1 = 25.0
            tf2 = 75.0
            tf3 = 125.0
            if t <= tf1
                x_ref =  bezier(t; t0=0.0, tf=tf1, Ps=[-325.0000, -306.9444, -288.8889, -270.8333, -252.7778, -234.7222, -216.6667, -198.6111, -180.5556, -162.5000])
                z_ref =  bezier(t; t0=0.0, tf=tf1, Ps=-50*ones(10))
            elseif t <= tf2
                x_ref =  bezier(t; t0=tf1, tf=tf2, Ps=[-162.5000, -126.3889, -90.2778, -54.1667, -18.0556, zeros(5)...])
                z_ref =  bezier(t; t0=tf1, tf=tf2, Ps=-50*ones(10))
            elseif t <= tf3
                x_ref =  bezier(t; t0=tf2, tf=tf3, Ps=zeros(10))
                z_ref =  bezier(t; t0=tf2, tf=tf3, Ps=[-50*ones(5)..., zeros(5)...])
            end
            return [x_ref, z_ref]
        end
    end
end


function sim_opt(
    plant,
    p_ref, ṗ_ref, p̈_ref;
    t0, tf, savestep, _ode_solver,
    T_s, interp, dt,
)
    println("Interpolation alg : $(interp)")
    println("Time step for optimization: $(T_s)")
    ts_cmd = 0.0:T_s:tf
    time_opt = @elapsed cmds = [solve_opt(plant, ṗ_ref(t), p̈_ref(t)) for t in ts_cmd]
    @show time_opt
    function eqconst_violation(plant, T_r, T_p, δ_e, θ, ṗ, p̈_ref)
        (; m, g) = plant
        R = FSimZoo.angle2rotmatrix2d(θ)
        F_des = R * m*(p̈_ref - g*[0, 1])
        L, D1, D2 = FSimZoo.aerodynamic_forces(plant, θ, ṗ; δ_e)
        return norm([T_p, T_r] - [1 0; 0 -1] * (F_des + [D1, L-D2]))
    end

    if interp == :linear
        interpolation = Interpolations.linear_interpolation
    elseif interp == :cubic
        interpolation = Interpolations.cubic_spline_interpolation
    end
    T_r_itp = interpolation(ts_cmd, [cmd.T_r for cmd in cmds], extrapolation_bc = Line())
    T_p_itp = interpolation(ts_cmd, [cmd.T_p for cmd in cmds], extrapolation_bc = Line())
    δ_e_itp = interpolation(ts_cmd, [cmd.δ_e for cmd in cmds], extrapolation_bc = Line())
    θ_ref = interpolation(ts_cmd, [cmd.θ for cmd in cmds], extrapolation_bc = Line())
    θ̇_ref = t -> ForwardDiff.derivative(θ_ref, t)
    θ̈_ref = t -> ForwardDiff.derivative(θ̇_ref, t)

    violations = [eqconst_violation(plant, T_r_itp(t), T_p_itp(t), δ_e_itp(t), θ_ref(t), ṗ_ref(t), p̈_ref(t)) for t in 0:0.001:tf]
    @show maximum(violations)

    controller = function (X, param, t)
        (; p, θ, ṗ, θ̇) = X
        (; m, g) = plant
        δ_e = δ_e_itp(t)
        L, D1, D2 = FSimZoo.aerodynamic_forces(plant, θ, ṗ; δ_e)
        η = desired_acceleration(p, p_ref(t), ṗ, ṗ_ref(t), p̈_ref(t))
        R = FSimZoo.angle2rotmatrix2d(θ)
        F_des = R * m*(η - g*[0, 1])
        T_p, T_r = [1 0; 0 -1] * (F_des + [D1, L-D2])
        # torque
        M = plant.J * desired_angular_acceleration(θ, θ_ref(t), θ̇, θ̇_ref(t), θ̈_ref(t))
        u = ComponentArray(; T_r, T_p, M, δ_e)
        return u
    end
    X0 = State(plant)(p_ref(t0), 0.0, ṗ_ref(t0), 0.0)
    
    simulator = Simulator(
        X0, apply_inputs(Dynamics!(plant); u=controller); solver=_ode_solver, t0, tf, dt,
    )
    time_ode = @elapsed df = solve(simulator; savestep, dt, adaptive=false,)
    @show time_ode
    ts = df.time
    df[!, :p_ref] = [p_ref(t) for t in ts]
    df[!, :ṗ_ref] = [ṗ_ref(t) for t in ts]
    df[!, :p̈_ref] = [p̈_ref(t) for t in ts]
    df[!, :θ_ref] = [θ_ref(t) for t in ts]
    elapsed_times = Dict("time_opt" => time_opt, "time_ode" => time_ode)
    return df, elapsed_times
end


function sim_tvopt(
    plant,
    p_ref, ṗ_ref, p̈_ref, p⃛_ref;
    t0, tf, savestep, _ode_solver, dt,
)
    env = TVOptSimplified2DVTOL(plant)
    p0 = p_ref(0.0)
    ṗ0 = ṗ_ref(0.0)
    initial_guess = solve_opt_feedback(env, p0, p_ref(0.0), ṗ0, ṗ_ref(0.0), p̈_ref(0.0))
    U0 = [initial_guess.θ, initial_guess.δ_e]
    X0 = State(env)(p0, 0.0, ṗ0, 0.0, U0)
    simulator = Simulator(X0,
                          apply_inputs(Dynamics!(env);
                                       p_ref=(X, p, t) -> p_ref(t),
                                       ṗ_ref=(X, p, t) -> ṗ_ref(t),
                                       p̈_ref=(X, p, t) -> p̈_ref(t),
                                       p⃛_ref=(X, p, t) -> p⃛_ref(t),
                                       ); solver=_ode_solver, t0, tf, dt)
    time_ode = @elapsed df = solve(simulator; savestep, dt, adaptive=false,)
    @show time_ode
    ts = df.time
    df[!, :p_ref] = [p_ref(t) for t in ts]
    df[!, :ṗ_ref] = [ṗ_ref(t) for t in ts]
    df[!, :p̈_ref] = [p̈_ref(t) for t in ts]
    df[!, :θ_ref] = [datum.θ_ref for datum in df.sol]
    elapsed_times = Dict("time_ode" => time_ode)
    return df, elapsed_times
end


"""
# Arguments
## Scenario
:h2c: hover-to-cruise
:c2h: cruise-to-hover
"""
function sim(
    method, scenario;
    ode_solver=:tsit5,
    t0=0.0, tf=125.0,
    savestep=0.1,
    T_s=0.1,
    interp=:linear,
    dt=0.01,
)
    println("ODE Solver: $(ode_solver)")
    if ode_solver == :euler  # recommended for normal scenario
        _ode_solver = Euler()
    elseif ode_solver == :bs3  # recommended for normal scenario
        _ode_solver = BS3()
    elseif ode_solver == :rk4  # recommended for normal scenario
        _ode_solver = RK4()
    elseif ode_solver == :dp5  # recommended for normal scenario
        _ode_solver = DP5()
    elseif ode_solver == :tsit5  # recommended for normal scenario
        _ode_solver = Tsit5()
    elseif ode_solver == :vern9  # for more accuracy
        _ode_solver = Vern9(lazy=false)
    end

    println("Scenario = $(scenario)")
    p_ref = generate_reference_position_trajectory(scenario)
    ṗ_ref = t -> ForwardDiff.derivative(p_ref, t)
    p̈_ref = t -> ForwardDiff.derivative(ṗ_ref, t)
    p⃛_ref = t -> ForwardDiff.derivative(p̈_ref, t)

    println("Method = $(method)")
    plant = LiftCruiseVTOL2D()
    if method == :opt
        df, elapsed_times = sim_opt(plant, p_ref, ṗ_ref, p̈_ref; t0, tf, savestep, _ode_solver,
                                    T_s, interp, dt)
    elseif method == :tvopt
        T_s = nothing
        interp = nothing
        df, elapsed_times = sim_tvopt(plant, p_ref, ṗ_ref, p̈_ref, p⃛_ref; t0, tf, savestep, _ode_solver, dt)
    end

    name =  "$(method)_$(scenario)_$(ode_solver)"
    if method == :opt
        T_s_str = @sprintf "%.2E" T_s
        name = name * "_interp_$(interp)_T_s_$(T_s_str)"
    end
    save("data/$(name)_$(now()).jld2", Dict(
        "method" => method,
        "scenario" => scenario,
        "df" => df,
        "elapsed_times" => elapsed_times,
        "t0" => t0,
        "ode_solver" => ode_solver,
        "tf" => tf,
        "T_s" => T_s,
        "interp" => interp,
        "θ_min" => plant.θ_min,
        "θ_max" => plant.θ_max,
        "δ_e_min" => plant.δ_e_min,
        "δ_e_max" => plant.δ_e_max,
        "savestep" => savestep,
    ))
    return df
end


function plot_result(path; lw=1.2)
    data = load(path)

    df = data["df"]
    # θ_min = data["θ_min"]
    # θ_max = data["θ_max"]
    δ_e_min = data["δ_e_min"]
    δ_e_max = data["δ_e_max"]
    elapsed_times = data["elapsed_times"]
    @show elapsed_times
    @show sum(t for (key, t) in elapsed_times)
    method = data["method"]
    scenario = data["scenario"]
    # T_s = data["T_s"]
    # ode_solver = data["ode_solver"]
    println("Plot info:")
    println("method: $(method)")
    println("scenario: $(scenario)")

    len_t = length(df.time)
    idx = 1:1:len_t
    idx_ref = 1:10:len_t  # for better density of dash plot, see https://discourse.julialang.org/t/constant-dash-density-in-plots-jl/34087
    ts = df.time[idx]
    t_refs = df.time[idx_ref]
    df_sol = df.sol[idx]
    ps = [datum.p for datum in df_sol]
    p_refs = df.p_ref[idx_ref]
    ṗs = [datum.ṗ for datum in df_sol]
    ṗ_refs = df.ṗ_ref[idx_ref]
    ẋs = [ṗ[1] for ṗ in ṗs]
    żs = [ṗ[2] for ṗ in ṗs]
    ż_refs = [ṗ_ref[2] for ṗ_ref in ṗ_refs]
    p̈s = [datum.p̈ for datum in df_sol]
    p̈_refs = df.p̈_ref[idx_ref]
    z̈s = [p̈[2] for p̈ in p̈s]
    z̈_refs = [p̈_ref[2] for p̈_ref in p̈_refs]
    xs = [p[1] for p in ps]
    zs = [p[2] for p in ps]
    x_refs = [p[1] for p in p_refs]
    z_refs = [p[2] for p in p_refs]
    θs = [datum.θ for datum in df_sol]
    θ_refs = df.θ_ref[idx_ref]
    θ̇s = [datum.θ̇ for datum in df_sol]
    T_rs = [datum.T_r for datum in df_sol]
    T_ps = [datum.T_p for datum in df_sol]
    T_r_cmds = [datum.T_r_cmd for datum in df_sol]
    T_p_cmds = [datum.T_p_cmd for datum in df_sol]
    Ms = [datum.M for datum in df_sol]
    δ_es = [datum.δ_e for datum in df_sol]
    δ_e_cmds = [datum.δ_e_cmd for datum in df_sol]

    fig = plot(; layout=(3, 3), size=(800, 500))
    lc_sol = :blue
    # lc_sol2 = :blue
    name = basename(path)[1:end-5]
    ls_ref = :dash
    lc_ref = :red

    if scenario == :h2c
        x_min = -10
        x_max = 400
        z_min = -70
        z_max = 10
        legend_thrust = :topright
        legend_x = :topleft
    elseif scenario == :c2h
        x_min = -400
        x_max = 10
        z_min = -70
        z_max = 10
        legend_thrust = :topleft
        legend_x = :bottomright
    end

    if method == :opt
        method_name = "OL-Opt"
    elseif method == :tvopt
        method_name = "CL-TVOpt"
    end
    if scenario == :h2c
        idx_zoom = len_t-425:len_t-325
        idx_zoom_ref = len_t-425:1:len_t-325
    elseif scenario == :c2h
        idx_zoom = 1:400
        idx_zoom_ref = 1:1:400
    end
    plot!(fig, ts, xs; subplot=1,
          ylabel=L"$x$ [m]", xlabel=L"$t$ [s]", lc=lc_sol, lw,
          label=method_name,
          legend=legend_x,
          ylim=(x_min, x_max),
          )
    plot!(fig, t_refs, x_refs; subplot=1, label=L"$x_{\textrm{ref}}$", lc=lc_ref, ls=ls_ref, lw)
    plot!(fig, ts, zs; subplot=4,
          ylabel=L"$z$ [m]", xlabel=L"$t$ [s]", lc=lc_sol, lw,
          label=nothing,
          ylim=(z_min, z_max),
          )
    plot!(fig, t_refs, z_refs; subplot=4, label=L"$z_{\textrm{ref}}$", lc=lc_ref, ls=ls_ref, lw)
    plot!(fig, ts, rad2deg.(θs); subplot=8,
          ylabel=L"$\theta$ [deg]", xlabel=L"$t$ [s]", lc=lc_sol, lw,
          label=nothing,
          ylim=(-5, 2.5),
          )
    plot!(fig, t_refs, rad2deg.(θ_refs); subplot=8, label=L"$\theta_{\textrm{ref}}$", lc=lc_ref, ls=ls_ref, lw)

    plot!(fig, ts, T_rs; subplot=5,
          ylabel=L"$T_{r}$ [N]",
          label=nothing,
          xlabel=L"$t$ [s]", lc=lc_sol, lw,
          legend=legend_thrust,
          ylim=(-1, 40),
          )
    plot!(fig, ts, T_ps; subplot=2,
          ylabel=L"$T_{p}$ [N]",
          label=nothing,
          xlabel=L"$t$ [s]", lc=lc_sol, lw,
          ylim=(-1, 15),
          )
    plot!(fig, ts, rad2deg.(δ_es); subplot=7,
          ylabel=L"$\delta_{e}$ [deg]", xlabel=L"$t$ [s]", lc=lc_sol, lw,
          label=nothing,
          ylim=(rad2deg(δ_e_min)-1, rad2deg(δ_e_max)+1),
          )
    plot!(fig, ts[idx_zoom], T_rs[idx_zoom]; subplot=6,
          ylabel=L"$T_{r}$ (zoomed-in)",
          label=nothing,
          xlabel=L"$t$ [s]", lc=lc_sol, lw,
          legend=legend_thrust,
          )
    plot!(fig, ts[idx_zoom_ref], T_r_cmds[idx_zoom_ref]; subplot=6,
          label="cmd",
          lc=lc_ref, lw,
          ls=ls_ref,
          )
    plot!(fig, ts[idx_zoom], T_ps[idx_zoom]; subplot=3,
          ylabel=L"$T_{p}$ (zoomed-in)",
          label=nothing,
          xlabel=L"$t$ [s]", lc=lc_sol, lw,
          )
    plot!(fig, ts[idx_zoom_ref], T_p_cmds[idx_zoom_ref]; subplot=3,
          label="cmd",
          lc=lc_ref, lw,
          ls=ls_ref,
          )
    # plot!(fig, ts[idx_zoom], rad2deg.(δ_es[idx_zoom]); subplot=9,
    #       ylabel=L"$\delta_{e}$ (zoomed-in)", xlabel=L"$t$ [s]", lc=lc_sol, lw,
    #       label=nothing,
    #       # ylim=(30-1e-7,30+1e-7)
    #       )
    plot!(fig, ts[idx_zoom], (rad2deg.(θs))[idx_zoom]; subplot=9,
          ylabel=L"$\theta$ [deg] (zoomed-in)", xlabel=L"$t$ [s]", lc=lc_sol, lw,
          label=nothing,
          ylim=(-1, 1),
          )
    plot!(fig, ts[idx_zoom_ref], (rad2deg.(df.θ_ref))[idx_zoom_ref]; subplot=9, label=L"$\theta_{\textrm{ref}}$", lc=lc_ref, ls=ls_ref, lw)

    savefig(fig, "figures/" * name * ".pdf")
    savefig(fig, "figures/" * name * ".png")
    return fig
end


function average_elapsed_time(method, scenario; dir_name="data_time")
    println("method: $(method)")
    println("scenario: $(scenario)")
    times = []
    files = readdir(dir_name)
    # @infiltrate
    for file in files
        chucks = split(file, "_")
        if chucks[1] == String(method)
            if chucks[2] == String(scenario)
                data = load(dir_name * "/" * file)
                elapsed_times = data["elapsed_times"]
                if method == :opt
                    push!(times, elapsed_times["time_opt"])
                elseif method == :tvopt
                    push!(times, elapsed_times["time_ode"])
                end
            end
        end
    end
    t_avg = sum(times) / length(times)
    println("average elapsed time: $(t_avg)")
    return t_avg
end


"""
Load data in `./data_plot`.
"""
function animate(path)
    plant = LiftCruiseVTOL2D()

    data = load(path)

    df = data["df"]
    δ_e_min = data["δ_e_min"]
    δ_e_max = data["δ_e_max"]
    method = data["method"]
    scenario = data["scenario"]
    println("Plot info:")
    println("method: $(method)")
    println("scenario: $(scenario)")
    ts = df.time

    anim = Animation()
    p_refs = df.p_ref
    θ_refs = df.θ_ref
    fig = plot()
    xlim = (0, 125)
    xticks = 0:20:120
    @showprogress for (idx, t) in enumerate(ts)
        if idx % 30 == 1
            lc_actual = :blue
            lc_ref = :red
            lc_lim = :black
            ls_ref = :dash
            lw = 2.0
            datum = df.sol[idx]
            p = datum.p
            θ = datum.θ
            X = State(plant)(p, θ)
            p_ref = p_refs[idx]
            width = 30
            p_center = p_ref
            zlim_lower = max(-p_center[2]-width, -1)
            zlim_upper = zlim_lower + 2*width
            # VTOL 3D view
            fig_vtol = plot(;
                            xlim=(-width, +width),
                            ylim=(+p_center[1]-width, +p_center[1]+width),
                            zlim=(zlim_lower, zlim_upper),
                            aspect_ratio=:equal,
                            camera=(60, 20),
                            # camera=(90, 0),
                            )
            plot!(
                fig_vtol,
                zeros(length(p_refs)),  # p[2] = z
                [p[1] for p in p_refs],  # p[1] = x
                [-p[2] for p in p_refs];  # p[2] = z
                label=nothing,
                lc=lc_ref,
                ls=ls_ref,
            )
            plot!(fig_vtol, plant, X; length_scale=10.0)
            # N
            ylim_N = scenario == :h2c ? (-10, 350) : (-350, 10)
            fig_N = plot(
                ts, [p[1] for p in p_refs];
                label=nothing, ylabel="N [m]",
                lc=lc_ref,
                ls=ls_ref,
                lw,
                ylim=ylim_N,
                xlim,
                xticks,
            )
            plot!(
                fig_N, ts[1:idx], [datum.p[1] for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
            )
            # U
            fig_U = plot(;
                         ylabel="U [m]",
                         ylim=(-5, 70),
                         xlim,
                         xticks,
                         )
            plot!(ts, [-p[2] for p in p_refs];
                  label=nothing,
                  lc=lc_ref,
                  ls=ls_ref,
                  lw,
                  )
            plot!(
                fig_U, ts[1:idx], [-datum.p[2] for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
            )
            # θ
            fig_θ = plot(
                ts, [rad2deg(θ) for θ in θ_refs];
                label=nothing, ylabel=L"$\theta$ [deg]",
                lc=lc_ref,
                ls=ls_ref,
                lw,
                ylim=(-6, 1.5),
            )
            plot!(
                fig_θ, ts[1:idx], [rad2deg(datum.θ) for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
            )
            # θ (zoomed-in)
            if scenario == :h2c
                xlim_θ_zoomed_in = (83, 93)
                ylim_θ_zoomed_in = (-0.5, 0.5)
                xticks_θ_zoomed_in = 84:2:92
                yticks_θ_zoomed_in = -1:0.5:1
                inset_θ = (1, bbox(0.15,0.55,0.4,0.3))
            elseif scenario == :c2h
                inset_θ = (1, bbox(0.60,0.55,0.4,0.4))
                if method == :opt
                    xlim_θ_zoomed_in = (0, 45)
                    xticks_θ_zoomed_in = 0:10:40
                elseif method == :tvopt
                    xlim_θ_zoomed_in = (0, 5)
                    xticks_θ_zoomed_in = 0:2:10
                end
                ylim_θ_zoomed_in = (-1, 1)
                yticks_θ_zoomed_in = -1:0.5:1
            else
                error("")
            end
            # fig_θ_zoomed_in = plot(;
            #                         ylabel=L"$\theta$ [deg] (zoomed-in)",
            #                         xlim=xlim_θ_zoomed_in,
            #                         xticks=xticks_θ_zoomed_in,
            #                         ylim=ylim_θ_zoomed_in,
            #                         yticks=yticks_θ_zoomed_in,
            #                         )
            vspan!(fig_θ, [xlim_θ_zoomed_in...]; color = :green, alpha=0.2, label=nothing);
            vspan!(fig_θ, [xlim_θ_zoomed_in...];
                   inset = inset_θ,
                   subplot=2,
                   color = :green, alpha=0.2, label=nothing);
            plot!(
                ts, [rad2deg(θ) for θ in θ_refs];
                lc=lc_ref,
                ls=ls_ref,
                subplot=2,
                label=nothing,
                xlim=xlim_θ_zoomed_in,
                xticks=xticks_θ_zoomed_in,
                ylim=ylim_θ_zoomed_in,
                yticks=yticks_θ_zoomed_in,
            )
            plot!(
                ts[1:idx], [rad2deg(datum.θ) for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
                subplot=2,
            )
            # T_r
            fig_T_r = plot(;
                           ylabel=L"$T_{r}$ [N]",
                           # ylim=(-60, 60),
                           ylim=(-5, 60),
                           xlim,
                           xticks,
                           )
            plot!(
                ts, 0*ones(length(ts));
                label=nothing,
                lc=lc_lim,
            )
            plot!(
                ts[1:idx], [datum.T_r_cmd for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_ref,
                ls=ls_ref,
                lw=lw/2,
            )
            plot!(
                ts[1:idx], [datum.T_r for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
            )
            # T_r (zoomed-in)
            if scenario == :h2c
                xlim_T_r_zoomed_in = (83, 93)
                ylim_T_r_zoomed_in = (-0.5, 0.5)
                xticks_T_r_zoomed_in = 84:2:92
                yticks_T_r_zoomed_in = -1:0.5:1
                inset_T_r = (1, bbox(0.5,0.0,0.4,0.4))
            elseif scenario == :c2h
                inset_T_r = (1, bbox(0.55,0.0,0.4,0.35))
                if method == :opt
                    xlim_T_r_zoomed_in = (0, 45)
                    xticks_T_r_zoomed_in = 0:10:40
                elseif method == :tvopt
                    xlim_T_r_zoomed_in = (0, 5)
                    xticks_T_r_zoomed_in = 0:2:10
                end
                if method == :opt
                    ylim_T_r_zoomed_in = (-550, 100)
                    yticks_T_r_zoomed_in = -500:100:100
                elseif method == :tvopt
                    ylim_T_r_zoomed_in = (-1, 20)
                    yticks_T_r_zoomed_in = 0:10:20
                end
            else
                error("")
            end
            # fig_T_r_zoomed_in = plot(;
            #                          ylabel=L"$T_{r}$ [N] (zoomed-in)",
            #                          xlim=xlim_T_r_zoomed_in,
            #                          xticks=xticks_T_r_zoomed_in,
            #                          ylim=ylim_T_r_zoomed_in,
            #                          yticks=yticks_T_r_zoomed_in,
            #                          )
            vspan!(fig_T_r, [xlim_T_r_zoomed_in...]; color = :green, alpha=0.2, label=nothing);
            vspan!(fig_T_r, [xlim_T_r_zoomed_in...];
                   inset = inset_T_r,
                   subplot=2,
                   color = :green, alpha=0.2, label=nothing);
            plot!(
                ts, 0*ones(length(ts));
                lc=lc_lim,
                label=nothing,
                subplot=2,
                xlim=xlim_T_r_zoomed_in,
                xticks=xticks_T_r_zoomed_in,
                ylim=ylim_T_r_zoomed_in,
                yticks=yticks_T_r_zoomed_in,
            )
            plot!(
                ts[1:idx], [datum.T_r_cmd for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_ref,
                ls=ls_ref,
                lw=lw/2,
                subplot=2,
            )
            plot!(
                ts[1:idx], [datum.T_r for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
                subplot=2,
            )
            # T_p
            ylim_T_p = (-1, 20)
            fig_T_p = plot(;
                           ylabel=L"$T_{p}$ [N]",
                           # ylim=(-20, 20),
                           ylim=ylim_T_p,
                           xlim,
                           xticks,
                           )
            plot!(
                ts, 0*ones(length(ts));
                label=nothing,
                lc=lc_lim,
            )
            plot!(
                ts[1:idx], [datum.T_p_cmd for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_ref,
                ls=ls_ref,
                lw=lw/2,
            )
            plot!(
                ts[1:idx], [datum.T_p for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
            )
            # T_p (zoomed-in)
            if scenario == :h2c
                xlim_T_p_zoomed_in = (83, 93)
                ylim_T_p_zoomed_in = (5, 15)
                xticks_T_p_zoomed_in = 84:2:92
                yticks_T_p_zoomed_in = 5:5:15
                inset_T_p = (1, bbox(0.2,0.0,0.4,0.4))
            elseif scenario == :c2h
                inset_T_p = (1, bbox(0.5,0.0,0.4,0.4))
                if method == :opt
                    xlim_T_p_zoomed_in = (0, 45)
                    xticks_T_p_zoomed_in = 0:10:40
                elseif method == :tvopt
                    xlim_T_p_zoomed_in = (0, 5)
                    xticks_T_p_zoomed_in = 0:2:10
                end
                ylim_T_p_zoomed_in = (-1, 15)
                yticks_T_p_zoomed_in = 0:5:15
            else
                error("")
            end
            # fig_T_p_zoomed_in = plot(;
            #                          ylabel=L"$T_{p}$ [N] (zoomed-in)",
            #                          xlim=xlim_T_p_zoomed_in,
            #                          xticks=xticks_T_p_zoomed_in,
            #                          ylim=ylim_T_p_zoomed_in,
            #                          yticks=yticks_T_p_zoomed_in,
            #                          )
            vspan!(fig_T_p, [xlim_T_p_zoomed_in...]; color = :green, alpha=0.2, label=nothing);
            vspan!(fig_T_p, [xlim_T_p_zoomed_in...];
                   inset = inset_T_p,
                   subplot=2,
                   color = :green, alpha=0.2, label=nothing);
            plot!(
                ts, 0*ones(length(ts));
                lc=lc_lim,
                label=nothing,
                subplot=2,
                xlim=xlim_T_p_zoomed_in,
                xticks=xticks_T_p_zoomed_in,
                ylim=ylim_T_p_zoomed_in,
                yticks=yticks_T_p_zoomed_in,
            )
            plot!(
                ts[1:idx], [datum.T_p_cmd for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_ref,
                ls=ls_ref,
                lw=lw/2,
                subplot=2,
            )
            plot!(
                ts[1:idx], [datum.T_p for datum in df.sol[1:idx]];
                label=nothing,
                lc=lc_actual,
                lw,
                subplot=2,
            )
            # δ_e
            fig_δ_e = plot(;
                            ylabel=L"$\delta_{e}$ [deg]",
                            ylim=(-35, 35),
                            xlim,
                            xticks,
                            )
            plot!(
                ts, rad2deg(δ_e_min)*ones(length(ts));
                label=nothing,
                lc=lc_lim,
            )
            plot!(
                ts, rad2deg(δ_e_max)*ones(length(ts));
                label=nothing,
                lc=lc_lim,
            )
            if method == :opt
                plot!(
                    ts, [rad2deg(datum.δ_e) for datum in df.sol];
                    lc=lc_ref,
                    ls=ls_ref,
                    label=nothing,
                    lw,
                )
            end
            plot!(
                ts[1:idx], [rad2deg(datum.δ_e) for datum in df.sol[1:idx]];
                lc=lc_actual,
                label=nothing,
                lw,
            )
            l = @layout [
                a{0.35w} [grid(3, 2)]
            ]
            fig = plot(
                fig_vtol,
                fig_N, fig_T_p,# fig_T_p_zoomed_in,
                fig_U, fig_T_r,# fig_T_r_zoomed_in,
                fig_δ_e, fig_θ,# fig_θ_zoomed_in;
                layout=l,
                size=(900, 600),
            )
            frame(anim, fig)
        end
    end
    # error("마지막 장면 오래 지속되게 하기")
    for _ in 1:20
        frame(anim, fig)
    end
    path_anim = "animations/$(basename(path)[1:end-5]).gif"
    return gif(anim, path_anim)
end
