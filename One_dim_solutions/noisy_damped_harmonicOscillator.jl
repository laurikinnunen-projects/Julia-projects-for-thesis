# Imports
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqNoiseProcess
using SciMLBase
using Random
using Plots
using DiffEqNoiseProcess


# parameters
p = (
    m = 1,                              # Mass [kg]
    w_rf = 2π * 2.0,                    # Frequency
    gamma = 2,                        # Damping coefficient
    D = 1                              # Squared noise ecoefficient
)
# Initial u values, time span and time range
u_init = [1,0]
t_span = (0.0, 10/(2*pi))
t_range = range(0.0,10/(2*pi), step = 0.01)
dt = 1e-4

# Random seed and Wiener process
rng = MersenneTwister(1234)
Wproc = WienerProcess(0.0, 0.0; rng=rng)

# Noisy damped harmonic oscillator equation deterministic part
function h!(du, u, p, t)
    (m, w_rf, gamma, D) = p
    du[1] = u[2] / m
    du[2] = -m * w_rf^2 * u[1] -gamma/m * u[2]

end    

# Noisy damped harmonic oscillator equation diffusion part
function g!(du, u, p, t)
    (m, w_rf, gamma, D) = p

    du[1] = 0
    du[2] = sqrt(D)

end

# Numerical solution 
prob = SDEProblem(h!, g!, u_init, t_span, p, noise = Wproc)
sol = solve(prob, EM(), adaptive = false, dt = dt)  # Euler-Maruyama method

# Wiener increments
dw = sol.W.dW
W = sol.W

# Analytical noisy damped harmonic oscillator solution, deterministic part
function q_deterministic(u_init, t, p)
    (m, w_rf, gamma, D) = p

    a = gamma / (2m)
    b = sqrt(w_rf^2 - a^2)

    A = u_init[1]
    B = u_init[2]/(m*b) + (a*u_init[1])/b

    exp.(-a .* t) .* (A .* cos.(b .* t) .+ B .* sin.(b .* t))
end

# Deterministic momentum 
function p_deterministic(u_init, t, p)
    (m, w_rf, gamma, D) = p

    a = gamma / (2m)
    b = sqrt(w_rf^2 - a^2)

    A = u_init[1]
    B = u_init[2]/(m*b) + (a*u_init[1])/b

    m .* exp.(-a .* t) .* ((-a*A + b*B) .* cos.(b .* t) .+(-b*A - a*B) .* sin.(b .* t))
end

# Analytical noisy damped harmonic oscillator solution, diffusion part
function q_noise(dw, t, p)
    (m, w_rf, gamma, D,) = p

    a = gamma / (2m)
    b = sqrt(w_rf^2 - a^2)
    
    N = length(t)
    qn = zeros(N)
    for i in 2:N
        s = t[1:i-1] # All previous times s < t[i]
        Gf = exp.(-a .* (t[i] .- s)) .* sin.(b .* (t[i] .- s)) # Green's function
        qn[i] = sqrt(D)/(m*b) * sum(Gf .* dw[1:i-1]) # Update q noise value
    end
    return qn
end

# Noisy momentum, simalr to the position but with the time derivative of the solution
function p_noise(dw, t, p)
    (m, w_rf, gamma, D) = p

    a = gamma/(2m)
    b = sqrt(w_rf^2 - a^2)

    N = length(t)
    pn = zeros(N)
    
    for i in 2:N
        s = t[1:i-1]
        Gf = exp.(-a .* (t[i] .- s)) .* (b .* cos.(b .*(t[i] .- s)) .- a .* sin.(b .* (t[i] .- s)))
        pn[i] = sqrt(D)/b * sum(Gf.* dw[1:i-1])
    end
    return pn
end
 
q_det = q_deterministic(u_init, t_range, p)
q_sto = q_noise(W, t_range, p)
q_ana = q_det + q_sto                           # Total semi-analytical positional solution

p_det = p_deterministic(u_init, t_range, p)
p_sto = p_noise(W, t_range, p)
p_ana = p_det + p_sto                           # Total semi-analytical momentum solution

# Plot
q = plot(sol.t, sol[1,:], label = "x")
plot!(sol.t, sol[2,:], label = "p")
plot!(t_range, q_ana, label = "x_a")
plot!(t_range, p_ana, label = "p_a")
plot(q)
