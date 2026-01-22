# Imports
using OrdinaryDiffEq
using Plots

# parameters 
p = (
    m = 1,                          # Mass [kg]
    w_rf = 2π * 2.0                 # Frequency
)

# initial range, time span and time range 
u_init = [1,0]
t_span = (0.0, 10/(2*pi))
t_range = range(0.0,10/(2*pi), step = 0.01)

# General Harmonic oscillator
function h!(du, u, p, t)
    (m, w_rf) = p
    du[1] = u[2] / m
    du[2] = -m * w_rf^2 * u[1]

end    

# Analytical solution for harmonic oscillator
function g(u_init, t, p)
    (m, w_rf) = p

    [u_init[1] * cos.(w_rf * t) + u_init[2] /(m*w_rf) * sin.(w_rf * t),
    u_init[2] * cos.(w_rf * t) - u_init[1] * m * w_rf * sin.(w_rf * t)]
end    

# Numerical solution for harmonic oscillator
prob = ODEProblem(h!, u_init, t_span, p)
sol = solve(prob, dt = 0.01)

# Call analytical solution
sol_a = g(u_init, t_range, p)

# Plot
q = plot(sol.t, sol[1,:], label = "x")
plot!(sol.t, sol[2,:], label = "p")
plot!(t_range, sol_a[1,:], label = "x_a")
plot!(t_range, sol_a[2,:], label = "p_a")
plot(q)
