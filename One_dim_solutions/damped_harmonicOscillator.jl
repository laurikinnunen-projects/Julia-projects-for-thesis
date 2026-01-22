# Imports
using OrdinaryDiffEq
using Plots

# parameters
p = (
    m = 1,                              # Mass [kg]
    w_rf = 2π * 2.0,                    # Frequency
    gamma = 1.5                         # Damping coefficient
)

# Initial u values, time span and time range
u_init = [1,0]
t_span = (0.0, 10/(2*pi))
t_range = range(0.0,10/(2*pi), step = 0.01)

# Damped harmonic oscillator equation
function h!(du, u, p, t)
    (m, w_rf, gamma) = p
    du[1] = u[2] / m
    du[2] = -m * w_rf^2 * u[1] -gamma/m * u[2]

end    

# Analytical damped harmonic oscillator solution
function g(u_init, t, p)
    (m, w_rf, gamma) = p

    a = gamma / (2m)
    b = sqrt(w_rf^2 - a^2)

    A = u_init[1]
    B = u_init[2] / (m * b) + (a * u_init[1]) / b

    [exp.(-a .* t) .* (A .* cos.(b .* t) .+ B .* sin.(b .* t)),
    m .* exp.(-a .* t) .* ((-a*A + b*B) .* cos.(b .* t) .+(-b*A - a*B) .* sin.(b .* t))]

end
 

# Numerical solution 
prob = ODEProblem(h!, u_init, t_span, p)
sol = solve(prob, dt = 0.01)

# Call for analytical solution
sol_a = g(u_init, t_range, p)

# Plot
q = plot(sol.t, sol[1,:], label = "x")
plot!(sol.t, sol[2,:], label = "p")
plot!(t_range, sol_a[1,:], label = "x_a")
plot!(t_range, sol_a[2,:], label = "p_a")
plot(q)
