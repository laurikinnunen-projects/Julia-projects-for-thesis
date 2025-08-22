using DifferentialEquations
using StochasticDiffEq
using SciMLBase
using DiffEqPhysics
using Random
using Plots

# Parameters
params = (
    a = 0.01,                          # Dimensionless DC parameter
    q = 0.1,                           # Dimensionless AC paramter
    m = 1e-15,                         # Mass [kg]
    w_rf = 2π * 1.0e6,                 # RF frequency [rad/s]
    E0 = 1.0e7,                        # Optical field amplitude [V/m]
    w0 = 1e-6,                         # Beam waist [m]
    a_p = 1.0e-30,                     # Particle polarizability SI units
    λ = 1500e-9,                       # Laser wavelength [m]
    σv = 1e-19,                        # Noise amplitude
    Γ = 1e5                          # Damping coefficient (Millen: for room temperature)
)

# Initial conditions: [vx, vy, vz, x, y, z]
v0 = [0.0, 0.0, 0.0]                  # momentum / m  (velocity form)
u0 = [1e-5, 1e-5, 1.0e-5]             # initial position
u_init = vcat(v0, u0)

# half a microsecond time interval, enough for few oscillations of secular motion
tspan = (0.0, 5e-5)                    

# Rayleigh range for Gaussian beam
function w_of_z(x, w0, λ)
    zR = π * w0^2 / λ
    return w0 * sqrt(1 + (x / zR)^2)
end

# Equations of motion for a cylindrically symmetric Paul trap
function f!(du, u, p, t)
    a, q, m, w_rf, E0, w0, a_p, λ, σv, Γ = p    # Function parameters

    # unpack state
    vx, vy, vz = u[1:3]     # Velocity
    x, y, z   = u[4:6]      # Position

    # dx/dt = velocity
    du[4] = vx
    du[5] = vy
    du[6] = vz

    # RF trap potential
    Vx_rf = -0.25 * m * w_rf^2 * (a + q * cos(w_rf * t)) * x
    Vy_rf = -0.25 * m * w_rf^2 * (a + q * cos(w_rf * t)) * y
    Vz_rf = -0.25 * m * w_rf^2 * (-2 * a - 4 * q * cos(w_rf * t)) * z

    # Optical potential force (Gaussian beam along x-axis)
    wz = w_of_z(x, w0, λ)
    r_sq = y^2 + z^2
    exp_factor = exp(-2 * r_sq^2 / wz^2)
    U0 = 0.5 * a_p * E0^2             # potential depth
    Vy_opt = (4 * U0 * x / wz^2) * exp_factor
    Vz_opt = (4 * U0 * x / wz^2) * exp_factor
    Vx_opt = 0.5 * a_p * E0^2 * w0/ (1 + (x * λ / pi * w0^2))  # no gradient along beam axis

    # acceleration = force / m
    du[1] = (Vx_rf + Vx_opt) / m - Γ * vx
    du[2] = (Vy_rf + Vy_opt) / m - Γ * vy
    du[3] = (Vz_rf + Vz_opt) / m - Γ * vz  
end

# Equations of motion for linear Paul trap
function h!(du, u, p, t)
    a, q, m, w_rf, E0, w0, a_p, λ, σv, Γ = p    # Function parameters

    # unpack state
    vx, vy, vz = u[1:3]     # Velocity
    x, y, z   = u[4:6]      # Position

    # dx/dt = velocity
    du[4] = vx
    du[5] = vy
    du[6] = vz

    # RF trap potential
    Vx_rf = -0.25 * m * w_rf^2 * (a + 2q * cos(w_rf * t)) * x
    Vy_rf = -0.25 * m * w_rf^2 * (a - 2q * cos(w_rf * t)) * y
    Vz_rf = -0.25 * m * w_rf^2 * (a - 0 * cos(w_rf * t)) * z 

    # Optical potential force (Gaussian beam along x-axis)
    wz = w_of_z(x, w0, λ)
    r_sq = y^2 + z^2
    exp_factor = exp(-2 * r_sq^2 / wz^2)
    U0 = 0.5 * a_p * E0^2             # potential depth
    Vy_opt = (4 * U0 * x / wz^2) * exp_factor
    Vz_opt = (4 * U0 * x / wz^2) * exp_factor
    Vx_opt = 0.5 * a_p * E0^2 * w0/ (1 + (x * λ / pi * w0^2))  # no gradient along beam axis

    # acceleration = force / m
    du[1] = (Vx_rf + Vx_opt) / m - Γ * vx
    du[2] = (Vy_rf + Vy_opt) / m - Γ * vy
    du[3] = (Vz_rf + Vz_opt) / m - Γ * vz  
end

# Diffusion term g(u,p,t) = noise part
function g!(du, u, p, t)
    a, q, m, w_rf, E0, w0, a_p, λ, σv = p   # Function parameters
    x, y, z = u[4:6]

    # Langevin noise term via stochastic solver Wiener process dW
    du .= 0.0
    du[1] = σv
    du[2] = σv
    du[3] = σv
end

# Define the problem as stochastic differential equation
prob_cyl = SDEProblem(f!, g!, u_init, tspan, params)

# Same for linear trap
prob_lin = SDEProblem(h!, g!, u_init, tspan, params)

# Solve with an SDE solver for Stratonovich problem, turning off adaptive steps
sol_cyl = solve(prob_cyl, RKMil(interpretation = SciMLBase.AlgorithmInterpretation.Stratonovich), adaptive=false, dt=1e-9, saveat=1e-8)

# Linear Trap solution
sol_lin = solve(prob_lin, RKMil(interpretation = SciMLBase.AlgorithmInterpretation.Stratonovich), adaptive=false, dt=1e-9, saveat=1e-8)

# Extract solution for cylindrical system
vx_vals_cyl = sol_cyl[1,:]; vy_vals_cyl = sol_cyl[2,:]; vz_vals_cyl = sol_cyl[3,:]
x_vals_cyl  = sol_cyl[4,:]; y_vals_cyl  = sol_cyl[5,:]; z_vals_cyl  = sol_cyl[6,:]
t_vals_cyl  = sol_cyl.t

# Extract solution for linear system
vx_vals_lin = sol_lin[1,:]; vy_vals_lin = sol_lin[2,:]; vz_vals_lin = sol_lin[3,:]
x_vals_lin  = sol_lin[4,:]; y_vals_lin  = sol_lin[5,:]; z_vals_lin  = sol_lin[6,:]
t_vals_lin  = sol_lin.t

#= Plot q(t) for x/y/z cylindrical system
p1_cyl = plot(t_vals_cyl, x_vals_cyl, label="x", xlabel="Time (s)", ylabel="Position (m)")
plot!(p1_cyl, t_vals_cyl, y_vals_cyl, label="y")
plot!(p1_cyl, t_vals_cyl, z_vals_cyl, label="z")

# Plot v(t) for x/y/z cylindrical system
p2_cyl = plot(t_vals_cyl, vx_vals_cyl, label="vx", xlabel="Time (s)", ylabel="Velocity (m/s)")
plot!(p2_cyl, t_vals_cyl, vy_vals_cyl, label="vy")
plot!(p2_cyl, t_vals_cyl, vz_vals_cyl, label="vz")

plot(p1_cyl, p2_cyl, layout=(2,1)) =#

# Plot q(t) for x/y/z cylindrical system
p1_lin = plot(t_vals_lin, x_vals_lin, label="x", xlabel="Time (s)", ylabel="Position (m)")
plot!(p1_lin, t_vals_lin, y_vals_lin, label="y")
plot!(p1_lin, t_vals_lin, z_vals_lin, label="z")

# Plot v(t) for x/y/z cylindrical system
p2_lin = plot(t_vals_lin, vx_vals_lin, label="vx", xlabel="Time (s)", ylabel="Velocity (m/s)")
plot!(p2_lin, t_vals_lin, vy_vals_lin, label="vy")
plot!(p2_lin, t_vals_lin, vz_vals_lin, label="vz")

plot(p1_lin, p2_lin, layout=(2,1)) #

# = savefig(p1_cyl, "Cylindrical_trap_q(t).png") = #
# = savefig(p1_lin, "Linear_trap_q(t).png") = #
# = savefig(p2_cyl, "Cylindrical_trap_v(t).png") = #
# = savefig(p2_lin, "Linear_trap_v(t).png") = #





#Animation of motion of motion in three dimensional position space for cylindrical system
#= anim = @animate for i in 1:5:length(t_vals_cyl)  # skip frames to speed things up
    plot3d(
        x_vals_cyl[1:i], y_vals_cyl[1:i], z_vals_cyl[1:i],                      # Position intervals
        xlim=(-1.5e-5, 1.5e-5), ylim=(-1.5e-5, 1.5e-5), zlim=(0, 4e-5),     # Axis limits
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",             # Axis labels
        label=false, lw=2,                                          # No graph label, linewidth = 2
        title="t = $(round(t_vals_cyl[i]*1e6, digits=2)) μs",           # Time step
        camera=(45,35),                                             # Plot tilt (horizontal, vertical)
        grid=true,                                                  # Add grid for clarity
        gridalpha=0.8                                               # Grid transparency
    )
end

# Save as GIF
gif(anim, "cyl_particle_trap_SDE.gif", fps=40) =#

#Animation of motion of motion in three dimensional position space for linear system
#=anim = @animate for i in 1:4:length(t_vals_lin)  # skip frames to speed things up
    plot3d(
        x_vals_lin[1:i], y_vals_lin[1:i], z_vals_lin[1:i],                      # Position intervals
        xlim=(-1.5e-5, 1.5e-5), ylim=(-1.5e-5, 1.5e-5), zlim=(-2e-5, 2e-5),     # Axis limits
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",             # Axis labels
        label=false, lw=2,                                          # No graph label, linewidth = 2
        title="t = $(round(t_vals_lin[i]*1e6, digits=2)) μs",           # Time step
        camera=(45,35),                                             # Plot tilt (horizontal, vertical)
        grid=true,                                                  # Add grid for clarity
        gridalpha=0.8                                               # Grid transparency
    )
end

# Save as GIF
gif(anim, "lin_particle_trap_SDE.gif", fps=40) =#
