# Imports, not all in use currently
using DifferentialEquations
using OrdinaryDiffEq
using StochasticDiffEq
using SciMLBase
using DiffEqPhysics
using Random
using Plots
using ForwardDiff
using QuadGK
using StaticArrays

# Parameters
params = (
    a = 0.01,                          # Dimensionless DC parameter
    q = 0.1,                           # Dimensionless AC paramter
    m = 1e-15,                         # Mass [kg]
    w_rf = 2π * 2.0e3,                 # RF frequency [rad/s]
    P1 = 150e-3,                       # (Low) Optical power in the positive arm
    P2 = 150e-3,                       # (Low) Optical power in the negative arm                        
    w01 = 350e-6,                      # Beam waist [m] in the positive direction
    w02 = 300e-6,                      # Beam waist [m] in the negative direction 
    a_p = -0.3e-25,                     # Particle polarizability SI units
    λ = 1064e-9,                       # Laser wavelength [m]
    σv = 1.3e-20,                      # Noise amplitude
    Γ = 1e-20,                         # Damping coefficient (Millen: for room temperature)
    n = 1.45,                          # Refractive index of a silica Particle
    c = 3e8,                           # Speed of ligth                     
    e0 = 9e-12,                        # Vacuum permittivity
    ϵ = 1e-12                          # Condition for dvision by zero
)

# Initial conditions: [vx, vy, vz, x, y, z]
v0 = [0,0,0]                       # Initial momentum / m  (velocity form)
u0 = [1e-10,1e-10,1e-10]           # Initial position
u_init = vcat(v0, u0)               


# Time span, adjustbale with the time step
tspan = (0.0, 5e-2)                   

# Rayleigh range for Gaussian beam
function w_of_z1(x, w01, λ, n)
    zR1 = π * w01^2 * n / λ
    return w01 * sqrt(1 + (x / zR1)^2)
end

function w_of_z2(x, w02, λ, n)
    zR2 = π * w02^2 * n / λ
    return w02 * sqrt(1 + (-x / zR2)^2)
end


function Eopt_total_sqrd(pos::AbstractVector, p)
    # Postional coordinates
    x, y, z = pos

    # Function parameters
    (; a, q, m, w_rf, w01, w02, c, e0, n, P1, P2, a_p, λ, σv, Γ, ϵ) = p

    # Waist function 
    wz1 = w_of_z1(x, w01, λ, n)
    wz2 = w_of_z2(x, w02, λ, n)

    # Inverse of the invariant part of Rayleigh range for gradients
    inv_zR1 = λ / (π * w01^2 * n)
    inv_zR2 = λ / (π * w02^2 * n)

    # Rayleigh range
    zR1 = π * w01^2 * n / λ
    zR2 = π * w02^2 * n / λ

    # Radial distance from optical axis
    r_sq = y^2 + z^2

    # Exponential factors
    exp1 = exp(- r_sq / wz1^2)
    exp2 = exp(- r_sq / wz2^2)

    # Field amplitude at origin
    E01 = sqrt(4 * P1 / (π * w01^2 * c * e0))
    E02 = sqrt(4 * P2 / (π * w02^2 * c * e0))

    # Wave number
    k = 2 * π * n / λ

    # Radius of curvature with condition for the case x=0
    Rz1 = abs(x) < ϵ ? x * (1 + (zR1 / (x + ϵ))^2) : x * (1 + (zR1 / x)^2)
    Rz2 = abs(x) < ϵ ? -x * (1 + (zR2 / (-x + ϵ))^2) : -x * (1 + (zR2 / -x)^2)

    # Ocsillation frequency
    freq_o = 2 * π * c / λ

    # The argument for the oscillating facor with Gouy phase
    φ1 = k*x + k*r_sq/(2*Rz1) - atan(x/zR1) #+ freq_o*t
    φ2 = -k*x + k*r_sq/(2*Rz2) - atan(-x/zR2) #+ freq_o*t

    # The total field without time avergaing, just for reminder
    # E_opt = ((E01 * w01/wz1)*exp1*cos(φ1) - (E02 * w02/wz2)*exp2*cos(φ2))^2

    # The parts of the field thats are left after squaring and will be taken gradient over
    E2_opt = ((E01 * w01/wz1)*exp1)^2 + ((E02 * w02/wz2)* exp2)^2

    # dE^2_opt / dx calculated analytically
    dE1_2_x = E01^2 * (((w01/wz1)^2) * (4 * r_sq * x * inv_zR1^2) / (w01^2 * (1 + (inv_zR1 * x)^2)^2) * exp1^2 - exp1^2 *((2 * inv_zR1^2 * x) / (1 + (inv_zR1 * x)^2)^2))
    dE2_2_x = E02^2 * (((w02/wz2)^2) * (4 * r_sq * x * inv_zR2^2) / (w02^2 * (1 + (-inv_zR2 * x)^2)^2) * exp2^2 - exp2^2 *((2 * inv_zR2^2 * x) / (1 + (-inv_zR2 * x)^2)^2))
    dE2_x = dE1_2_x + dE2_2_x
    
    # dE^2_opt / dy calculated analytically
    dE1_2_y = E01^2 * (w01/wz1)^2 * (- 4 * y / w01^2 * (1 + (inv_zR1 * x)^2)) * exp1^2
    dE2_2_y = E02^2 * (w02/wz2)^2 * (- 4 * y / w02^2 * (1 + (-inv_zR2 * x)^2)) * exp2^2
    dE2_y = dE1_2_y + dE2_2_y

    # dE^2_opt / dz calculated analytically
    dE1_2_z = E01^2 * (w01/wz1)^2 * (- 4 * z / w01^2 * (1 + (inv_zR1 * x)^2)) * exp1^2
    dE2_2_z = E02^2 * (w02/wz2)^2 * (- 4 * z / w02^2 * (1 + (-inv_zR2 * x)^2)) * exp2^2
    dE2_z = dE1_2_z + dE2_2_z

    return SVector(dE2_x, dE2_y, dE2_z)
end


# Equations of motion
function h!(du, u, p, t)
    # Function parameters
    (; a, q, m, w_rf, w01, w02, c, e0, n, P1, P2, a_p, λ, σv, Γ) = p

    # unpack state
    vx, vy, vz = u[1:3]     # Velocity
    x, y, z   = u[4:6]      # Position

    # dx/dt = velocity
    du[4] = vx
    du[5] = vy
    du[6] = vz

    # The gradient of the Paul trap potential
    Fx_rf = - m * w_rf^2 / 4 *  (a + 2q * cos(w_rf * t)) * x
    Fy_rf = - m * w_rf^2 / 4 * (a - 2q * cos(w_rf * t)) * y
    Fz_rf = - m * w_rf^2 / 4 * (a - 0 * cos(w_rf * t)) * z 

    # Force = 1/2 α ∇⟨E^2⟩, stores the svector array values to the variable as ∇E2 = dE2_x, dE2_y, dE2_z
    ∇E2 = Eopt_total_sqrd(@SVector[x, y, z], p)
    
    # Assign to each force component the corresponding gradient component from the statevector and multiply by 1/2*a_p
    Fx_opt, Fy_opt, Fz_opt = 0.5 * a_p .* ∇E2 

    # The scattering force for the dipole approimation still needed here from the Poyinting vector
    
    # Accelerations and the damping # RF field commented out
    du[1] = (#=Fx_rf + =#Fx_opt) / m - Γ * vx
    du[2] = (#=Fy_rf + =#Fy_opt) / m - Γ * vy
    du[3] = (#=Fz_rf + =#Fz_opt) / m - Γ * vz
end

# Diffusion term g(u,p,t) = Wiener process
function g!(du, u, p, t)
    (; σv) = p
    fill!(du, 0.0)
    du[1] = σv
    du[2] = σv
    du[3] = σv
end


# Define the problem as stochastic differential equation
prob_lin = SDEProblem(h!, g!, u_init, tspan, params)

# Solve with an SDE solver for Stratonovich problem, turning off adaptive steps. SRI for 
sol_lin = solve(prob_lin, RKMil(interpretation=SciMLBase.AlgorithmInterpretation.Stratonovich), adaptive=false, dt = 5e-7)

# Extract solution for linear system
vx_vals_lin = sol_lin[1,:]; vy_vals_lin = sol_lin[2,:]; vz_vals_lin = sol_lin[3,:]
x_vals_lin  = sol_lin[4,:]; y_vals_lin  = sol_lin[5,:]; z_vals_lin  = sol_lin[6,:]
t_vals_lin  = sol_lin.t

# Plot q(t) for x/y/z
p1_lin = plot(t_vals_lin, x_vals_lin, label="x", xlabel="Time (s)", ylabel="Position (m)")
plot!(p1_lin, t_vals_lin, y_vals_lin, label="y")
plot!(p1_lin, t_vals_lin, z_vals_lin, label="z")

# Plot v(t) for x/y/z
p2_lin = plot(t_vals_lin, vx_vals_lin, label="vx", xlabel="Time (s)", ylabel="Velocity (m/s)")
plot!(p2_lin, t_vals_lin, vy_vals_lin, label="vy")
plot!(p2_lin, t_vals_lin, vz_vals_lin, label="vz")

plot(p1_lin, p2_lin, layout=(2,1))


#savefig(p1_lin, "Linear_trap_q(t)_Opt.png")  
#savefig(p2_lin, "Linear_trap_v(t)_Opt.png") 
