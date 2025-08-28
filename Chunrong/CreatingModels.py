import matplotlib.pyplot as plt
import numpy as np

import pybamm


# # --------creating a simple ODE model--------------------
# model   = pybamm.BaseModel()
# x       = pybamm.Variable('x')
# y       = pybamm.Variable('y')
#
# dxdt = 4 * x - 2 * y
# dydt = 3 * x - y
#
# # the governing equations must be provied in the explicit form d/dt = rhs
# # since pybamm only stores the right hand side (rhs) and assumes that the left hand side is the time derivative
# model.rhs = {x: dxdt, y: dydt}
#
# model.initial_conditions = {x: pybamm.Scalar(1), y: pybamm.Scalar(2)}
#
# disc = pybamm.Discretisation()  # use the default discretisation
# disc.process_model(model)
#
# solver = pybamm.ScipySolver()
# t = np.linspace(0, 1, 20)
# solution = solver.solve(model, t)
#
# t_sol, y_sol = solution.t, solution.y  # get solution times and states
# x = solution["x"]  # extract and process x from the solution
# y = solution["y"]  # extract and process y from the solution
#
# t_fine = np.linspace(0, t[-1], 1000)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
# ax1.plot(t_fine, 2 * np.exp(t_fine) - np.exp(2 * t_fine), t_sol, x(t_sol), "o")
# ax1.set_xlabel("t")
# ax1.legend(["2*exp(t) - exp(2*t)", "x"], loc="best")
#
# ax2.plot(t_fine, 3 * np.exp(t_fine) - np.exp(2 * t_fine), t_sol, y(t_sol), "o")
# ax2.set_xlabel("t")
# ax2.legend(["3*exp(t) - exp(2*t)", "y"], loc="best")
#
# plt.tight_layout()
# plt.show()


# # --------creating a simple ODE model--------------------
# model = pybamm.BaseModel()
#
# c = pybamm.Variable("Concentration", domain="negative particle")
# N = -pybamm.grad(c)  # define the flux
# dcdt = -pybamm.div(N)  # define the rhs equation
#
# model.rhs = {c: dcdt}  # add the equation to rhs dictionary
# # initial conditions
# model.initial_conditions = {c: pybamm.Scalar(1)}
#
# # boundary conditions
# lbc = pybamm.Scalar(0)
# rbc = pybamm.Scalar(2)
# model.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}
# model.variables = {"Concentration": c, "Flux": N}
# # =============================================
# # Dirichlet vs Neumann 边界条件
# # =============================================
#
# # PDE 一般形式 (以热传导为例):
# #   ∂T/∂t = α ∇²T
#
# # 1. Dirichlet 边界条件
# # ----------------------
# # 在边界上直接规定函数的值
# # 例如： T(x=0, t) = 300 K
# # 含义：边界温度始终固定为 300K
# # 物理意义：恒温边界（接触一个大热源/恒温体）
#
# # 2. Neumann 边界条件
# # ----------------------
# # 在边界上规定函数沿法向的导数 (∂T/∂n)
# # 例如： ∂T/∂x |_(x=0) = 0
# # 含义：边界温度梯度为零
# # 物理意义：绝热边界（没有热流穿过边界）
# #           或者规定一个固定的热流 q'' = -k ∂T/∂n
#
# # 3. Robin 边界条件（补充）
# # ----------------------
# # a*T + b*(∂T/∂n) = c
# # 含义：函数值和导数的线性组合
# # 物理意义：对流换热边界 (Newton cooling law)
# #           q'' = h*(T_surface - T_inf)
#
# # =============================================
# # 总结：
# # - Dirichlet: 边界值已知 (函数值)
# # - Neumann:   边界通量已知 (函数导数)
# # =============================================
# # define geometry
# r = pybamm.SpatialVariable(
#     "r", domain=["negative particle"], coord_sys="spherical polar"
# )
# geometry = {
#     "negative particle": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
# }
# # mesh and discretise
# submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
# var_pts = {r: 20}
# mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
#
# spatial_methods = {"negative particle": pybamm.FiniteVolume()}
# disc = pybamm.Discretisation(mesh, spatial_methods)
# disc.process_model(model)
#
# # solve
# solver = pybamm.ScipySolver()
# t = np.linspace(0, 1, 100)
# solution = solver.solve(model, t)
#
# # post-process, so that the solution can be called at any time t or space r
# # (using interpolation)
# c = solution["Concentration"]
#
# # plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
#
# ax1.plot(solution.t, c(solution.t, r=1))
# ax1.set_xlabel("t")
# ax1.set_ylabel("Surface concentration")
# r = np.linspace(0, 1, 100)
# ax2.plot(r, c(t=0.5, r=r))
# ax2.set_xlabel("r")
# ax2.set_ylabel("Concentration at t=0.5")
# plt.tight_layout()
# plt.show()


# # --------a step towards the single particle model--------------------
# model = pybamm.BaseModel()
#
# R = pybamm.Parameter("Particle radius [m]")
# D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
# j = pybamm.Parameter("Interfacial current density [A.m-2]")
# F = pybamm.Parameter("Faraday constant [C.mol-1]")
# c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
#
# c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")
#
# # governing equations
# N = -D * pybamm.grad(c)  # flux
# dcdt = -pybamm.div(N)
# model.rhs = {c: dcdt}
#
# # boundary conditions
# lbc = pybamm.Scalar(0)
# rbc = -j / F / D
# model.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}
#
# # initial conditions
# model.initial_conditions = {c: c0}
#
# model.variables = {
#     "Concentration [mol.m-3]": c,
#     "Surface concentration [mol.m-3]": pybamm.surf(c),
#     "Flux [mol.m-2.s-1]": N,
# }
#
#
# param = pybamm.ParameterValues(
#     {
#         "Particle radius [m]": 10e-6,
#         "Diffusion coefficient [m2.s-1]": 3.9e-14,
#         "Interfacial current density [A.m-2]": 1.4,
#         "Faraday constant [C.mol-1]": 96485,
#         "Initial concentration [mol.m-3]": 2.5e4,
#     }
# )
#
# r = pybamm.SpatialVariable(
#     "r", domain=["negative particle"], coord_sys="spherical polar"
# )
# geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}
#
# param.process_model(model)
# param.process_geometry(geometry)
#
# submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
# var_pts = {r: 20}
# mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
#
# spatial_methods = {"negative particle": pybamm.FiniteVolume()}
# disc = pybamm.Discretisation(mesh, spatial_methods)
# disc.process_model(model)
#
# # solve
# solver = pybamm.ScipySolver()
# t = np.linspace(0, 3600, 600)
# solution = solver.solve(model, t)
#
# # post-process, so that the solution can be called at any time t or space r
# # (using interpolation)
# c = solution["Concentration [mol.m-3]"]
# c_surf = solution["Surface concentration [mol.m-3]"]
#
# # plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
#
# ax1.plot(solution.t, c_surf(solution.t))
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Surface concentration [mol.m-3]")
#
# r = mesh["negative particle"].nodes  # radial position
# time = 1000  # time in seconds
# ax2.plot(r * 1e6, c(t=time, r=r), label=f"t={time}[s]")
# ax2.set_xlabel("Particle radius [microns]")
# ax2.set_ylabel("Concentration [mol.m-3]")
# ax2.legend()
#
# plt.tight_layout()
# plt.show()

# # --------comparing full and reduced-order models--------------------
# full_model = pybamm.BaseModel(name="full model")
# reduced_model = pybamm.BaseModel(name="reduced model")
# models = [full_model, reduced_model]
# R = pybamm.Parameter("Particle radius [m]")
# D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
# j = pybamm.Parameter("Interfacial current density [A.m-2]")
# F = pybamm.Parameter("Faraday constant [C.mol-1]")
# c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
# c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")
# c_av = pybamm.Variable("Average concentration [mol.m-3]")
#
# # governing equations for full model
# N = -D * pybamm.grad(c)  # flux
# dcdt = -pybamm.div(N)
# full_model.rhs = {c: dcdt}
#
# # governing equations for reduced model
# dc_avdt = -3 * j / R / F
# reduced_model.rhs = {c_av: dc_avdt}
#
# # initial conditions (these are the same for both models)
# full_model.initial_conditions = {c: c0}
# reduced_model.initial_conditions = {c_av: c0}
#
# # boundary conditions (only required for full model)
# lbc = pybamm.Scalar(0)
# rbc = -j / F / D
# full_model.boundary_conditions = {
#     c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}
# }
#
# # full model
# full_model.variables = {
#     "Concentration [mol.m-3]": c,
#     "Surface concentration [mol.m-3]": pybamm.surf(c),
#     "Average concentration [mol.m-3]": pybamm.r_average(c),
# }
#
# # reduced model
# reduced_model.variables = {
#     "Concentration [mol.m-3]": pybamm.PrimaryBroadcast(c_av, "negative particle"),
#     "Surface concentration [mol.m-3]": c_av,  # in this model the surface concentration is just equal to the scalar average concentration
#     "Average concentration [mol.m-3]": c_av,
# }
#
# param = pybamm.ParameterValues(
#     {
#         "Particle radius [m]": 10e-6,
#         "Diffusion coefficient [m2.s-1]": 3.9e-14,
#         "Interfacial current density [A.m-2]": 1.4,
#         "Faraday constant [C.mol-1]": 96485,
#         "Initial concentration [mol.m-3]": 2.5e4,
#     }
# )
#
# # geometry
# r = pybamm.SpatialVariable(
#     "r", domain=["negative particle"], coord_sys="spherical polar"
# )
# geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}
# param.process_geometry(geometry)
#
# # models
# for model in models:
#     param.process_model(model)
#
# # mesh
# submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
# var_pts = {r: 20}
# mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
#
# # discretisation
# spatial_methods = {"negative particle": pybamm.FiniteVolume()}
# disc = pybamm.Discretisation(mesh, spatial_methods)
#
# # process models
# for model in models:
#     disc.process_model(model)
#
# # loop over models to solve
# t = np.linspace(0, 3600, 600)
# solutions = [None] * len(models)  # create list to hold solutions
# for i, model in enumerate(models):
#     solver = pybamm.ScipySolver()
#     solutions[i] = solver.solve(model, t)
#
# # post-process the solution of the full model
# c_full = solutions[0]["Concentration [mol.m-3]"]
# c_av_full = solutions[0]["Average concentration [mol.m-3]"]
#
#
# # post-process the solution of the reduced model
# c_reduced = solutions[1]["Concentration [mol.m-3]"]
# c_av_reduced = solutions[1]["Average concentration [mol.m-3]"]
#
# # plot
# r = mesh["negative particle"].nodes  # radial position
#
#
# def plot(t):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
#
#     # Plot concetration as a function of r
#     ax1.plot(r * 1e6, c_full(t=t, r=r), label="Full Model")
#     ax1.plot(r * 1e6, c_reduced(t=t, r=r), label="Reduced Model")
#     ax1.set_xlabel("Particle radius [microns]")
#     ax1.set_ylabel("Concentration [mol.m-3]")
#     ax1.legend()
#
#     # Plot average concentration over time
#     t_hour = np.linspace(0, 3600, 600)  # plot over full hour
#     c_min = c_av_reduced(t=3600) * 0.98  # minimum axes limit
#     c_max = param["Initial concentration [mol.m-3]"] * 1.02  # maximum axes limit
#
#     ax2.plot(t_hour, c_av_full(t=t_hour), label="Full Model")
#     ax2.plot(t_hour, c_av_reduced(t=t_hour), label="Reduced Model")
#     ax2.plot([t, t], [c_min, c_max], "k--")  # plot line to track time
#     ax2.set_xlabel("Time [s]")
#     ax2.set_ylabel("Average concentration [mol.m-3]")
#     ax2.legend()
#
#     plt.tight_layout()
#     plt.show()
#
# t   = np.linspace(0,3600,1)
# plot(t=t)


# # --------a half cell model-------------------
# model = pybamm.BaseModel()
# phi = pybamm.Variable("Positive electrode potential [V]", domain="positive electrode")
# phi_e_s = pybamm.Variable("Separator electrolyte potential [V]", domain="separator")
# phi_e_p = pybamm.Variable("Positive electrolyte potential [V]", domain="positive electrode")
# phi_e = pybamm.concatenation(phi_e_s, phi_e_p)
#
# c = pybamm.Variable(
#     "Positive particle concentration [mol.m-3]",
#     domain="positive particle",
#     auxiliary_domains={
#         "secondary": "positive electrode",
#     },
# )
#
# F = pybamm.Parameter("Faraday constant [C.mol-1]")
# R = pybamm.Parameter("Molar gas constant [J.mol-1.K-1]")
# T = pybamm.Parameter("Temperature [K]")
#
# a = pybamm.Parameter("Surface area per unit volume [m-1]")
# R_p = pybamm.Parameter("Positive particle radius [m]")
# L_s = pybamm.Parameter("Separator thickness [m]")
# L_p = pybamm.Parameter("Positive electrode thickness [m]")
# A = pybamm.Parameter("Electrode cross-sectional area [m2]")
#
# sigma = pybamm.Parameter("Positive electrode conductivity [S.m-1]")
# kappa = pybamm.Parameter("Electrolyte conductivity [S.m-1]")
# D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
#
# I_app = pybamm.Parameter("Applied current [A]")
# c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
#
# c_surf = pybamm.surf(c)  # get the surface concentration
# inputs = {"Positive particle surface concentration [mol.m-3]": c_surf}
# j0 = pybamm.FunctionParameter(
#     "Positive electrode exchange-current density [A.m-2]", inputs
# )
# U = pybamm.FunctionParameter("Positive electrode OCP [V]", inputs)
#
# j_s = pybamm.PrimaryBroadcast(0, "separator")
# j_p = 2 * j0 * pybamm.sinh((F / 2 / R / T) * (phi - phi_e_p - U))
# j = pybamm.concatenation(j_s, j_p)
#
# # charge conservation equations
# i = -sigma * pybamm.grad(phi)
# i_e = -kappa * pybamm.grad(phi_e)
# model.algebraic = {
#     phi: pybamm.div(i) + a * j_p,
#     phi_e: pybamm.div(i_e) - a * j,
# }
# # particle equations (mass conservation)
# N = -D * pybamm.grad(c)  # flux
# dcdt = -pybamm.div(N)
# model.rhs = {c: dcdt}
#
# # boundary conditions
# model.boundary_conditions = {
#     phi: {
#         "left": (pybamm.Scalar(0), "Neumann"),
#         "right": (-I_app / A / sigma, "Neumann"),
#     },
#     phi_e: {
#         "left": (pybamm.Scalar(0), "Dirichlet"),
#         "right": (pybamm.Scalar(0), "Neumann"),
#     },
#     c: {"left": (pybamm.Scalar(0), "Neumann"), "right": (-j_p / F / D, "Neumann")},
# }
#
# # initial conditions
# inputs = {"Initial concentration [mol.m-3]": c0}
# U_init = pybamm.FunctionParameter("Positive electrode OCP [V]", inputs)
# model.initial_conditions = {phi: U_init, phi_e: 0, c: c0}
#
# model.variables = {
#     "Positive electrode potential [V]": phi,
#     "Electrolyte potential [V]": phi_e,
#     "Positive particle concentration [mol.m-3]": c,
#     "Positive particle surface concentration [mol.m-3]": c_surf,
#     "Average positive particle surface concentration [mol.m-3]": pybamm.x_average(
#         c_surf
#     ),
#     "Positive electrode interfacial current density [A.m-2]": j_p,
#     "Positive electrode OCP [V]": pybamm.boundary_value(U, "right"),
#     "Voltage [V]": pybamm.boundary_value(phi, "right"),
# }
#
# from pybamm import tanh
#
# # both functions will depend on the maximum concentration
# c_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
#
#
# def exchange_current_density(c_surf):
#     k = 6 * 10 ** (-7)  # reaction rate [(A/m2)(m3/mol)**1.5]
#     c_e = 1000  # (constant) electrolyte concentration [mol.m-3]
#     return k * c_e**0.5 * c_surf**0.5 * (c_max - c_surf) ** 0.5
#
#
# def open_circuit_potential(c_surf):
#     stretch = 1.062
#     sto = stretch * c_surf / c_max
#
#     u_eq = (
#         2.16216
#         + 0.07645 * tanh(30.834 - 54.4806 * sto)
#         + 2.1581 * tanh(52.294 - 50.294 * sto)
#         - 0.14169 * tanh(11.0923 - 19.8543 * sto)
#         + 0.2051 * tanh(1.4684 - 5.4888 * sto)
#         + 0.2531 * tanh((-sto + 0.56478) / 0.1316)
#         - 0.02167 * tanh((sto - 0.525) / 0.006)
#     )
#     return u_eq
#
# param = pybamm.ParameterValues(
#     {
#         "Surface area per unit volume [m-1]": 0.15e6,
#         "Positive particle radius [m]": 10e-6,
#         "Separator thickness [m]": 25e-6,
#         "Positive electrode thickness [m]": 100e-6,
#         "Electrode cross-sectional area [m2]": 2.8e-2,
#         "Applied current [A]": 0.9,
#         "Positive electrode conductivity [S.m-1]": 10,
#         "Electrolyte conductivity [S.m-1]": 1,
#         "Diffusion coefficient [m2.s-1]": 1e-13,
#         "Faraday constant [C.mol-1]": 96485,
#         "Initial concentration [mol.m-3]": 25370,
#         "Molar gas constant [J.mol-1.K-1]": 8.314,
#         "Temperature [K]": 298.15,
#         "Maximum concentration in positive electrode [mol.m-3]": 51217,
#         "Positive electrode exchange-current density [A.m-2]": exchange_current_density,
#         "Positive electrode OCP [V]": open_circuit_potential,
#     }
# )
#
# r = pybamm.SpatialVariable(
#     "r",
#     domain=["positive particle"],
#     auxiliary_domains={"secondary": "positive electrode"},
#     coord_sys="spherical polar",
# )
# x_s = pybamm.SpatialVariable("x_s", domain=["separator"], coord_sys="cartesian")
# x_p = pybamm.SpatialVariable(
#     "x_p", domain=["positive electrode"], coord_sys="cartesian"
# )
#
#
# geometry = {
#     "separator": {x_s: {"min": -L_s, "max": 0}},
#     "positive electrode": {x_p: {"min": 0, "max": L_p}},
#     "positive particle": {r: {"min": 0, "max": R_p}},
# }
#
# param.process_model(model)
# param.process_geometry(geometry)
#
# submesh_types = {
#     "separator": pybamm.Uniform1DSubMesh,
#     "positive electrode": pybamm.Uniform1DSubMesh,
#     "positive particle": pybamm.Uniform1DSubMesh,
# }
# var_pts = {x_s: 10, x_p: 20, r: 30}
# mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
#
# spatial_methods = {
#     "separator": pybamm.FiniteVolume(),
#     "positive electrode": pybamm.FiniteVolume(),
#     "positive particle": pybamm.FiniteVolume(),
# }
# disc = pybamm.Discretisation(mesh, spatial_methods)
# disc.process_model(model)
#
# # solve
# solver = pybamm.IDAKLUSolver()
# t_eval = [0, 3600]
# solution = solver.solve(model, t_eval)
#
# # plot
# pybamm.dynamic_plot(
#     solution,
#     [
#         "Positive electrode potential [V]",
#         "Electrolyte potential [V]",
#         "Positive electrode interfacial current density [A.m-2]",
#         "Positive particle surface concentration [mol.m-3]",
#         "Average positive particle surface concentration [mol.m-3]",
#         ["Positive electrode OCP [V]", "Voltage [V]"],
#     ],
# )

# --------creating a simple model for SEI growth-------------------
# A model is defined in six steps:
#
# 1.Initialise model
# 2.Define parameters and variables
# 3.State governing equations
# 4.State boundary conditions
# 5.State initial conditions
# 6.State output variables

# 1
model = pybamm.BaseModel()
# 2
# dimensional parameters
k = pybamm.Parameter("Reaction rate constant [m.s-1]")
L_0 = pybamm.Parameter("Initial thickness [m]")
V_hat = pybamm.Parameter("Partial molar volume [m3.mol-1]")
c_inf = pybamm.Parameter("Bulk electrolyte solvent concentration [mol.m-3]")


def D(cc):
    return pybamm.FunctionParameter(
        "Diffusivity [m2.s-1]", {"Solvent concentration [mol.m-3]": cc}
    )

xi = pybamm.SpatialVariable("xi", domain="SEI layer", coord_sys="cartesian")
c = pybamm.Variable("Solvent concentration [mol.m-3]", domain="SEI layer")
L = pybamm.Variable("SEI thickness [m]")

# 3
# SEI reaction flux
R = k * pybamm.BoundaryValue(c, "left")

# solvent concentration equation
N = -1 / L * D(c) * pybamm.grad(c)
dcdt = (V_hat * R) / L * pybamm.inner(xi, pybamm.grad(c)) - 1 / L * pybamm.div(N)

# SEI thickness equation
dLdt = V_hat * R

model.rhs = {c: dcdt, L: dLdt}

# 4
D_left = pybamm.BoundaryValue(
    D(c), "left"
)  # pybamm requires BoundaryValue(D(c)) and not D(BoundaryValue(c))
grad_c_left = R * L / D_left

c_right = c_inf

model.boundary_conditions = {
    c: {"left": (grad_c_left, "Neumann"), "right": (c_right, "Dirichlet")}
}

# 5
c_init = c_inf
L_init = L_0

model.initial_conditions = {c: c_init, L: L_init}

# 6
model.variables = {
    "SEI thickness [m]": L,
    "SEI growth rate [m]": dLdt,
    "Solvent concentration [mol.m-3]": c,
}

#------------
# define geometry
geometry = pybamm.Geometry(
    {"SEI layer": {xi: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
)


def Diffusivity(cc):
    return cc * 10 ** (-12)


# parameter values (not physically based, for example only!)
param = pybamm.ParameterValues(
    {
        "Reaction rate constant [m.s-1]": 1e-6,
        "Initial thickness [m]": 1e-6,
        "Partial molar volume [m3.mol-1]": 10,
        "Bulk electrolyte solvent concentration [mol.m-3]": 1,
        "Diffusivity [m2.s-1]": Diffusivity,
    }
)

# process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# mesh and discretise
submesh_types = {"SEI layer": pybamm.Uniform1DSubMesh}
var_pts = {xi: 100}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"SEI layer": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = [0, 100]  # solve for 100s
solution = solver.solve(model, t)

# post-process output variables
L_out = solution["SEI thickness [m]"]
c_out = solution["Solvent concentration [mol.m-3]"]

# plot SEI thickness in microns as a function of t in microseconds
# and concentration in mol/m3 as a function of x in microns
L_0_eval = param.evaluate(L_0)
xi = np.linspace(0, 1, 100)  # dimensionless space


def plot(t):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(solution.t, L_out(solution.t) * 1e6)
    ax1.plot(t, L_out(t) * 1e6, "r.")
    ax1.set_ylabel(r"SEI thickness [$\mu$m]")
    ax1.set_xlabel(r"t [s]")

    ax2.plot(xi * L_out(t) * 1e6, c_out(t, xi))
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, L_out(solution.t[-1]) * 1e6)
    ax2.set_ylabel("Solvent concentration [mol.m-3]")
    ax2.set_xlabel(r"x [$\mu$m]")

    plt.tight_layout()
    plt.show()

# 画几个时间点
for t in [0, 10, 20, 30, 40]:
    plot(t)
