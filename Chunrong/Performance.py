import numpy as np
import matplotlib.pyplot as plt
import pybamm
import time
import traceback
import tracemalloc
# ---------understanding the PyBaMM pipeline and performance profile of the Simulation class---
# # load model
# model = pybamm.lithium_ion.DFN()
#
# # Step 1: Parameter replacement
# parameter_values = model.default_parameter_values
# parameter_values.process_model(model)
# geometry = model.default_geometry
# parameter_values.process_geometry(geometry)
#
# # Step 2: Discretisation
# mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
# disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
# disc.process_model(model)
#
# # Step 3: Solver setup
# #    The `set_up` method is normally called by the `solve` method if not called explicitly.
# #    We also need to update `_model_set_up` to include the initial conditions or the solver
# #    will call `set_up` again during `solve`.
# solver = pybamm.IDAKLUSolver()
# solver.set_up(model)
# solver._model_set_up.update(
#     {model: {"initial conditions": model.concatenated_initial_conditions}}
# )
#
# # Step 4: Solver solve
# solution = solver.solve(model, [0, 3600])
#
# # Step 5: Post-processing
# t_evals = np.linspace(0, 3600, 100)
# voltage = solution["Terminal voltage [V]"](t_evals)
#
# # Plot the results
# plt.plot(t_evals, voltage)
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")
# plt.show()

# # load model
# model = pybamm.lithium_ion.DFN()
#
# # Step 1: Parameter replacement
# total_start = time.perf_counter()
# start = time.perf_counter()
# parameter_values = model.default_parameter_values
# parameter_values.process_model(model)
# geometry = model.default_geometry
# parameter_values.process_geometry(geometry)
# end = time.perf_counter()
# print(f"Parameter replacement took {end - start:.3f} seconds")
#
# # Step 2: Discretisation
# start = time.perf_counter()
# mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
# disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
# disc.process_model(model)
# end = time.perf_counter()
# print(f"Discretisation took {end - start:.3f} seconds")
#
# # Step 3: Solver setup
# start = time.perf_counter()
# solver = pybamm.IDAKLUSolver()
# solver.set_up(model)
# solver._model_set_up.update(
#     {model: {"initial conditions": model.concatenated_initial_conditions}}
# )
# end = time.perf_counter()
# print(f"Solver setup took {end - start:.3f} seconds")
#
# # Step 4: Solver solve
# start = time.perf_counter()
# solution = solver.solve(model, [0, 3600])
# end = time.perf_counter()
# print(f"Solver solve took {end - start:.3f} seconds")
#
# # Step 5: Post-processing
# start = time.perf_counter()
# t_evals = np.linspace(0, 3600, 100)
# voltage = solution["Terminal voltage [V]"](t_evals)
# end = time.perf_counter()
# print(f"Post-processing took {end - start:.3f} seconds")
# total_end = time.perf_counter()
# print(f"Total time taken: {total_end - total_start:.3f} seconds")


# pybamm.set_logging_level("INFO")
#
# # load model
# model = pybamm.lithium_ion.DFN()
#
# # Step 1: Parameter replacement
# parameter_values = model.default_parameter_values
# parameter_values.process_model(model)
# geometry = model.default_geometry
# parameter_values.process_geometry(geometry)
#
# # Step 2: Discretisation
# var = pybamm.standard_spatial_vars
# mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
# disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
# disc.process_model(model)
#
# # Step 3 & 4: Solver setup
# solver = pybamm.IDAKLUSolver()
# t_eval = [0, 3600]
# solution = solver.solve(model, t_eval)
#
# # Step 5: Post-processing
# t_interp = np.linspace(0, 3600, 100)
# voltage = solution["Terminal voltage [V]"](t_interp)
#
# pybamm.set_logging_level("WARNING")


# # create simulation
# model = pybamm.lithium_ion.DFN()
# solver = pybamm.IDAKLUSolver()
# sim = pybamm.Simulation(model, solver=solver)
#
# # solve
# solution = sim.solve([0, 3600])
#
# # Post-processing
# t_evals = np.linspace(0, 3600, 100)
# voltage = solution["Terminal voltage [V]"](t_evals)
#
# # Plot the results
# plt.plot(t_evals, voltage)
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")
# plt.show()

# model = pybamm.lithium_ion.DFN()
# solver = pybamm.IDAKLUSolver()
# sim = pybamm.Simulation(model, solver=solver)
#
# # solve
# start = time.perf_counter()
# solution = sim.solve([0, 3600])
# end = time.perf_counter()
# print(f"First solve took {end - start:.3f} seconds")
#
# # solve again
# start = time.perf_counter()
# solution = sim.solve([0, 3600])
# end = time.perf_counter()
# print(f"Second solve took {end - start:.3f} seconds")
#
# # solve again
# start = time.perf_counter()
# solution = sim.solve([0, 3600])
# end = time.perf_counter()
# print(f"Third solve took {end - start:.3f} seconds")
#
# # Post-processing
# start = time.perf_counter()
# t_evals = np.linspace(0, 3600, 100)
# voltage = solution["Terminal voltage [V]"](t_evals)
# end = time.perf_counter()
# print(f"Post-processing took {end - start:.3f} seconds")


# -------Using input parameters to efficiently re-run simulations with different parameters------
# model = pybamm.lithium_ion.SPM()
# parameter_values = model.default_parameter_values
# parameter_values["Current function [A]"] = "[input]"
#
# model = pybamm.BaseModel()
# k = pybamm.InputParameter("k")
# x = pybamm.Variable("x")
# model.rhs = {x: -k * x}
#
# # average_time = 0
# # n = 9
# # model = pybamm.lithium_ion.SPM()
# # solver = pybamm.IDAKLUSolver()
# # params = model.default_parameter_values
# # for current in np.linspace(-1.1, 1.0, n):
# #     time_start = time.perf_counter()
# #     params["Current function [A]"] = current
# #     sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
# #     sol = sim.solve([0, 3600])
# #     t_evals = np.linspace(0, 3600, 100)
# #     voltage = sol["Terminal voltage [V]"](t_evals)
# #     time_end = time.perf_counter()
# #     average_time += time_end - time_start
# # print(f"Average time taken: {average_time / n:.3f} seconds")
# #
# #
# # average_time = 0
# # n = 10
# # model = pybamm.lithium_ion.SPM()
# # solver = pybamm.IDAKLUSolver()
# # params = model.default_parameter_values
# # params["Current function [A]"] = "[input]"
# # sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
# # for current in np.linspace(0.1, 1.0, n):
# #     time_start = time.perf_counter()
# #     sol = sim.solve([0, 3600], inputs={"Current function [A]": current})
# #     t_evals = np.linspace(0, 3600, 100)
# #     voltage = sol["Terminal voltage [V]"](t_evals)
# #     time_end = time.perf_counter()
# #     average_time += time_end - time_start
# # print(f"Average time taken: {average_time / n:.3f} seconds")
#
#
# model = pybamm.lithium_ion.SPM()
# model_params = model.get_parameter_info()
# for param in model_params:
#     if model_params[param][1] == "Parameter":
#         params = model.default_parameter_values
#         original_param = params[param]
#         params[param] = "[input]"
#         sim = pybamm.Simulation(model, parameter_values=params)
#         try:
#             sim.solve([0, 3600], inputs={param: original_param})
#         except Exception as e:
#             print(f"Failed for parameter {param}. Error was {e}")
#             tb = traceback.format_exc()
#             print(tb)

# -------The PyBaMM Solvers---------------------------------------------------
# CasadiSolver
# IDAKLUSolver
# ScipySolver
# IDAKLUJax
# JaxSolver

# first_solve_time = np.zeros((3, 3))
# second_solve_time = np.zeros((3, 3))
# for i, model_cls in enumerate(
#     [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN]
# ):
#     for j, solver_cls in enumerate(
#         [pybamm.CasadiSolver, pybamm.IDAKLUSolver, pybamm.ScipySolver]
#     ):
#         if solver_cls == pybamm.ScipySolver and model_cls == pybamm.lithium_ion.DFN:
#             first_solve_time[i, j] = np.nan
#             second_solve_time[i, j] = np.nan
#             continue
#         sim = pybamm.Simulation(model_cls(), solver=solver_cls())
#         start_time = time.perf_counter()
#         sol = sim.solve([0, 3600])
#         voltage = sol["Terminal voltage [V]"](0)
#         end_time = time.perf_counter()
#         first_solve_time[i, j] = end_time - start_time
#         start_time = time.perf_counter()
#         sol = sim.solve([0, 3600])
#         voltage = sol["Terminal voltage [V]"](0)
#         end_time = time.perf_counter()
#         second_solve_time[i, j] = end_time - start_time
#
#
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# for i, model_cls in enumerate(
#     [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN]
# ):
#     ax[0].plot(first_solve_time[i, :], label=model_cls.__name__)
#     ax[1].plot(second_solve_time[i, :], label=model_cls.__name__)
# ax[0].set_xticks(np.arange(3))
# ax[0].set_xticklabels(["Casadi", "IDAKLU", "Scipy"])
# ax[0].set_ylabel("Time (s)")
# ax[0].set_title("First solve time")
# ax[0].legend()
# ax[1].set_xticks(np.arange(3))
# ax[1].set_xticklabels(["Casadi", "IDAKLU", "Scipy"])
# ax[1].set_ylabel("Time (s)")
# ax[1].set_title("Second solve time")
# ax[1].legend()
# plt.tight_layout()
# plt.show()


# --------------------Interpolate and evaluation points--------------------------------------
# # |-------|-------------------------------|---------------------------|----------|------|-----|
# # 0      0.9                             4.9                         7.3        8.5     9.1   10
#
# sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), solver=pybamm.IDAKLUSolver())
# sol = sim.solve([0, 10])
# print("solution was generated at times", sol.t)
#
# # |-------|-----------------------------|--|--------|-----------------|-----------------|-----|
# # 0      0.9                           4.9 5        5.9               7.3               9     10
# sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), solver=pybamm.IDAKLUSolver())
# sol = sim.solve(t_eval=np.array([0, 5, 10]))
# print("solution was generated at times", sol.t)
#
# sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), solver=pybamm.IDAKLUSolver())
# sol = sim.solve(t_eval=np.array([0, 5, 10]))
# print("solution was generated at times", sol.t)
#
# # |-------|---------*----------------*----|--------------------*------|------*---|------|-----|
# # 0                 2                4                         6             8                10
# sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), solver=pybamm.IDAKLUSolver())
# sol = sim.solve(t_eval=[0, 10], t_interp=[2, 4, 6])
# print("solution was generated at times", sol.t)
#
# parameter_values = pybamm.ParameterValues("Chen2020")
# parameter_values.set_initial_stoichiometries(1)
# experiment = pybamm.step.CRate(0.1, period=10, duration=36000)
# sim = pybamm.Simulation(
#     pybamm.lithium_ion.DFN(),
#     solver=pybamm.IDAKLUSolver(),
#     parameter_values=parameter_values,
#     experiment=experiment,
# )
# sol = sim.solve()
# print(f"Number of internal time steps: {len(sol.t)}")
# t_final = sol["Time [h]"].entries[-1]
# t_data = np.linspace(0, t_final, 1000)
#
# start_time = time.perf_counter()
# sol = sim.solve()
# voltage = sol["Terminal voltage [V]"](t_data)
# end_time = time.perf_counter()
# print(f"Time to solve (no t_interp): {end_time - start_time}s")
#
# start_time = time.perf_counter()
# sol = sim.solve(t_interp=t_data)
# voltage = sol["Terminal voltage [V]"].data
# end_time = time.perf_counter()
# print(f"Time to solve (with t_interp): {end_time - start_time}s")
#
# t_data = np.linspace(0, t_final, 10000)
# start_time = time.perf_counter()
# sol = sim.solve()
# voltage = sol["Terminal voltage [V]"](t_data)
# end_time = time.perf_counter()
# print(f"Time to solve (no t_interp): {end_time - start_time}s")
#
# start_time = time.perf_counter()
# sol = sim.solve(t_interp=t_data)
# voltage = sol["Terminal voltage [V]"].data
# end_time = time.perf_counter()
# print(f"Time to solve (with t_interp): {end_time - start_time}s")

# --------------------Solver tolerances--------------------------------------
# time how long it takes to solve the model with different tolerances
# models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.DFN()]
# model_names = ["SPM", "DFN"]
# atols = [1e-2, 1e-4, 1e-6, 1e-8]
# rtols = [1e-2, 1e-4, 1e-6, 1e-8]
# results = np.zeros((len(models), len(atols), len(rtols)))
# for imodel, model in enumerate(models):
#     for iatol, atol in enumerate(atols):
#         for irtol, rtol in enumerate(rtols):
#             solver = pybamm.IDAKLUSolver(atol=atol, rtol=rtol)
#             sim = pybamm.Simulation(model, solver=solver)
#             sol = sim.solve([0, 3600])
#             start_time = time.perf_counter()
#             sol = sim.solve([0, 3600])
#             end_time = time.perf_counter()
#             results[imodel, iatol, irtol] = end_time - start_time
#
# # plot results in a separate plot for each model
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# for imodel, _model in enumerate(models):
#     ax[imodel].imshow(
#         results[imodel],
#         vmin=results[imodel].min(),
#         vmax=results[imodel].max(),
#         cmap="viridis",
#     )
#     ax[imodel].set_xticks(range(len(rtols)))
#     ax[imodel].set_xticklabels([f"{rtol:.0e}" for rtol in rtols])
#     ax[imodel].set_xlabel("rtol")
#     ax[imodel].set_yticks(range(len(atols)))
#     ax[imodel].set_yticklabels([f"{atol:.0e}" for atol in atols])
#     ax[imodel].set_ylabel("atol")
#     ax[imodel].set_title(model_names[imodel])
#     ax[imodel].text(
#         0,
#         0,
#         f"{results[imodel].min():.4f} s",
#         ha="center",
#         va="center",
#         color="white",
#         fontsize=12,
#     )
#     ax[imodel].text(
#         3,
#         3,
#         f"{results[imodel].max():.4f} s",
#         ha="center",
#         va="center",
#         color="black",
#         fontsize=12,
#     )
#
# plt.tight_layout()
# plt.show()

#
# models = pybamm.lithium_ion.DFN()
# high_tol = 1e-10
# goal_rtol = 1e-4
# goal_atol = 1e-6
# t_eval = np.linspace(0, 3600, 100)
# solver = pybamm.IDAKLUSolver(atol=high_tol, rtol=high_tol)
# sim = pybamm.Simulation(models, solver=solver)
# high_tol_sol = sim.solve([0, 3600])["Voltage [V]"](t_eval)
#
# tols = [1e-2, 1e-4, 1e-6, 1e-8]
# results = np.zeros(len(tols))
# for itol, tol in enumerate(tols):
#     solver = pybamm.IDAKLUSolver(atol=tol, rtol=tol)
#     sim = pybamm.Simulation(models, solver=solver)
#     sol = sim.solve([0, 3600])["Voltage [V]"](t_eval)
#     results[itol] = np.sqrt(
#         np.sum(
#             ((sol - high_tol_sol) / (goal_rtol * np.abs(high_tol_sol) + goal_atol)) ** 2
#         )
#         / len(sol)
#     )
#
# # plot results on a log scale
# plt.figure()
# plt.plot(tols, results, "o-")
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("tolerance")
# plt.ylabel("error norm")
# plt.show()


# --------------Defining output variables to reduce memory usage--------------------------------------
# # model = pybamm.lithium_ion.DFN()
# # print("The model state variables are:", [var.name for var in model.rhs.keys()])
# # sim = pybamm.Simulation(model)
# # sim.build()
# # print(
# #     "The concatenated state vector is a vector of shape",
# #     sim.built_model.concatenated_rhs.shape,
# # )
# # print(model.variables["Positive electrode capacity [A.h]"])
# # # Solve the model, storing the state vector at each time step
# # solution = sim.solve([0, 3600])
# #
# # # Extract the positive electrode capacity using the function $h(y)$ and the stored state vector $y$
# # pos_elec_capacity = solution["Positive electrode capacity [A.h]"]
# #
# model = pybamm.lithium_ion.DFN()
# solver = pybamm.IDAKLUSolver()
# experiment = pybamm.Experiment(
#     [
#         "Discharge at 0.1 A for 1 hour",
#         "Charge at 0.1 A for 1 hour",
#     ]
#     * 100
# )
#
# tracemalloc.start()
# sim = pybamm.Simulation(model, solver=solver, experiment=experiment)
# time_start = time.perf_counter()
# sol = sim.solve()
# t_eval = np.linspace(0, 3600 * 10, 100)
# voltage = sol["Terminal voltage [V]"](t_eval)
# time_end = time.perf_counter()
# print("Time to solve: ", time_end - time_start)
#
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics("lineno")
# total_size = sum(stat.size for stat in top_stats)
# print("Total allocated size: %.1f MB" % (total_size / 10**6))
# tracemalloc.stop()
#
# #
# # model = pybamm.lithium_ion.DFN()
# # solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]"])
# # experiment = pybamm.Experiment(
# #     [
# #         "Discharge at 0.1 A for 1 hour",
# #         "Charge at 0.1 A for 1 hour",
# #     ]
# #     * 100
# # )
# #
# # tracemalloc.start()
# # sim = pybamm.Simulation(model, solver=solver, experiment=experiment)
# # time_start = time.perf_counter()
# # sol = sim.solve()
# # t_eval = np.linspace(0, 3600 * 10, 100)
# # voltage = sol["Voltage [V]"](t_eval)
# # time_end = time.perf_counter()
# # print("Time to solve: ", time_end - time_start)
# #
# # snapshot = tracemalloc.take_snapshot()
# # top_stats = snapshot.statistics("lineno")
# # total_size = sum(stat.size for stat in top_stats)
# # print("Total allocated size: %.1f MB" % (total_size / 10**6))
# # tracemalloc.stop()
#
# try:
#     sol["X-averaged positive particle surface concentration [mol.m-3]"]
# except KeyError as e:
#     print("Error:", e)


# ------------Running many simulations in parallel in OpenMP--------------------------
# solver = pybamm.IDAKLUSolver(options={"num_threads": 2})

# model = pybamm.lithium_ion.DFN()
# params = model.default_parameter_values
# params["Current function [A]"] = "[input]"
# sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
# sim.solve(
#     [0, 3600], inputs=[{"Current function [A]": 1}, {"Current function [A]": 0.5}]
# )
#
# n = 1e3
# current_inputs = [
#     {"Current function [A]": current} for current in np.linspace(0, 0.6, int(n))
# ]
# num_threads_list = [1, 2, 4, 8, 16, 32]
# for num_threads in reversed(num_threads_list):
#     model = pybamm.lithium_ion.DFN()
#     params = model.default_parameter_values
#     params.update(
#         {
#             "Current function [A]": "[input]",
#         }
#     )
#     solver = pybamm.IDAKLUSolver(options={"num_threads": num_threads})
#     sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
#     start_time = time.perf_counter()
#     sol = sim.solve([0, 3600], inputs=current_inputs)
#     end_time = time.perf_counter()
#     print(
#         f"Time taken to solve 1000 DFN simulation for {num_threads} threads: {end_time - start_time:.2f} s"
#     )


n = 1e3
current_inputs = [
    {"Current function [A]": current} for current in np.linspace(0, 0.6, int(n))
]
num_threads_list = [1, 2, 4, 8, 16, 32]
for num_threads in reversed(num_threads_list):
    model = pybamm.lithium_ion.SPM()
    params = model.default_parameter_values
    params.update(
        {
            "Current function [A]": "[input]",
        }
    )
    solver = pybamm.IDAKLUSolver(options={"num_threads": num_threads})
    sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
    start_time = time.perf_counter()
    sol = sim.solve([0, 3600], inputs=current_inputs)
    end_time = time.perf_counter()
    print(
        f"Time taken to solve 1000 SPM simulation for {num_threads} threads: {end_time - start_time:.2f} s"
    )
