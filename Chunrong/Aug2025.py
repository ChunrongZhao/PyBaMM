import pybamm
print(pybamm.__version__)
import numpy as np
import os
# # ------------tutorial 1-------------------
# model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
# sim = pybamm.Simulation(model)
# sim.solve([0, 3600])  # solve for 1 hour
# sim.plot()


# # ------------tutorial 2-------------------
# models  = [pybamm.lithium_ion.SPM(),
#           pybamm.lithium_ion.SPMe(),
#           pybamm.lithium_ion.DFN(),]
#
# sims    = []
#
# for model in models:
#     sim    = pybamm.Simulation(model)
#     sim.solve([0,3600])
#     sims.append(sim)
#
# pybamm.dynamic_plot(sims)


# # ------------tutorial 3-------------------
# model   = pybamm.lithium_ion.DFN()
# sim     = pybamm.Simulation(model)
# sim.solve([0,3600])
# model.variable_names()
# # model.variables.search('electrolyte')
# # output_variables    = ['Electrolyte concentration [mol.m-3]', 'Voltage [V]']
# # sim.plot(output_variables=output_variables)
#
# output_variables = ["Voltage [V]"]
# # sim.plot(output_variables=output_variables)
# # sim.plot(
# #     [
# #         ["Electrode current density [A.m-2]", "Electrolyte current density [A.m-2]"],
# #         "Voltage [V]",
# #     ]
# # )
# # sim.plot_voltage_components()
# sim.plot_voltage_components(split_by_electrode=True)


# # ------------tutorial 4-------------------
# parameter_values = pybamm.ParameterValues("Chen2020")
# # print(parameter_values)
# # print(parameter_values["Electrode height [m]"])
# # parameter_values.search("electrolyte")
# model = pybamm.lithium_ion.DFN()
# sim = pybamm.Simulation(model, parameter_values=parameter_values)
# sim.solve([0, 3600])
# sim.plot()


# # ------------tutorial 5-------------------
# # experiment = pybamm.Experiment(
# #     [
# #         "Discharge at C/10 for 10 hours or until 3.3 V",
# #         "Rest for 1 hour",
# #         "Charge at 1 A until 4.1 V",
# #         "Hold at 4.1 V until 50 mA",
# #         "Rest for 1 hour",
# #     ]
# # )
# #
# # experiment = pybamm.Experiment(
# #     [
# #         (
# #             "Discharge at C/10 for 10 hours or until 3.3 V",
# #             "Rest for 1 hour",
# #             "Charge at 1 A until 4.1 V",
# #             "Hold at 4.1 V until 50 mA",
# #             "Rest for 1 hour",
# #         )
# #     ]
# #     * 3
# #     + [
# #         "Discharge at 1C until 3.3 V",
# #     ]
# # )
# model = pybamm.lithium_ion.DFN()
# # sim = pybamm.Simulation(model, experiment=experiment)
# # sim.solve()
# # # sim.plot()
# # # sim.solution.cycles[0].plot()
# # # pybamm.step.string(
# # #     "Discharge at 1C for 1 hour", period="1 minute", temperature="25oC", tags=["tag1"]
# # # )
# # pybamm.step.current(1, duration="1 hour", termination="2.5 V")  # Step(1, duration=1 hour, termination=2.5 V, direction=Discharge)
# # pybamm.step.string("Discharge at 1A for 1 hour or until 2.5V")
#
# t = np.linspace(0, 1, 60)
# sin_t = 0.5 * np.sin(2 * np.pi * t)
# drive_cycle_power = np.column_stack([t, sin_t])
# experiment = pybamm.Experiment([pybamm.step.power(drive_cycle_power)])
# sim = pybamm.Simulation(model, experiment=experiment)
# sim.solve()
# sim.plot()


# # ------------tutorial 6-------------------
# model = pybamm.lithium_ion.SPMe()
# sim = pybamm.Simulation(model)
# sim.solve([0, 3600])
# # solution = sim.solution
# solution = sim.solve([0, 3600])
# t = solution["Time [s]"]
# V = solution["Voltage [V]"]
# # print(V.entries)
# # print(t.entries)
# print(V([200, 400, 780, 1236]) )  # times in seconds
#
# # sim.save("SPMe.pkl")
# # sim2 = pybamm.load("SPMe.pkl")
# # sim2.plot()
#
# sol = sim.solution
# # sol.save("SPMe_sol.pkl")
# # sol2 = pybamm.load("SPMe_sol.pkl")
# # sol2.plot()
#
# sol.save_data("sol_data.pkl", ["Current [A]", "Voltage [V]"])
# sol.save_data("sol_data.csv", ["Current [A]", "Voltage [V]"], to_format="csv")
# # matlab needs names without spaces
# sol.save_data(
#     "sol_data.mat",
#     ["Current [A]", "Voltage [V]"],
#     to_format="matlab",
#     short_names={"Current [A]": "I", "Voltage [V]": "V"},
# )
#
# os.remove("SPMe.pkl")
# os.remove("SPMe_sol.pkl")
# os.remove("sol_data.pkl")
# os.remove("sol_data.csv")
# os.remove("sol_data.mat")

# # ------------tutorial 7-------------------
# options = {"thermal": "lumped"}
# model = pybamm.lithium_ion.SPMe(options=options)  # loading in options
#
# sim = pybamm.Simulation(model)
# sim.solve([0, 3600])
# sim.plot(
#     ["Cell temperature [K]", "Total heating [W.m-3]", "Current [A]", "Voltage [V]"]
# )

# # ------------tutorial 8-------------------
# model = pybamm.lithium_ion.DFN()
# param = model.default_parameter_values
# param["Lower voltage cut-off [V]"] = 3.6
#
# solver_baseline = pybamm.IDAKLUSolver(rtol=1e-4, atol=1e-6)
# solver_high_accuracy = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-12)
#
# # create simulations
# sim_baseline = pybamm.Simulation(model, parameter_values=param, solver=solver_baseline)
# sim_high_accuracy = pybamm.Simulation(
#     model, parameter_values=param, solver=solver_high_accuracy
# )
#
# # solve
# sim_baseline.solve([0, 3600])
# print(f"Baseline mode solve time: {sim_baseline.solution.solve_time}")
# sim_high_accuracy.solve([0, 3600])
# print(f"High accuracy mode solve time: {sim_high_accuracy.solution.solve_time}")
#
# # plot solutions
# pybamm.dynamic_plot(
#     [sim_baseline, sim_high_accuracy],
#     labels=["Baseline", "High accuracy"],
# )


# ------------tutorial 9-------------------
# model = pybamm.lithium_ion.SPMe()
# print(model.default_var_pts)
# create our dictionary
# var_pts = {
#     "x_n": 10,  # negative electrode
#     "x_s": 10,  # separator
#     "x_p": 10,  # positive electrode
#     "r_n": 10,  # negative particle
#     "r_p": 10,  # positive particle
# }
# sim = pybamm.Simulation(model, var_pts=var_pts)
# sim.solve([0, 3600])
# sim.plot()

npts = [4, 8, 16, 32, 64]
# choose model and parameters
model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues("Ecker2015")

# loop over number of mesh points
solutions = []
for N in npts:
    var_pts = {
        "x_n": N,  # negative electrode
        "x_s": N,  # separator
        "x_p": N,  # positive electrode
        "r_n": N,  # negative particle
        "r_p": N,  # positive particle
    }
    sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
    sim.solve([0, 3600])
    solutions.append(sim.solution)

pybamm.dynamic_plot(solutions, ["Voltage [V]"], labels=npts)
