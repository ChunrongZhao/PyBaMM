import numpy as np
import matplotlib.pyplot as plt
import pybamm
import os
import pandas as pd
import json
import timeit


def compare_PyBaMM_with_COMSOL_curves():
    C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}
    # load model and geometry
    model = pybamm.lithium_ion.DFN()
    geometry = model.default_geometry
    data_loader = pybamm.DataLoader()

    # load parameters and process model and geometry
    param = model.default_parameter_values
    param.update(
        {
            "Electrode width [m]": 1,
            "Electrode height [m]": 1,
            "Negative electrode conductivity [S.m-1]": 126,
            "Positive electrode conductivity [S.m-1]": 16.6,
            "Current function [A]": "[input]",
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)

    # create mesh
    var_pts = {"x_n": 31, "x_s": 11, "x_p": 31, "r_n": 11, "r_p": 11}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplots_adjust(left=-0.1)
    discharge_curve = plt.subplot(211)
    plt.xlim([0, 26])
    plt.ylim([3.2, 3.9])
    plt.xlabel(r"Discharge Capacity (Ah)")
    plt.ylabel("Voltage (V)")
    plt.title(r"Comsol $\cdots$ PyBaMM $-$")
    voltage_difference_plot = plt.subplot(212)
    plt.xlim([0, 26])
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel(r"Discharge Capacity (Ah)")
    plt.ylabel(r"$\vert V - V_{comsol} \vert$")
    colors = iter(plt.cycler(color="bgrcmyk"))

    # loop over C_rates dict to create plot
    for key, C_rate in C_rates.items():
        # load the comsol results
        comsol_results_path = pybamm.get_parameters_filepath(
            data_loader.get_data(f"comsol_{key}C.json"),
        )

        comsol_variables = json.load(open(comsol_results_path, "rb"))

        comsol_time = np.array(comsol_variables["time"])
        comsol_voltage = np.array(comsol_variables["voltage"])

        # update current density
        current = 24 * C_rate

        # solve model at comsol times
        solver = pybamm.IDAKLUSolver()
        t_eval = [comsol_time[0], comsol_time[-1]]
        solution = solver.solve(model, t_eval, inputs={"Current function [A]": current})
        time_in_seconds = comsol_time
        # discharge capacity
        discharge_capacity = solution["Discharge capacity [A.h]"]
        discharge_capacity_sol = discharge_capacity(time_in_seconds)
        comsol_discharge_capacity = comsol_time * current / 3600

        # extract the voltage
        voltage = solution["Voltage [V]"]
        voltage_sol = voltage(time_in_seconds)

        # calculate the difference between the two solution methods
        end_index = min(len(time_in_seconds), len(comsol_time))
        voltage_difference = np.abs(voltage_sol[0:end_index] - comsol_voltage[0:end_index])

        # plot discharge curves and absolute voltage_difference
        color = next(colors)["color"]
        discharge_curve.plot(
            comsol_discharge_capacity, comsol_voltage, color=color, linestyle=":"
        )
        discharge_curve.plot(
            discharge_capacity_sol,
            voltage_sol,
            color=color,
            linestyle="-",
            label=f"{C_rate} C",
        )
        voltage_difference_plot.plot(
            discharge_capacity_sol[0:end_index], voltage_difference, color=color
        )

    discharge_curve.legend(loc="best")
    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
    )
    plt.show()


def compare_with_experimental_data():
    data_loader = pybamm.DataLoader()
    voltage_data_1C = pd.read_csv(
        f"{data_loader.get_data('Ecker_1C.csv')}", header=None
    ).to_numpy()
    voltage_data_5C = pd.read_csv(
        f"{data_loader.get_data('Ecker_5C.csv')}", header=None
    ).to_numpy()

    # choose DFN
    model = pybamm.lithium_ion.DFN()

    # pick parameters, keeping C-rate as an input to be changed for each solve
    parameter_values = pybamm.ParameterValues("Ecker2015")
    parameter_values.update({"Current function [A]": "[input]"})

    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: int(parameter_values.evaluate(model.param.n.L / 1e-6)),
        var.x_s: int(parameter_values.evaluate(model.param.s.L / 1e-6)),
        var.x_p: int(parameter_values.evaluate(model.param.p.L / 1e-6)),
        var.r_n: int(parameter_values.evaluate(model.param.n.prim.R_typ / 1e-7)),
        var.r_p: int(parameter_values.evaluate(model.param.p.prim.R_typ / 1e-7)),
    }

    sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)

    C_rates = [1, 5]  # C-rates to solve for
    capacity = parameter_values["Nominal cell capacity [A.h]"]
    t_evals = [
        [0, 3800],
        [0, 720],
    ]  # times to return the solution at
    solutions = [None] * len(C_rates)  # empty list that will hold solutions

    # loop over C-rates
    for i, C_rate in enumerate(C_rates):
        current = C_rate * capacity
        sim.solve(
            t_eval=t_evals[i],
            inputs={"Current function [A]": current},
        )
        solutions[i] = sim.solution

    # plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # plot the 1C results
    t_sol = solutions[0]["Time [s]"].entries
    ax1.plot(t_sol, solutions[0]["Voltage [V]"](t_sol))
    ax1.plot(voltage_data_1C[:, 0], voltage_data_1C[:, 1], "o")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]")
    ax1.set_title("1C")
    ax1.legend(["DFN", "Experiment"], loc="best")

    # plot the 5C results
    t_sol = solutions[1]["Time [s]"].entries
    ax2.plot(t_sol, solutions[1]["Voltage [V]"](t_sol))
    ax2.plot(voltage_data_5C[:, 0], voltage_data_5C[:, 1], "o")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Voltage [V]")
    ax2.set_title("5C")
    ax2.legend(["DFN", "Experiment"], loc="best")

    plt.tight_layout()
    plt.show()


def compare_lithium_ion_battery_models():
    # 1. load models
    dfn = pybamm.lithium_ion.DFN()
    spme = pybamm.lithium_ion.SPMe()
    spm = pybamm.lithium_ion.SPM()

    models = {"DFN": dfn, "SPM": spm, "SPMe": spme}
    models["DFN"]

    geometry = {
        "DFN": dfn.default_geometry,
        "SPM": spm.default_geometry,
        "SPMe": spme.default_geometry,
    }
    # 2. process parameters
    param = dfn.default_parameter_values
    param["Current function [A]"] = "[input]"

    for model_name in models.keys():
        param.process_model(models[model_name])
        param.process_geometry(geometry[model_name])

    # 3. mesh geometry
    mesh = {}
    for model_name, model in models.items():
        mesh[model_name] = pybamm.Mesh(
            geometry[model_name], model.default_submesh_types, model.default_var_pts
        )

    # 4. discretise model
    for model_name, model in models.items():
        disc = pybamm.Discretisation(mesh[model_name], model.default_spatial_methods)
        disc.process_model(model)

    # 5. solve model
    timer = pybamm.Timer()
    solutions = {}
    t_eval = [0, 3600]  # time in seconds
    for model_name, model in models.items():
        solver = pybamm.IDAKLUSolver()
        timer.reset()
        solution = solver.solve(model, t_eval, inputs={"Current function [A]": 1})
        print(f"Solved the {model.name} in {timer.time()}")
        solutions[model_name] = solution

    # plot results
    for model_name, model in models.items():
        time_solution = solutions[model_name]["Time [s]"].entries
        time = np.linspace(time_solution[0], time_solution[-1], 100)
        voltage = solutions[model_name]["Voltage [V]"](time)
        plt.plot(time, voltage, lw=2, label=model.name)
    plt.xlabel("Time [s]", fontsize=15)
    plt.ylabel("Voltage [V]", fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

    list_of_solutions = list(solutions.values())
    quick_plot = pybamm.QuickPlot(list_of_solutions)
    quick_plot.dynamic_plot()
    # update parameter values and solve again
    # simulate for shorter time
    t_eval = np.linspace(0, 800, 300)
    for model_name, model in models.items():
        solutions[model_name] = model.default_solver.solve(
            model, t_eval, inputs={"Current function [A]": 3}
        )

    # Plot
    list_of_solutions = list(solutions.values())
    quick_plot = pybamm.QuickPlot(list_of_solutions)
    quick_plot.dynamic_plot()


def compare_particle_diffusion_models():
    particle_options = [
    "Fickian diffusion",
    "uniform profile",
    "quadratic profile",
    "quartic profile",
    ]
    models = [
        pybamm.lithium_ion.DFN(options={"particle": opt}, name=opt)
        for opt in particle_options
    ]

    simulations = []
    for model in models:
        param = model.default_parameter_values
        param["Current function [A]"] = "[input]"
        simulations.append(pybamm.Simulation(model, parameter_values=param))

    t_eval = np.linspace(0, 3600, 72)

    solutions_1C = []
    for sim in simulations:
        sim.solve(t_eval, inputs={"Current function [A]": 0.68})
        solutions_1C.append(sim.solution)
        print(f"Particle model: {sim.model.name}")
        print(f"Solve time: {sim.solution.solve_time}s")

    plt.figure(figsize=(15, 15))
    style = ["k", "r*", "b^", "g--"]
    for i in range(len(models)):
        plt.plot(
            solutions_1C[i]["Time [s]"].entries,
            solutions_1C[i]["Voltage [V]"].entries,
            style[i],
            label=particle_options[i],
        )
    plt.legend()
    plt.title("Model Comparison 1C")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.grid()

    t_eval = np.linspace(0, 1800, 72)
    solutions_2C = []
    for sim in simulations:
        sim.solve(t_eval, inputs={"Current function [A]": 2 * 0.68})
        solutions_2C.append(sim.solution)

    plt.figure(figsize=(15, 15))
    for i in range(len(models)):
        plt.plot(
            solutions_2C[i]["Time [s]"].entries,
            solutions_2C[i]["Voltage [V]"].entries,
            style[i],
            label=particle_options[i],
        )
    plt.legend()
    plt.title("Model Comparison 2C")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.grid()

    t_eval = np.linspace(0, 360, 72)
    solutions_6C = []
    for sim in simulations:
        sim.solve(t_eval, inputs={"Current function [A]": 6 * 0.68})
        solutions_6C.append(sim.solution)

    plt.figure(figsize=(15, 15))
    for i in range(len(models)):
        plt.plot(
            solutions_6C[i]["Time [s]"].entries,
            solutions_6C[i]["Voltage [V]"].entries,
            style[i],
            label=particle_options[i],
        )
    plt.legend()
    plt.title("Model Comparison 6C")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.grid()

    pybamm.dynamic_plot(solutions_6C);


def composite_electrode_particle_model():
    """A composite electrode particle model is developed for (negative) electrodes with two phases, e.g. graphite/silicon in LG M50 battery cells."""
    start = timeit.default_timer()
    model = pybamm.lithium_ion.DFN(
        {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }
    )
    param = pybamm.ParameterValues("Chen2020_composite")

    param.update({"Upper voltage cut-off [V]": 4.5})
    param.update({"Lower voltage cut-off [V]": 2.5})

    param.update(
        {
            "Primary: Maximum concentration in negative electrode [mol.m-3]": 28700,
            "Primary: Initial concentration in negative electrode [mol.m-3]": 23000,
            "Primary: Negative particle diffusivity [m2.s-1]": 5.5e-14,
            "Secondary: Negative particle diffusivity [m2.s-1]": 1.67e-14,
            "Secondary: Initial concentration in negative electrode [mol.m-3]": 277000,
            "Secondary: Maximum concentration in negative electrode [mol.m-3]": 278000,
        }
    )

    # --------single cycle simulations------------------
    C_rate = 0.5
    capacity = param["Nominal cell capacity [A.h]"]
    I_load = C_rate * capacity

    t_eval = [0, 10000]

    param["Current function [A]"] = I_load

    # --------------------------------------------------
    v_si = [0.001, 0.04, 0.1]
    total_am_volume_fraction = 0.75     # total active material volume fraction
    solution = []
    for v in v_si:
        param.update(
            {
                "Primary: Negative electrode active material volume fraction": (1 - v)
                * total_am_volume_fraction,  # primary
                "Secondary: Negative electrode active material volume fraction": v
                * total_am_volume_fraction,
            }
        )
        print(v)
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
        )
        solution.append(sim.solve(t_eval=t_eval))
    stop = timeit.default_timer()
    print("running time: " + str(stop - start) + "s")

    # ---------------------- results -----------------------
    # -----------
    ltype = ["k-", "r--", "b-.", "g:", "m-", "c--", "y-."]
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        V_i = solution[i]["Voltage [V]"].entries
        plt.plot(t_i, V_i, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Voltage [V]")
    plt.legend()

    # --------------
    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        j_n_p1_av = solution[i][
            "X-averaged negative electrode primary interfacial current density [A.m-2]"
        ].entries
        plt.plot(t_i, j_n_p1_av, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Averaged interfacial current density [A/m$^{2}$]")
    plt.legend()
    plt.title("Graphite")

    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        j_n_p2_av = solution[i][
            "X-averaged negative electrode secondary interfacial current density [A.m-2]"
        ].entries
        plt.plot(t_i, j_n_p2_av, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Averaged interfacial current density [A/m$^{2}$]")
    plt.legend()
    plt.title("Silicon")

    # -------------
    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        j_n_p1_Vav = solution[i][
            "X-averaged negative electrode primary volumetric interfacial current density [A.m-3]"
        ].entries
        plt.plot(t_i, j_n_p1_Vav, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Averaged volumetric interfacial current density [A/m$^{3}$]")
    plt.legend()
    plt.title("Graphite")

    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        j_n_p2_Vav = solution[i][
            "X-averaged negative electrode secondary volumetric interfacial current density [A.m-3]"
        ].entries
        plt.plot(t_i, j_n_p2_Vav, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Averaged volumetric interfacial current density [A/m$^{3}$]")
    plt.legend()
    plt.title("Silicon")

    # -------------------
    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        c_s_xrav_n_p1 = solution[i][
            "Average negative primary particle concentration"
        ].entries
        plt.plot(t_i, c_s_xrav_n_p1, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("$c_\mathrm{g}/c_\mathrm{g,max}$")
    plt.legend()
    plt.title("Graphite")

    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        c_s_xrav_n_p2 = solution[i][
            "Average negative secondary particle concentration"
        ].entries
        plt.plot(t_i, c_s_xrav_n_p2, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("$c_\mathrm{si}/c_\mathrm{si,max}$")
    plt.legend()
    plt.title("Silicon")

    # -------------------
    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        ocp_p1 = solution[i][
            "X-averaged negative electrode primary open-circuit potential [V]"
        ].entries
        plt.plot(t_i, ocp_p1, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Equilibruim potential [V]")
    plt.legend()
    plt.title("Graphite")

    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        ocp_p2 = solution[i][
            "X-averaged negative electrode secondary open-circuit potential [V]"
        ].entries
        plt.plot(t_i, ocp_p2, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Equilibruim potential [V]")
    plt.legend()
    plt.title("Silicon")

    plt.figure()
    for i in range(0, len(v_si)):
        t_i = solution[len(v_si) - 1 - i]["Time [s]"].entries / 3600
        ocp_p = solution[len(v_si) - 1 - i][
            "X-averaged positive electrode open-circuit potential [V]"
        ].entries
        plt.plot(
            t_i,
            ocp_p,
            ltype[len(v_si) - 1 - i],
            label="$V_\mathrm{si}=$" + str(v_si[len(v_si) - 1 - i]),
        )
    plt.xlabel("Time [h]")
    plt.ylabel("Equilibrium potential [V]")
    plt.legend()
    plt.title("NMC811")

    # -------------------
    experiment = pybamm.Experiment(
        [
            (
                "Discharge at C/2 until 3.0 V",
                "Rest for 1 hour",
                "Charge at C/2 until 4.2 V",
                "Rest for 1 hour",
            ),
        ]
        * 2
    )

    solution = []
    for v in v_si:
        param.update(
            {
                "Primary: Negative electrode active material volume fraction": (1 - v)
                * total_am_volume_fraction,  # primary
                "Secondary: Negative electrode active material volume fraction": v
                * total_am_volume_fraction,
            }
        )
        print(v)
        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=param,
        )
        solution.append(sim.solve(calc_esoh=False))
    stop = timeit.default_timer()
    print("running time: " + str(stop - start) + "s")

    # ----------------------
    ltype = ["k-", "r--", "b-.", "g:", "m-", "c--", "y-."]
    for i in range(0, len(v_si)):
        t_i = solution[i]["Time [s]"].entries / 3600
        V_i = solution[i]["Voltage [V]"].entries
        plt.plot(t_i, V_i, ltype[i], label="$V_\mathrm{si}=$" + str(v_si[i]))
    plt.xlabel("Time [h]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.show()


def model_coupled_degradation_mechanisms():
    model = pybamm.lithium_ion.DFN(
    {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",  # alias for "SEI porosity change"
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "stress-driven",
    }
    )
    param = pybamm.ParameterValues("OKane2022")
    var_pts = {
        "x_n": 5,  # negative electrode
        "x_s": 5,  # separator
        "x_p": 5,  # positive electrode
        "r_n": 30,  # negative particle
        "r_p": 30,  # positive particle
    }

    cycle_number = 10
    exp = pybamm.Experiment(
        [
            "Hold at 4.2 V until C/100",
            "Rest for 4 hours",
            "Discharge at 0.1C until 2.5 V",  # initial capacity check
            "Charge at 0.3C until 4.2 V",
            "Hold at 4.2 V until C/100",
        ]
        + [
            (
                "Discharge at 1C until 2.5 V",  # ageing cycles
                "Charge at 0.3C until 4.2 V",
                "Hold at 4.2 V until C/100",
            )
        ]
        * cycle_number
        + ["Discharge at 0.1C until 2.5 V"],  # final capacity check
    )
    solver = pybamm.IDAKLUSolver()
    sim = pybamm.Simulation(
        model, parameter_values=param, experiment=exp, solver=solver, var_pts=var_pts
    )
    sol = sim.solve()

    # ------------------------
    Qt = sol["Throughput capacity [A.h]"].entries
    Q_SEI = sol["Loss of capacity to negative SEI [A.h]"].entries
    Q_SEI_cr = sol["Loss of capacity to negative SEI on cracks [A.h]"].entries
    Q_plating = sol["Loss of capacity to negative lithium plating [A.h]"].entries
    Q_side = sol["Total capacity lost to side reactions [A.h]"].entries
    Q_LLI = (
        sol["Total lithium lost [mol]"].entries * 96485.3 / 3600
    )  # convert from mol to A.h
    plt.figure()
    plt.plot(Qt, Q_SEI, label="SEI", linestyle="dashed")
    plt.plot(Qt, Q_SEI_cr, label="SEI on cracks", linestyle="dashdot")
    plt.plot(Qt, Q_plating, label="Li plating", linestyle="dotted")
    plt.plot(Qt, Q_side, label="All side reactions", linestyle=(0, (6, 1)))
    plt.plot(Qt, Q_LLI, label="All LLI")
    plt.xlabel("Throughput capacity [A.h]")
    plt.ylabel("Capacity loss [A.h]")
    plt.legend()
    plt.show()

    # ------------------------
    Qt = sol["Throughput capacity [A.h]"].entries
    LLI = sol["Loss of lithium inventory [%]"].entries
    LAM_neg = sol["Loss of active material in negative electrode [%]"].entries
    LAM_pos = sol["Loss of active material in positive electrode [%]"].entries
    plt.figure()
    plt.plot(Qt, LLI, label="LLI")
    plt.plot(Qt, LAM_neg, label="LAM (negative)")
    plt.plot(Qt, LAM_pos, label="LAM (positive)")
    plt.xlabel("Throughput capacity [A.h]")
    plt.ylabel("Degradation modes [%]")
    plt.legend()
    plt.show()

    # ------------------------
    eps_neg_avg = sol["X-averaged negative electrode porosity"].entries
    eps_neg_sep = sol["Negative electrode porosity"].entries[-1, :]
    eps_neg_CC = sol["Negative electrode porosity"].entries[0, :]
    plt.figure()
    plt.plot(Qt, eps_neg_avg, label="Average")
    plt.plot(Qt, eps_neg_sep, label="Separator", linestyle="dotted")
    plt.plot(Qt, eps_neg_CC, label="Current collector", linestyle="dashed")
    plt.xlabel("Throughput capacity [A.h]")
    plt.ylabel("Negative electrode porosity")
    plt.legend()
    plt.show()


def Doyler_Fuller_Newman_model_with_particle_size_distributions():
    """the extension of the Doyle-Fuller-Newman (DFN) model to include a distribution of particle sizes at every macroscale location
    (e.g. through-cell coordinate within the electrodes."""
    # choose option(s)
    options = {"particle size": "distribution"} # default, single

    # load model
    model = pybamm.lithium_ion.DFN(options=options, name="MP-DFN")

    # Base parameter set (no distribution parameters by default)
    params = pybamm.ParameterValues("Marquis2019")

    # Add distribution parameters to the set, with default values (lognormals)
    params = pybamm.get_size_distribution_parameters(params)

    # load parameter values into simuluation
    sim = pybamm.Simulation(model, parameter_values=params)

    # solve
    sim.solve(t_eval=[0, 3500])

    # plot some variables that depend on particle size
    output_variables = [
        "Negative particle surface concentration distribution [mol.m-3]",
        "Positive particle surface concentration distribution [mol.m-3]",
        "X-averaged negative particle surface concentration distribution [mol.m-3]",
        "Negative area-weighted particle-size distribution [m-1]",
        "Positive area-weighted particle-size distribution [m-1]",
        "Voltage [V]",
    ]

    sim.plot(output_variables=output_variables)

    # The discrete sizes or "bins" used
    R_p = sim.solution["Positive particle sizes [m]"].entries[
        :, 0, 0
    ]  # const in the x and current collector direction
    R_n = sim.solution["Negative particle sizes [m]"].entries[:, 0, 0]

    # The distributions (number, area, and volume-weighted)
    f_a_p = sim.solution[
        "X-averaged positive area-weighted particle-size distribution [m-1]"
    ].entries[:, 0]
    f_num_p = sim.solution[
        "X-averaged positive number-based particle-size distribution [m-1]"
    ].entries[:, 0]
    f_v_p = sim.solution[
        "X-averaged positive volume-weighted particle-size distribution [m-1]"
    ].entries[:, 0]
    f_a_n = sim.solution[
        "X-averaged negative area-weighted particle-size distribution [m-1]"
    ].entries[:, 0]
    f_num_n = sim.solution[
        "X-averaged negative number-based particle-size distribution [m-1]"
    ].entries[:, 0]
    f_v_n = sim.solution[
        "X-averaged negative volume-weighted particle-size distribution [m-1]"
    ].entries[:, 0]

    # plot
    f, axs = plt.subplots(1, 2, figsize=(10, 4))

    # negative electrode
    width_n = (R_n[-1] - R_n[-2]) / 1e-6
    axs[0].bar(
        R_n / 1e-6,
        f_a_n * 1e-6,
        width=width_n,
        alpha=0.3,
        color="tab:blue",
        label="area-weighted",
    )
    axs[0].bar(
        R_n / 1e-6,
        f_num_n * 1e-6,
        width=width_n,
        alpha=0.3,
        color="tab:red",
        label="number-weighted",
    )
    axs[0].bar(
        R_n / 1e-6,
        f_v_n * 1e-6,
        width=width_n,
        alpha=0.3,
        color="tab:green",
        label="volume-weighted",
    )
    axs[0].set_xlim((0, 25))
    axs[0].set_xlabel("Particle size $R_{\mathrm{n}}$ [$\mu$m]", fontsize=12)
    axs[0].set_ylabel("[$\mu$m$^{-1}$]", fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].set_title("Discretized distributions (histograms) in negative electrode")

    # positive electrode
    width_p = (R_p[-1] - R_p[-2]) / 1e-6
    axs[1].bar(
        R_p / 1e-6,
        f_a_p * 1e-6,
        width=width_p,
        alpha=0.3,
        color="tab:blue",
        label="area-weighted",
    )
    axs[1].bar(
        R_p / 1e-6,
        f_num_p * 1e-6,
        width=width_p,
        alpha=0.3,
        color="tab:red",
        label="number-weighted",
    )
    axs[1].bar(
        R_p / 1e-6,
        f_v_p * 1e-6,
        width=width_p,
        alpha=0.3,
        color="tab:green",
        label="volume-weighted",
    )
    axs[1].set_xlim((0, 25))
    axs[1].set_xlabel("Particle size $R_{\mathrm{p}}$ [$\mu$m]", fontsize=12)
    axs[1].set_ylabel("[$\mu$m$^{-1}$]", fontsize=12)
    axs[1].set_title("Positive electrode")
    plt.tight_layout()
    plt.show()

    # ---------Custom size distributions--------
    # Set the area-weighted mean radius to be the reference value from the parameter set
    R_av_p_dim = params["Positive particle radius [m]"]

    # Standard deviation (dimensional)
    sd_p_dim = 0.6 * R_av_p_dim

    # Minimum and maximum particle sizes (dimensional)
    R_min_p = 0
    R_max_p = 3 * R_av_p_dim

    # Set the area-weighted particle-size distribution.
    # Choose a lognormal (but any pybamm function could be used)

    def f_a_dist_p_dim(R):
        return pybamm.lognormal(R, R_av_p_dim, sd_p_dim)

    # Note: the only argument must be the particle size R
    # input params to the dictionary
    distribution_params = {
        "Positive minimum particle radius [m]": R_min_p,
        "Positive maximum particle radius [m]": R_max_p,
        "Positive area-weighted " + "particle-size distribution [m-1]": f_a_dist_p_dim,
    }
    params.update(distribution_params, check_already_exists=False)

    # load parameter values into simulation
    sim_custom = pybamm.Simulation(model, parameter_values=params)

    # solve
    sim_custom.solve(t_eval=[0, 3500])

    # plot
    output_variables = [
        "X-averaged negative area-weighted particle-size distribution [m-1]",
        "X-averaged positive area-weighted particle-size distribution [m-1]",
        "Voltage [V]",
    ]
    quickplot = pybamm.QuickPlot(
        [sim, sim_custom],
        output_variables=output_variables,
        labels=["default lognormals", "custom"],
    )
    quickplot.plot(0)

    # --Compare MP-DFN to MPM and DFN models----------------------
    models = [
        pybamm.lithium_ion.DFN(options={"particle size": "distribution"}, name="MP-DFN"),
        pybamm.lithium_ion.MPM(name="MPM"),
        pybamm.lithium_ion.DFN(name="DFN"),
    ]

    # parameters
    params = pybamm.ParameterValues("Marquis2019")
    params = pybamm.get_size_distribution_parameters(params)

    # experiment
    experiment = pybamm.Experiment(
        [
            "Discharge at 1C for 3450 seconds",
            "Rest for 3600 seconds",
        ]
    )


    # solve
    sims = []
    for model in models:
        sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
        sim.solve()
        sims.append(sim)
    # plot current, voltage
    qp = pybamm.QuickPlot(sims, output_variables=["Current [A]", "Voltage [V]"])
    qp.plot(0)


if __name__ == '__main__':
    # compare_PyBaMM_with_COMSOL_curves()
    # compare_with_experimental_data()
    # compare_lithium_ion_battery_models()
    # compare_particle_diffusion_models()
    # composite_electrode_particle_model()
    # model_coupled_degradation_mechanisms()
    Doyler_Fuller_Newman_model_with_particle_size_distributions()
