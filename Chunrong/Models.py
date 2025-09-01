import numpy as np
import matplotlib.pyplot as plt
import pybamm
import os
import pandas as pd
import json
import timeit
import scipy.interpolate as interp
# -------------------------------DFN model--------------------------------------
# 外部输入：
#  ├─ 外部电流 I(t)      ← 充放电工况
#  ├─ 温度 T             ← 环境/热耦合
#  └─ 初始状态 (c_s0, c_e0)
#
# -------------------------------------------------
# 1. 电荷守恒 (Charge conservation)
# -------------------------------------------------
#  固相电子电流:   I - i_e = -σ ∂φ_s/∂x
#  液相离子电流:   i_e = κ_e(...) [ -∂φ_e/∂x + 浓度梯度项 ]
#
#  结果：得到 φ_s(x), φ_e(x) 和 i_e(x)
#
# -------------------------------------------------
# 2. 质量守恒 (Mass conservation)
# -------------------------------------------------
#  电解液浓度:   ∂c_e/∂t = -∂N_e/∂x + (1/F)∂i_e/∂x
#    N_e = -D_e ∂c_e/∂x + (t⁺/F) i_e
#
#  固相颗粒浓度: ∂c_s/∂t = -(1/r²) ∂(r² N_s)/∂r
#    N_s = -D_s ∂c_s/∂r
#
#  结果：得到 c_e(x,t), c_s(r,t)
#
# -------------------------------------------------
# 3. 界面反应动力学 (Electrochemical kinetics)
# -------------------------------------------------
#  Butler–Volmer:
#    j = 2 j₀ sinh( Fη / 2RT )
#
#  交换电流密度:
#    j₀ ∝ c_e^(1/2) · c_s^(1/2) · (1-c_s)^(1/2)
#
#  过电位:
#    η = φ_s - φ_e - U(c_s|表面)
#
#  结果：得到反应电流 j(x,t)
#
# -------------------------------------------------
# 4. 耦合关系 (Coupling)
# -------------------------------------------------
#  ├─ j 反馈到电荷守恒 (a·j 出现在 ∂i_e/∂x 方程)
#  ├─ j 决定固相边界通量 (N_s|r=R = j/F)
#  ├─ j 决定电解液锂离子源项 ((1/F)∂i_e/∂x)
#
# -------------------------------------------------
# 5. 边界条件 (Boundary conditions)
# -------------------------------------------------
#  ├─ 电流：i_e,n|x=0=0, i_e,p|x=L=0
#  ├─ 浓度：相邻区域连续
#  ├─ 颗粒：r=0 对称，r=R 表面通量 = j/F
#  └─ 电位：φ_s,neg(集流体)=0 (参考电位)
#
# -------------------------------------------------
# 6. 输出结果
# -------------------------------------------------
#  ├─ 电池端电压 V(t) = φ_s,p|x=L - φ_s,n|x=0
#  ├─ 空间分布：c_e(x,t), c_s(r,t), φ_s(x), φ_e(x)
#  ├─ 反应分布：j(x,t)
#  └─ 性能指标：极化、电压曲线、SOC、降解趋势
# --------------------------------------------------------------------------

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


def electrode_state_of_health():
    spm = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(
        [
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
            "Discharge at 1C until 2.8V",
            "Hold at 2.8V until C/50",
        ]
    )
    parameter_values = pybamm.ParameterValues("Mohtat2020")

    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)
    spm_sol = sim.solve()
    spm_sol.plot(
        [
            "Voltage [V]",
            "Current [A]",
            "Negative electrode stoichiometry",
            "Positive electrode stoichiometry",
        ]
    )

    param = pybamm.LithiumIonParameters()

    Vmin = 2.8
    Vmax = 4.2
    Q_n = parameter_values.evaluate(param.n.Q_init)
    Q_p = parameter_values.evaluate(param.p.Q_init)
    Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)

    U_n = param.n.prim.U
    U_p = param.p.prim.U
    T_ref = param.T_ref

    # First we solve for x_100 and y_100

    model = pybamm.BaseModel()

    x_100 = pybamm.Variable("x_100")
    y_100 = (Q_Li - x_100 * Q_n) / Q_p

    y_100_min = 1e-10

    x_100_upper_limit = (Q_Li - y_100_min * Q_p) / Q_n

    model.algebraic = {x_100: U_p(y_100, T_ref) - U_n(x_100, T_ref) - Vmax}

    model.initial_conditions = {x_100: x_100_upper_limit}

    model.variables = {"x_100": x_100, "y_100": y_100}

    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sol = sim.solve([0])

    x_100 = sol["x_100"].data[0]
    y_100 = sol["y_100"].data[0]

    for var in ["x_100", "y_100"]:
        print(var, ":", sol[var].data[0])

    # Based on the calculated values for x_100 and y_100 we solve for x_0
    model = pybamm.BaseModel()

    x_0 = pybamm.Variable("x_0")
    Q = Q_n * (x_100 - x_0)
    y_0 = y_100 + Q / Q_p

    model.algebraic = {x_0: U_p(y_0, T_ref) - U_n(x_0, T_ref) - Vmin}
    model.initial_conditions = {x_0: 0.1}

    model.variables = {
        "Q": Q,
        "x_0": x_0,
        "y_0": y_0,
    }

    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sol = sim.solve([0])


    for var in ["Q", "x_0", "y_0"]:
        print(var, ":", sol[var].data[0])

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

    inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}

    esoh_sol = esoh_solver.solve(inputs)

    for var in ["x_100", "y_100", "Q", "x_0", "y_0"]:
        print(var, ":", esoh_sol[var])

    t = spm_sol["Time [h]"].data
    x_spm = spm_sol["Negative electrode stoichiometry"].data
    y_spm = spm_sol["Positive electrode stoichiometry"].data

    x_0 = esoh_sol["x_0"].data * np.ones_like(t)
    y_0 = esoh_sol["y_0"].data * np.ones_like(t)
    x_100 = esoh_sol["x_100"].data * np.ones_like(t)
    y_100 = esoh_sol["y_100"].data * np.ones_like(t)

    fig, axes = plt.subplots(1, 2)

    axes[0].plot(t, x_spm, "b")
    axes[0].plot(t, x_0, "k:")
    axes[0].plot(t, x_100, "k:")
    axes[0].set_ylabel("x")

    axes[1].plot(t, y_spm, "r")
    axes[1].plot(t, y_0, "k:")
    axes[1].plot(t, y_100, "k:")
    axes[1].set_ylabel("y")

    for k in range(2):
        axes[k].set_xlim([t[0], t[-1]])
        axes[k].set_ylim([0, 1])
        axes[k].set_xlabel("Time [h]")

    fig.tight_layout()

    all_parameter_sets = [
        k
        for k, v in pybamm.parameter_sets.items()
        if v["chemistry"] == "lithium_ion"
        and k
        not in [
            "Xu2019",
            "Chen2020_composite",
            "Ecker2015_graphite_halfcell",
            "OKane2022_graphite_SiOx_halfcell",
        ]
    ]


    def solve_esoh_sweep_QLi(parameter_set, param):
        parameter_values = pybamm.ParameterValues(parameter_set)

        # Vmin = parameter_values["Lower voltage cut-off [V]"]
        # Vmax = parameter_values["Upper voltage cut-off [V]"]
        Vmin = parameter_values["Open-circuit voltage at 0% SOC [V]"]
        Vmax = parameter_values["Open-circuit voltage at 100% SOC [V]"]

        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)

        Q = parameter_values.evaluate(param.Q)
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param, known_value="cell capacity"
        )
        inputs = {"V_max": Vmax, "V_min": Vmin, "Q": Q, "Q_n": Q_n, "Q_p": Q_p}
        sol_init_Q = esoh_solver.solve(inputs)

        Q_Li_init = parameter_values.evaluate(param.Q_Li_particles_init)
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
        inputs = {"V_max": Vmax, "V_min": Vmin, "Q_Li": Q_Li_init, "Q_n": Q_n, "Q_p": Q_p}
        sol_init_QLi = esoh_solver.solve(inputs)

        Q_Li_sweep = np.linspace(1e-6, Q_n + Q_p)
        sweep = {}
        variables = ["Q_Li", "x_0", "x_100", "y_0", "y_100", "Q"]
        for var in variables:
            sweep[var] = []

        for Q_Li in Q_Li_sweep:
            inputs["Q_Li"] = Q_Li
            try:
                sol = esoh_solver.solve(inputs)
                for var in variables:
                    sweep[var].append(sol[var])
            except (ValueError, pybamm.SolverError):
                pass

        return sweep, sol_init_QLi, sol_init_Q


    for parameter_set in ["Chen2020"]:
        sweep, sol_init_QLi, sol_init_Q = solve_esoh_sweep_QLi(parameter_set, param)

    def plot_sweep(sweep, sol_init, sol_init_Q, parameter_set):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        parameter_values = pybamm.ParameterValues(parameter_set)
        parameter_values.evaluate(param.n.Q_init)
        parameter_values.evaluate(param.p.Q_init)
        # Plot min/max stoichimetric limits, including the value with the given Q_Li
        for i, ks in enumerate([["x_0", "x_100"], ["y_0", "y_100"], ["Q"]]):
            ax = axes.flat[i]
            for j, k in enumerate(ks):
                if i == 0 and j == 0:
                    label1 = "Stoichiometric envelope"
                    label2 = "Calculation from cyclable lithium"
                    label3 = "Calculation from cell capacity"
                else:
                    label1 = label2 = label3 = None
                ax.plot(sweep["Q_Li"], sweep[k], "b-", label=label1)
                ax.axhline(sol_init_QLi[k], c="k", linestyle="--", label=label2)
                ax.axhline(sol_init_Q[k], c="r", linestyle="--", label=label3)
            ax.set_xlabel("Cyclable lithium [A.h]")
            ax.set_ylabel(ks[0][0])
            ax.set_xlim([np.min(sweep["Q_Li"]), np.max(sweep["Q_Li"])])
            ax.axvline(sol_init_QLi["Q_Li"], c="k", linestyle="--")
            ax.axvline(sol_init_Q["Q_Li"], c="r", linestyle="--")
            # Plot capacities of electrodes
            # ax.axvline(Qn,c="b",linestyle="--")
            # ax.axvline(Qp,c="r",linestyle="--")
        axes[-1].set_ylabel("Cell capacity [A.h]")

        # Plot initial values of stoichometries
        parameter_values.evaluate(param.n.prim.sto_init_av)
        parameter_values.evaluate(param.p.prim.sto_init_av)
        # axes[0].axhline(sto_n_init,c="g",linestyle="--")
        # axes[1].axhline(sto_p_init,c="g",linestyle="--")

        axes[1].set_title(parameter_set)
        fig.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        return fig, axes


    plot_sweep(sweep, sol_init_QLi, sol_init_Q, "Chen2020")

    # Skip the MSMR example parameter set since we need to set up the ESOH solver differently
    all_parameter_sets.remove("MSMR_Example")
    # Loop over all parameter sets and solve the ESOH problem
    for parameter_set in all_parameter_sets:
        print(parameter_set)
        try:
            sweep, sol_init_QLi, sol_init_Q = solve_esoh_sweep_QLi(parameter_set, param)
            fig, axes = plot_sweep(sweep, sol_init_QLi, sol_init_Q, parameter_set)
        except ValueError:
            pass
    plt.show()


def simulate_graded_electrode():

    model = pybamm.lithium_ion.DFN()
    parameter_values = pybamm.ParameterValues("Chen2020")

    L_n = parameter_values["Negative electrode thickness [m]"]
    L_s = parameter_values["Separator thickness [m]"]
    L_p = parameter_values["Positive electrode thickness [m]"]

    eps_n_0 = parameter_values["Negative electrode porosity"]
    eps_p_0 = parameter_values["Positive electrode porosity"]

    eps_ns = [
        eps_n_0,
        lambda x: eps_n_0 * (1.1 - 0.2 * (x / L_n)),
        lambda x: eps_n_0 * (0.9 + 0.2 * (x / L_n)),
    ]

    eps_ps = [
        eps_p_0,
        lambda x: eps_p_0 * (0.9 - 0.2 / L_p * (L_n + L_s) + 0.2 * (x / L_p)),
        lambda x: eps_p_0 * (1.1 + 0.2 / L_p * (L_n + L_s) - 0.2 * (x / L_p)),
    ]

    solutions = []

    experiment = pybamm.Experiment(["Discharge at 3C until 2.5 V"])

    for eps_n, eps_p in zip(eps_ns, eps_ps, strict=False):
        parameter_values["Negative electrode porosity"] = eps_n
        parameter_values["Positive electrode porosity"] = eps_p
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        sol = sim.solve()
        solutions.append(sol)

    pybamm.dynamic_plot(
        solutions,
        labels=[
            "Constant porosity",
            "Low porosity at separator",
            "High porosity at separator",
        ],
    )
    pybamm.dynamic_plot(
        solutions,
        output_variables=["Negative electrode porosity", "Positive electrode porosity"],
        labels=[
            "Constant porosity",
            "Low porosity at separator",
            "High porosity at separator",
        ],
        )


def half_cell_models():
    # model
    model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
    # parameters
    param_nmc = pybamm.ParameterValues("Xu2019")
    # experiment
    exp_slow = pybamm.Experiment(
        ["Discharge at C/25 until 3.5 V", "Charge at C/25 until 4.2 V"]
    )
    # simulation
    sim1 = pybamm.Simulation(model, parameter_values=param_nmc, experiment=exp_slow)
    # solve
    sol1 = sim1.solve()
    t = sol1["Time [s]"].entries
    V = sol1["Voltage [V]"].entries
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.show()

    exp_fast = pybamm.Experiment(
        ["Discharge at 1C until 3.5 V", "Charge at 1C until 4.2 V"]
    )
    sim2 = pybamm.Simulation(model, parameter_values=param_nmc, experiment=exp_fast)
    sol2 = sim2.solve()
    t = sol2["Time [s]"].entries
    V = sol2["Voltage [V]"].entries
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.show()

    model_with_degradation = pybamm.lithium_ion.DFN(
        {
            "working electrode": "positive",
            "SEI": "reaction limited",  # SEI on both electrodes
            "SEI porosity change": "true",
            "particle mechanics": "swelling and cracking",
            "SEI on cracks": "true",
            "lithium plating": "partially reversible",
            "lithium plating porosity change": "true",  # alias for "SEI porosity change"
        }
    )
    param_GrSi = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
    param_GrSi.update({"SEI reaction exchange current density [A.m-2]": 1.5e-07})
    var_pts = {"x_n": 1, "x_s": 5, "x_p": 7, "r_n": 1, "r_p": 30}
    exp_degradation = pybamm.Experiment(
        ["Charge at 0.3C until 1.5 V", "Discharge at 0.3C until 0.005 V"]
    )
    sim3 = pybamm.Simulation(
        model_with_degradation,
        parameter_values=param_GrSi,
        experiment=exp_degradation,
        var_pts=var_pts,
    )
    sol3 = sim3.solve()
    t = sol3["Time [s]"].entries
    V = sol3["Voltage [V]"].entries
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.show()

    Q_SEI_n = sol3["Loss of capacity to negative SEI [A.h]"].entries
    Q_SEI_p = sol3["Loss of capacity to positive SEI [A.h]"].entries
    Q_SEI_cr = sol3["Loss of capacity to positive SEI on cracks [A.h]"].entries
    Q_pl = sol3["Loss of capacity to positive lithium plating [A.h]"].entries
    plt.figure()
    plt.plot(t, Q_SEI_n, label="Negative SEI")
    plt.plot(t, Q_SEI_p, label="Positive SEI")
    plt.plot(t, Q_SEI_cr, label="SEI on cracks")
    plt.plot(t, Q_pl, label="Lithium plating")
    plt.xlabel("Time [s]")
    plt.ylabel("Loss of lithium inventory [A.h]")
    plt.legend()
    plt.show()

    param_GrSi.update({"SEI reaction exchange current density [A.m-2]": 6e-07})
    sim4 = pybamm.Simulation(
        model_with_degradation,
        parameter_values=param_GrSi,
        experiment=exp_degradation,
        var_pts=var_pts,
    )
    sol4 = sim4.solve()
    t = sol4["Time [s]"].entries
    V = sol4["Voltage [V]"].entries
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.show()

    Q_SEI_n = sol4["Loss of capacity to negative SEI [A.h]"].entries
    Q_SEI_p = sol4["Loss of capacity to positive SEI [A.h]"].entries
    Q_SEI_cr = sol4["Loss of capacity to positive SEI on cracks [A.h]"].entries
    Q_pl = sol4["Loss of capacity to positive lithium plating [A.h]"].entries
    plt.figure()
    plt.plot(t, Q_SEI_n, label="Negative SEI")
    plt.plot(t, Q_SEI_p, label="Positive SEI")
    plt.plot(t, Q_SEI_cr, label="SEI on cracks")
    plt.plot(t, Q_pl, label="Lithium plating")
    plt.xlabel("Time [s]")
    plt.ylabel("Loss of lithium inventory [A.h]")
    plt.legend()
    plt.show()


def hysteresis_state_models():

    # ============================================================
    # 电池迟滞模型 (Hysteresis Models)
    # ------------------------------------------------------------
    # 定义：
    # 在电池建模中，迟滞 (hysteresis) 指同一个 SOC 下，
    # 充电(嵌锂) 与放电(脱锂) 的开路电压 (OCP) 不同，
    # 出现一条“迟滞回线”。这是由于电极材料的
    # 微观结构、界面反应和相变特性造成的。
    #
    # 为了描述这种路径依赖性，通常在 OCP 模型中引入
    # 一个额外的“迟滞状态变量 h”，其取值范围 [-1,1]，
    # 并随充/放电历史演化。
    #
    # 下表总结了三种常见的迟滞建模方法：
    # ============================================================

    # ------------------------------------------------------------
    # 模型: One-state hysteresis (Axen, 2022)
    # 状态变量: h(z,t)，范围 [-1,1]，由 ODE 控制随时间演化
    # 控制方程: dh/dt = γ * (i_vol / (F * c_max * ε)) * (1 - sgn(i_vol) * h)
    # 关键参数: γ_lith, γ_delith (充/放电不同衰减率)
    # 特点:
    #   ✅ 能捕捉路径依赖 (经典迟滞)
    #   ❌ 需要解额外 ODE，计算量大

    # ------------------------------------------------------------
    # 模型: One-state differential capacity hysteresis (Wycisk, 2022)
    # 状态变量: h(z,t)，与 Axen 类似，但引入微分容量依赖
    # 控制方程: dh/dt = (γ * i_surf / Q_cell) * (1 - sgn(i_surf) * h)
    #   其中 γ(z) = Γ(z) * (C_diff(z))^(-x)
    #   且 C_diff(z) = dQ / dU_eq(z)
    # 关键参数: Γ, x (拟合参数), C_diff (微分容量)
    # 特点:
    #   ✅ 更物理合理，衰减率与电极电化学特性相关
    #   ❌ 实现复杂，需要微分容量数据

    # ------------------------------------------------------------
    # 模型: Current Sigmoid (Ai, 2022)
    # 状态变量: h(t) = 1 - 2 * sigmoid(-K * I(t) / Q_cell)
    # 控制方程: U = sigmoid(-K*I/Q_cell) * U_delith + sigmoid(K*I/Q_cell) * U_lith
    # 关键参数: K (拟合参数，通常固定为 100)
    # 特点:
    #   ✅ 简单高效，不需解 ODE
    #   ❌ 纯经验公式，缺乏严格物理意义，历史依赖性弱
    # ------------------------------------------------------------

    models = {
        "current sigmoid": pybamm.lithium_ion.DFN(
            {
                "open-circuit potential": (("single", "current sigmoid"), "single"),
                "particle phases": ("2", "1"),
                "thermal": "lumped",
            }
        ),
        "one-state differential capacity hysteresis": pybamm.lithium_ion.DFN(
            {
                "open-circuit potential": (
                    ("single", "one-state differential capacity hysteresis"),
                    "single",
                ),
                "particle phases": ("2", "1"),
                "thermal": "lumped",
            }
        ),
        "one-state hysteresis": pybamm.lithium_ion.DFN(
            {
                "open-circuit potential": (("single", "one-state hysteresis"), "single"),
                "particle phases": ("2", "1"),
                "thermal": "lumped",
            }
        ),
    }

    parameters = pybamm.ParameterValues("Chen2020_composite")
    parameters.update(
        {
            "Negative current collector density [kg.m-3]": 8933.0,
            "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
            "Negative current collector thermal conductivity [W.m-1.K-1]": 398.0,
            "Negative electrode density [kg.m-3]": 1555.0,
            "Negative electrode specific heat capacity [J.kg-1.K-1]": 1437.0,
            "Negative electrode thermal conductivity [W.m-1.K-1]": 1.58,
            "Separator density [kg.m-3]": 1017.0,
            "Separator specific heat capacity [J.kg-1.K-1]": 1978.0,
            "Separator thermal conductivity [W.m-1.K-1]": 0.34,
            "Positive electrode density [kg.m-3]": 3699.0,
            "Positive electrode specific heat capacity [J.kg-1.K-1]": 1270.0,
            "Positive electrode thermal conductivity [W.m-1.K-1]": 1.04,
            "Positive current collector density [kg.m-3]": 2702.0,
            "Positive current collector specific heat capacity [J.kg-1.K-1]": 903.0,
            "Positive current collector thermal conductivity [W.m-1.K-1]": 238.0,
            "Total heat transfer coefficient [W.m-2.K-1]": 5.0,
        },
        check_already_exists=False,
    )

    # Wycisk parameters
    parameters_wycisk = parameters.copy()
    parameters_wycisk.update(
        {
            "Secondary: Negative particle hysteresis decay rate": 100,
            "Secondary: Negative particle hysteresis switching factor": 10,
            "Secondary: Initial hysteresis state in negative electrode": 1,
        },
        check_already_exists=False,
    )

    # Axen parameters
    parameters_axen = parameters.copy()
    parameters_axen.update(
        {
            "Secondary: Negative particle lithiation hysteresis decay rate": 100,
            "Secondary: Negative particle delithiation hysteresis decay rate": 100,
            "Secondary: Initial hysteresis state in negative electrode": 1,
        },
        check_already_exists=False,
    )

    parameter_values = {
        "current sigmoid": parameters,
        "one-state differential capacity hysteresis": parameters_wycisk,
        "one-state hysteresis": parameters_axen,
    }

    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 1C for 1 hour or until 2.5 V",
                "Rest for 15 minutes",
                "Charge at 1C until 4.2 V",
                "Hold at 4.2 V until 0.05 C",
                "Rest for 15 minutes",
            ),
        ]
    )

    solutions = {}
    for name, model in models.items():
        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=parameter_values[name]
        )
        solutions[name] = sim.solve(calc_esoh=False)


    output_variables = [
        "Current [A]",
        "Negative electrode secondary stoichiometry",
        "Voltage [V]",
        "Volume-averaged cell temperature [K]",
        "Total heating [W.m-3]",
        "Hysteresis electrochemical heating [W.m-3]",
        "X-averaged negative electrode secondary hysteresis state",
        "X-averaged negative electrode secondary open-circuit potential [V]",
    ]

    pybamm.dynamic_plot(
        list(solutions.values()),
        labels=list(solutions.keys()),
        colors=["black", "blue", "red"],
        linestyles=["-", "--", "-."],
        output_variables=output_variables,
    )


def jelly_roll_model():
    N = pybamm.Parameter("Number of winds")
    r0 = pybamm.Parameter("Inner radius")
    eps = (1 - r0) / N  # ratio of sandwich thickness to cell radius
    delta = pybamm.Parameter("Current collector thickness")
    delta_p = delta  # assume same thickness
    delta_n = delta  # assume same thickness
    l = 1 / 2 - delta_p - delta_n  # active material thickness
    sigma_p = pybamm.Parameter("Positive current collector conductivity")
    sigma_n = pybamm.Parameter("Negative current collector conductivity")
    sigma_a = pybamm.Parameter("Active material conductivity")

    # geometry
    r = pybamm.SpatialVariable("radius", domain="cell", coord_sys="cylindrical polar")
    geometry = {"cell": {r: {"min": r0, "max": 1}}}

    # model
    model = pybamm.BaseModel()
    phi_p = pybamm.Variable("Positive potential", domain="cell")
    phi_n = pybamm.Variable("Negative potential", domain="cell")

    A_p = (2 * sigma_a / eps**4 / l) / (delta_p * sigma_p / 2 / np.pi**2)
    A_n = (2 * sigma_a / eps**4 / l) / (delta_n * sigma_n / 2 / np.pi**2)
    model.algebraic = {
        phi_p: pybamm.div((1 / r**2) * pybamm.grad(phi_p)) + A_p * (phi_n - phi_p),
        phi_n: pybamm.div((1 / r**2) * pybamm.grad(phi_n)) - A_n * (phi_n - phi_p),
    }

    model.boundary_conditions = {
        phi_p: {
            "left": (0, "Neumann"),
            "right": (1, "Dirichlet"),
        },
        phi_n: {
            "left": (0, "Dirichlet"),
            "right": (0, "Neumann"),
        },
    }

    model.initial_conditions = {phi_p: 1, phi_n: 0}  # initial guess for solver

    model.variables = {"Negative potential": phi_n, "Positive potential": phi_p}

    params = pybamm.ParameterValues(
        {
            "Number of winds": 20,
            "Inner radius": 0.25,
            "Current collector thickness": 0.05,
            "Positive current collector conductivity": 5e6,
            "Negative current collector conductivity": 5e6,
            "Active material conductivity": 1,
        }
    )
    params.process_geometry(geometry)
    params.process_model(model)

    # mesh
    submesh_types = {"cell": pybamm.Uniform1DSubMesh}
    var_pts = {r: 100}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    # method
    spatial_methods = {"cell": pybamm.FiniteVolume()}
    # discretise
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)

    # solver
    solver = pybamm.CasadiAlgebraicSolver()
    solution = solver.solve(model)

    # extract numerical parameter values
    # Note: this overrides the definition of the `pybamm.Parameter` objects
    N = params.evaluate(N)
    r0 = params.evaluate(r0)
    eps = params.evaluate(eps)
    delta = params.evaluate(delta)

    # post-process homogenised potential
    phi_n = solution["Negative potential"]
    phi_p = solution["Positive potential"]


    def alpha(r):
        return 2 * (phi_n(r=r) - phi_p(r=r))


    def phi_am1(r, theta):
        # careful here - phi always returns a column vector so we need to add a new axis to r to get the right shape
        return alpha(r) * (r[:, np.newaxis] / eps - r0 / eps - delta - theta / 2 / np.pi) / (
            1 - 4 * delta
        ) + phi_p(r=r)


    def phi_am2(r, theta):
        # careful here - phi always returns a column vector so we need to add a new axis to r to get the right shape
        return alpha(r) * (
            r0 / eps + 1 - delta + theta / 2 / np.pi - r[:, np.newaxis] / eps
        ) / (1 - 4 * delta) + phi_p(r=r)

    # define spiral


    def spiral_pos_inner(t):
        return r0 - eps * delta + eps * t / (2 * np.pi)


    def spiral_pos_outer(t):
        return r0 + eps * delta + eps * t / (2 * np.pi)


    def spiral_neg_inner(t):
        return r0 - eps * delta + eps / 2 + eps * t / (2 * np.pi)


    def spiral_neg_outer(t):
        return r0 + eps * delta + eps / 2 + eps * t / (2 * np.pi)


    def spiral_am1_inner(t):
        return r0 + eps * delta + eps * t / (2 * np.pi)


    def spiral_am1_outer(t):
        return r0 - eps * delta + eps / 2 + eps * t / (2 * np.pi)


    def spiral_am2_inner(t):
        return r0 + eps * delta + eps / 2 + eps * t / (2 * np.pi)


    def spiral_am2_outer(t):
        return r0 - eps * delta + eps + eps * t / (2 * np.pi)

    # Setup fine mesh with nr points per layer
    nr = 10
    rr = np.linspace(r0, 1, nr)
    tt = np.arange(0, (N + 1) * 2 * np.pi, 2 * np.pi)
    # N+1 winds of pos c.c.
    r_mesh_pos = np.zeros((len(tt), len(rr)))
    for i in range(len(tt)):
        r_mesh_pos[i, :] = np.linspace(spiral_pos_inner(tt[i]), spiral_pos_outer(tt[i]), nr)
    # N winds of neg, am1, am2
    r_mesh_neg = np.zeros((len(tt) - 1, len(rr)))
    r_mesh_am1 = np.zeros((len(tt) - 1, len(rr)))
    r_mesh_am2 = np.zeros((len(tt) - 1, len(rr)))
    for i in range(len(tt) - 1):
        r_mesh_am2[i, :] = np.linspace(spiral_am2_inner(tt[i]), spiral_am2_outer(tt[i]), nr)
        r_mesh_neg[i, :] = np.linspace(spiral_neg_inner(tt[i]), spiral_neg_outer(tt[i]), nr)
        r_mesh_am1[i, :] = np.linspace(spiral_am1_inner(tt[i]), spiral_am1_outer(tt[i]), nr)
    # Combine and sort
    r_total_mesh = np.vstack((r_mesh_pos, r_mesh_neg, r_mesh_am1, r_mesh_am2))
    r_total_mesh = np.sort(r_total_mesh, axis=None)

    # plot homogenised potential
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(r_total_mesh, phi_n(r=r_total_mesh), "b", label=r"$\phi^-$")
    ax.plot(r_total_mesh, phi_p(r=r_total_mesh), "r", label=r"$\phi^+$")
    for i in range(len(tt)):
        ax.plot(
            r_mesh_pos[i, :],
            phi_p(r=r_mesh_pos[i, :]),
            "k",
            label=r"$\phi$" if i == 0 else "",
        )
    for i in range(len(tt) - 1):
        ax.plot(r_mesh_neg[i, :], phi_n(r=r_mesh_neg[i, :]), "k")
        ax.plot(r_mesh_am1[i, :], phi_am1(r_mesh_am1[i, :], tt[i]), "k")
        ax.plot(r_mesh_am2[i, :], phi_am2(r_mesh_am2[i, :], tt[i]), "k")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi$")
    ax.legend()
    plt.show()


def use_latexify():
    model = pybamm.lithium_ion.SPM()
    model.latexify()
    # model.latexify(newline=False)
    # model.latexify("spm_equations_.png")

    model_spme = pybamm.lithium_ion.SPMe()
    spme_latex = model_spme.latexify(newline=False)
    for line in spme_latex:
        print(line)


def lead_acid_models():
    full = pybamm.lead_acid.Full()
    loqs = pybamm.lead_acid.LOQS()

    # load models
    models = [loqs, full]

    # process parameters
    param = models[0].default_parameter_values
    param["Current function [A]"] = "[input]"
    for model in models:
        param.process_model(model)

    for model in models:
        # load and process default geometry
        geometry = model.default_geometry
        param.process_geometry(geometry)

        # discretise using default settings
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

    timer = pybamm.Timer()
    solutions = {}
    t_eval = np.linspace(0, 3600 * 17, 100)  # time in seconds
    for model in models:
        solver = pybamm.CasadiSolver()
        timer.reset()
        solution = solver.solve(model, t_eval, inputs={"Current function [A]": 1})
        print(f"Solved the {model.name} in {timer.time()}")
        solutions[model] = solution

    for model in models:
        time = solutions[model]["Time [h]"].entries
        voltage = solutions[model]["Voltage [V]"].entries
        plt.plot(time, voltage, lw=2, label=model.name)
    plt.xlabel("Time [h]", fontsize=15)
    plt.ylabel("Voltage [V]", fontsize=15)
    plt.legend(fontsize=15)
    plt.show()


def modelling_lithium_plating():

    # choose models
    plating_options = ["reversible", "irreversible", "partially reversible"]
    models = {
        option: pybamm.lithium_ion.DFN(options={"lithium plating": option}, name=option)
        for option in plating_options
    }

    # pick parameters
    parameter_values = pybamm.ParameterValues("OKane2022")
    parameter_values.update({"Ambient temperature [K]": 268.15})
    parameter_values.update({"Upper voltage cut-off [V]": 4.21})
    # parameter_values.update({"Lithium plating kinetic rate constant [m.s-1]": 1E-9})
    parameter_values.update({"Lithium plating transfer coefficient": 0.5})
    parameter_values.update({"Dead lithium decay constant [s-1]": 1e-4})

    # specify experiments
    pybamm.citations.register("Ren2018")

    s = pybamm.step.string
    experiment_discharge = pybamm.Experiment(
        [
            (
                s("Discharge at C/20 until 2.5 V"),
                s("Rest for 1 hour"),
            ),
        ]
    )

    sims_discharge = []
    for model in models.values():
        # we need to adjust the tolerances to get a good solution for the partially reversible case
        solver = pybamm.IDAKLUSolver(atol=1e-5, rtol=1e-5)
        sim_discharge = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment_discharge,
            solver=solver,
        )
        sol_discharge = sim_discharge.solve(calc_esoh=False)
        model.set_initial_conditions_from(sol_discharge, inplace=True)
        sims_discharge.append(sim_discharge)

    C_rates = ["2C", "1C", "C/2", "C/4", "C/8"]
    experiments = {}
    for C_rate in C_rates:
        experiments[C_rate] = pybamm.Experiment(
            [
                (
                    f"Charge at {C_rate} until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Rest for 1 hour",
                )
            ]
        )

    def define_and_solve_sims(model, experiments, parameter_values):
        sims = {}
        for C_rate, experiment in experiments.items():
            sim = pybamm.Simulation(
                model, experiment=experiment, parameter_values=parameter_values
            )
            sim.solve(calc_esoh=False)
            sims[C_rate] = sim

        return sims


    sims_reversible = define_and_solve_sims(
        models["partially reversible"], experiments, parameter_values
    )

    colors = ["tab:purple", "tab:cyan", "tab:red", "tab:green", "tab:blue"]
    linestyles = ["dashed", "dotted", "solid"]

    param = models["reversible"].param
    A = parameter_values.evaluate(param.L_y * param.L_z)
    F = parameter_values.evaluate(param.F)
    L_n = parameter_values.evaluate(param.n.L)

    currents = [
        "X-averaged negative electrode volumetric interfacial current density [A.m-3]",
        "X-averaged negative electrode lithium plating volumetric interfacial current density [A.m-3]",
        "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
    ]


    def plot(sims):
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        for (C_rate, sim), color in zip(sims.items(), colors, strict=False):
            # Isolate final equilibration phase
            sol = sim.solution.cycles[0].steps[2]

            # Voltage vs time
            t = sol["Time [min]"].entries
            t = t - t[0]
            V = sol["Voltage [V]"].entries
            axs[0, 0].plot(t, V, color=color, linestyle="solid", label=C_rate)

            # Currents
            for current, ls in zip(currents, linestyles, strict=False):
                j = sol[current].entries
                axs[0, 1].plot(t, j, color=color, linestyle=ls)

            # Plated lithium capacity
            Q_Li = sol["Loss of capacity to negative lithium plating [A.h]"].entries
            axs[1, 0].plot(t, Q_Li, color=color, linestyle="solid")

            # Capacity vs time
            Q_main = (
                sol["Negative electrode volume-averaged concentration [mol.m-3]"].entries
                * F
                * A
                * L_n
                / 3600
            )
            axs[1, 1].plot(t, Q_main, color=color, linestyle="solid")

        axs[0, 0].legend()
        axs[0, 0].set_ylabel("Voltage [V]")
        axs[0, 1].set_ylabel("Volumetric interfacial current density [A.m-3]")
        axs[0, 1].legend(("Deintercalation current", "Stripping current", "Total current"))
        axs[1, 0].set_ylabel("Plated lithium capacity [A.h]")
        axs[1, 1].set_ylabel("Intercalated lithium capacity [A.h]")

        for ax in axs.flat:
            ax.set_xlabel("Time [minutes]")

        fig.tight_layout()
        plt.show()
        return fig, axs


    plot(sims_reversible)

    sims_irreversible = define_and_solve_sims(
        models["irreversible"], experiments, parameter_values
    )

    plot(sims_irreversible)

    sims_partially_reversible = define_and_solve_sims(
        models["partially reversible"], experiments, parameter_values
    )

    plot(sims_partially_reversible)


def modelling_lithium_plating_on_composite_electrodes():
    # Defining the composite model
    model = pybamm.lithium_ion.DFN(
        {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }
    )
    # Invoking the Chen2020 parameter set
    parameter_values = pybamm.ParameterValues("Chen2020_composite")

    # Definig temperature/concentration dependent parameters for plating
    def graphite_plating_exchange_current_density_OKane2020(c_e, c_Li, T):
        """
        Exchange-current density for Li plating reaction [A.m-2].
        References
        ----------
        .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
        Monica Marinescu. "Physical origin of the differential voltage minimum associated
        with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
        167, no. 9 (2020): 090540.
        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration [mol.m-3]
        c_Li : :class:`pybamm.Symbol`
            Plated lithium concentration [mol.m-3]
        T : :class:`pybamm.Symbol`
            Temperature [K]
        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density [A.m-2]
        """

        k_plating = pybamm.Parameter(
            "Primary: Lithium plating kinetic rate constant [m.s-1]"
        )

        return pybamm.constants.F * k_plating * c_e


    def graphite_stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
        """
        Exchange-current density for Li stripping reaction [A.m-2].
        References
        ----------
        .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
        Monica Marinescu. "Physical origin of the differential voltage minimum associated
        with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
        167, no. 9 (2020): 090540.
        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration [mol.m-3]
        c_Li : :class:`pybamm.Symbol`
            Plated lithium concentration [mol.m-3]
        T : :class:`pybamm.Symbol`
            Temperature [K]
        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density [A.m-2]
        """

        k_plating = pybamm.Parameter(
            "Primary: Lithium plating kinetic rate constant [m.s-1]"
        )

        return pybamm.constants.F * k_plating * c_Li


    def graphite_SEI_limited_dead_lithium_OKane2022(L_sei):
        """
        Decay rate for dead lithium formation [s-1].
        References
        ----------
        .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
        Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
        and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
        Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
        Parameters
        ----------
        L_sei : :class:`pybamm.Symbol`
            Total SEI thickness [m]
        Returns
        -------
        :class:`pybamm.Symbol`
            Dead lithium decay rate [s-1]
        """

        gamma_0 = pybamm.Parameter("Primary: Dead lithium decay constant [s-1]")
        L_inner_0 = pybamm.Parameter("Primary: Initial inner SEI thickness [m]")
        L_outer_0 = pybamm.Parameter("Primary: Initial outer SEI thickness [m]")
        L_sei_0 = L_inner_0 + L_outer_0

        gamma = gamma_0 * L_sei_0 / L_sei

        return gamma


    def silicon_plating_exchange_current_density_OKane2020(c_e, c_Li, T):
        """
        Exchange-current density for Li plating reaction [A.m-2].
        References
        ----------
        .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
        Monica Marinescu. "Physical origin of the differential voltage minimum associated
        with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
        167, no. 9 (2020): 090540.
        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration [mol.m-3]
        c_Li : :class:`pybamm.Symbol`
            Plated lithium concentration [mol.m-3]
        T : :class:`pybamm.Symbol`
            Temperature [K]
        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density [A.m-2]
        """

        k_plating = pybamm.Parameter(
            "Secondary: Lithium plating kinetic rate constant [m.s-1]"
        )

        return pybamm.constants.F * k_plating * c_e


    def silicon_stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
        """
        Exchange-current density for Li stripping reaction [A.m-2].
        References
        ----------
        .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
        Monica Marinescu. "Physical origin of the differential voltage minimum associated
        with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
        167, no. 9 (2020): 090540.
        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration [mol.m-3]
        c_Li : :class:`pybamm.Symbol`
            Plated lithium concentration [mol.m-3]
        T : :class:`pybamm.Symbol`
            Temperature [K]
        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density [A.m-2]
        """

        k_plating = pybamm.Parameter(
            "Secondary: Lithium plating kinetic rate constant [m.s-1]"
        )

        return pybamm.constants.F * k_plating * c_Li


    def silicon_SEI_limited_dead_lithium_OKane2022(L_sei):
        """
        Decay rate for dead lithium formation [s-1].
        References
        ----------
        .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
        Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
        and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
        Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
        Parameters
        ----------
        L_sei : :class:`pybamm.Symbol`
            Total SEI thickness [m]
        Returns
        -------
        :class:`pybamm.Symbol`
            Dead lithium decay rate [s-1]
        """

        gamma_0 = pybamm.Parameter("Secondary: Dead lithium decay constant [s-1]")
        L_inner_0 = pybamm.Parameter("Secondary: Initial inner SEI thickness [m]")
        L_outer_0 = pybamm.Parameter("Secondary: Initial outer SEI thickness [m]")
        L_sei_0 = L_inner_0 + L_outer_0

        gamma = gamma_0 * L_sei_0 / L_sei

        return gamma


    def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
        c_e, c_s_surf, c_s_max, T
    ):
        """
        Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
        EC:DMC.

        References
        ----------
        .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
        Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
        Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
        Electrochemical Society 167 (2020): 080534.

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration [mol.m-3]
        c_s_surf : :class:`pybamm.Symbol`
            Particle concentration [mol.m-3]
        c_s_max : :class:`pybamm.Symbol`
            Maximum particle concentration [mol.m-3]
        T : :class:`pybamm.Symbol`
            Temperature [K]

        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density [A.m-2]
        """
        m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
        E_r = 35000
        arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

        return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

    # Adding plating parameters to the parameter set
    # plating_parameters =
    parameter_values.update(
        {
            # Plating parameters referred from OKane2022
            "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
            "Primary: Lithium plating kinetic rate constant [m.s-1]": 1e-09,
            "Primary: Exchange-current density for plating [A.m-2]"
            "": graphite_plating_exchange_current_density_OKane2020,
            "Primary: Exchange-current density for stripping [A.m-2]"
            "": graphite_stripping_exchange_current_density_OKane2020,
            "Primary: Initial plated lithium concentration [mol.m-3]": 0.0,
            "Primary: Typical plated lithium concentration [mol.m-3]": 1000.0,
            "Primary: Lithium plating transfer coefficient": 0.65,
            "Primary: Dead lithium decay constant [s-1]": 1e-06,
            "Primary: Dead lithium decay rate [s-1]"
            "": graphite_SEI_limited_dead_lithium_OKane2022,
            "Secondary: Lithium plating kinetic rate constant [m.s-1]": 1e-09,
            "Secondary: Exchange-current density for plating [A.m-2]"
            "": silicon_plating_exchange_current_density_OKane2020,
            "Secondary: Exchange-current density for stripping [A.m-2]"
            "": silicon_stripping_exchange_current_density_OKane2020,
            "Secondary: Initial plated lithium concentration [mol.m-3]": 0.0,
            "Secondary: Typical plated lithium concentration [mol.m-3]": 1000.0,
            "Secondary: Lithium plating transfer coefficient": 0.65,
            "Secondary: Dead lithium decay constant [s-1]": 1e-06,
            "Secondary: Dead lithium decay rate [s-1]"
            "": silicon_SEI_limited_dead_lithium_OKane2022,
        },
        check_already_exists=False,
    )

    # choose models
    composite_options = {
        "particle phases": ("2", "1"),
        "open-circuit potential": (("single", "current sigmoid"), "single"),
    }
    plating_options = ["reversible", "irreversible", "partially reversible"]
    models = {}
    for option in plating_options:
        # Merge the plating option with the default options
        options = composite_options.copy()
        options["lithium plating"] = option

        # Initialize the DFN model with these options
        models[option] = pybamm.lithium_ion.DFN(options=options, name=option)

    # pick parameters
    parameter_values.update({"Ambient temperature [K]": 268.15})
    parameter_values.update({"Upper voltage cut-off [V]": 4.21})
    parameter_values.update(
        {"Lithium plating transfer coefficient": 0.5}, check_already_exists=False
    )
    parameter_values.update(
        {"Dead lithium decay constant [s-1]": 1e-4}, check_already_exists=False
    )
    parameter_values.update(
        {"Primary: Initial inner SEI thickness [m]": 5e-09}, check_already_exists=False
    )
    parameter_values.update(
        {"Primary: Initial outer SEI thickness [m]": 5e-09}, check_already_exists=False
    )
    parameter_values.update(
        {"Secondary: Initial inner SEI thickness [m]": 5e-09}, check_already_exists=False
    )
    parameter_values.update(
        {"Secondary: Initial outer SEI thickness [m]": 5e-09}, check_already_exists=False
    )
    # parameter_values.update({"Lithium plating kinetic rate constant [m.s-1]": 1E-9})

    # specify experiments
    pybamm.citations.register("Ren2018")

    s = pybamm.step.string
    experiment_discharge = pybamm.Experiment(
        [
            (
                "Discharge at C/20 until 2.5 V",
                "Rest for 1 hour",
            )
        ]
    )

    sims_discharge = []
    for model in models.values():
        sim_discharge = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment_discharge
        )
        sol_discharge = sim_discharge.solve(calc_esoh=False)
        model.set_initial_conditions_from(sol_discharge, inplace=True)
        sims_discharge.append(sim_discharge)

    C_rates = ["2C", "1C", "C/2", "C/4", "C/8"]
    experiments = {}
    for C_rate in C_rates:
        experiments[C_rate] = pybamm.Experiment(
            [
                (
                    f"Charge at {C_rate} until 4.2 V",
                    "Hold at 4.2 V until C/20",
                    "Rest for 1 hour",
                )
            ]
        )

    def define_and_solve_sims(model, experiments, parameter_values):
        sims = {}
        for C_rate, experiment in experiments.items():
            sim = pybamm.Simulation(
                model, experiment=experiment, parameter_values=parameter_values
            )
            sim.solve(calc_esoh=False)
            sims[C_rate] = sim

        return sims


    sims_reversible = define_and_solve_sims(
        models["reversible"], experiments, parameter_values
    )


    colors = ["tab:purple", "tab:cyan", "tab:red", "tab:green", "tab:blue"]
    linestyles = ["dashed", "dotted", "solid"]

    param = models["reversible"].param
    A = parameter_values.evaluate(param.L_y * param.L_z)
    F = parameter_values.evaluate(param.F)
    L_n = parameter_values.evaluate(param.n.L)

    currents = [
        "X-averaged negative electrode volumetric interfacial current density [A.m-3]",
        "X-averaged negative electrode lithium plating volumetric interfacial current density [A.m-3]",
        "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
    ]


    def plot(sims):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        ax2 = axs[1, 1].twinx()  # Create a secondary y-axis
        for (C_rate, sim), color in zip(sims.items(), colors, strict=False):
            # Isolate final equilibration phase
            sol = sim.solution.cycles[0].steps[2]

            # Voltage vs time
            t = sol["Time [min]"].entries
            t = t - t[0]
            V = sol["Voltage [V]"].entries
            axs[0, 0].plot(t, V, color=color, linestyle="solid", label=C_rate)

            # Currents
            for current, ls in zip(currents, linestyles, strict=False):
                j = sol[current].entries
                axs[0, 1].plot(t, j, color=color, linestyle=ls)

            # Plated lithium capacity
            Q_Li = sol["Loss of capacity to negative lithium plating [A.h]"].entries
            axs[1, 0].plot(t, Q_Li, color=color, linestyle="solid")

            # Capacity vs time
            Q_prim = (
                sol[
                    "Negative electrode primary volume-averaged concentration [mol.m-3]"
                ].entries
                * F
                * A
                * L_n
                / 3600
            )
            Q_sec = (
                sol[
                    "Negative electrode secondary volume-averaged concentration [mol.m-3]"
                ].entries
                * F
                * A
                * L_n
                / 3600
            )

            axs[1, 1].plot(t, Q_prim, color=color, linestyle="solid")
            ax2.plot(t, Q_sec, color=color, linestyle="dashed")

        axs[0, 0].legend()
        axs[0, 0].set_ylabel("Voltage [V]")
        axs[0, 1].set_ylabel("Volumetric interfacial current density [A.m-3]")
        axs[0, 1].legend(("Deintercalation current", "Stripping current", "Total current"))
        axs[1, 0].set_ylabel("Plated lithium capacity [A.h]")
        axs[1, 1].set_ylabel("Intercalated lithium capacity (Primary) [A.h]")
        ax2.set_ylabel("Intercalated lithium capacity (Secondary) [A.h]")

        for ax in axs.flat:
            ax.set_xlabel("Time [minutes]")

        fig.tight_layout()
        plt.show()
        return fig, axs


    plot(sims_reversible)

    sims_irreversible = define_and_solve_sims(
        models["irreversible"], experiments, parameter_values)

    plot(sims_irreversible)

    sims_partially_reversible = define_and_solve_sims(
        models["partially reversible"], experiments, parameter_values
    )

    plot(sims_partially_reversible)


def many_particle_model():
    model = pybamm.lithium_ion.MPM()
    model.variables.search("X-averaged negative particle concentration")

    c_n_R_dependent = model.variables[
        "X-averaged negative particle concentration distribution [mol.m-3]"
    ]
    c_n_R_dependent.domains

    model.variables.search("X-averaged negative electrode interfacial current density")

    for k, t in model.default_submesh_types.items():
        print(k, "is of type", t.__name__)
    for var, npts in model.default_var_pts.items():
        print(var, "has", npts, "mesh points")

    sim = pybamm.Simulation(model)
    sim.solve(t_eval=[0, 3600])

    # plot some variables that depend on R
    output_variables = [
        "X-averaged negative particle surface concentration distribution [mol.m-3]",
        "X-averaged positive particle surface concentration distribution [mol.m-3]",
        "X-averaged positive electrode interfacial current density distribution [A.m-2]",
        "X-averaged negative area-weighted particle-size distribution [m-1]",
        "X-averaged positive area-weighted particle-size distribution [m-1]",
        "Voltage [V]",
    ]

    sim.plot(output_variables=output_variables)

    # Concentrations as a function of t, r and R
    c_s_n = sim.solution[
        "X-averaged negative particle concentration distribution [mol.m-3]"
    ]
    c_s_p = sim.solution[
        "X-averaged positive particle concentration distribution [mol.m-3]"
    ]

    # r_n, r_p
    r_n = sim.solution["r_n [m]"].entries[:, 0, 0]
    r_p = sim.solution["r_p [m]"].entries[:, 0, 0]
    # dimensional R_n, R_p
    R_n = sim.solution["Negative particle sizes [m]"].entries[:, 0]
    R_p = sim.solution["Positive particle sizes [m]"].entries[:, 0]
    t = sim.solution["Time [s]"].entries


    def plot_concentrations(t):
        f, axs = plt.subplots(1, 2, figsize=(10, 3))
        plot_c_n = axs[0].pcolormesh(
            R_n, r_n, c_s_n(r=r_n, R=R_n, t=t), vmin=0.15, vmax=0.8
        )
        plot_c_p = axs[1].pcolormesh(
            R_p, r_p, c_s_p(r=r_p, R=R_p, t=t), vmin=0.6, vmax=0.95
        )
        axs[0].set_xlabel(r"$R_n$ [$\mu$m]")
        axs[1].set_xlabel(r"$R_p$ [$\mu$m]")
        axs[0].set_ylabel(r"$r_n / R_n$")
        axs[1].set_ylabel(r"$r_p / R_p$")
        axs[0].set_title("Concentration in negative particles [mol.m-3]")
        axs[1].set_title("Concentration in positive particles [mol.m-3]")
        plt.colorbar(plot_c_n, ax=axs[0])
        plt.colorbar(plot_c_p, ax=axs[1])

        plt.show()


    # initial time
    plot_concentrations(t[0])

    # final time
    plot_concentrations(t[-1])

    # Parameter set (no distribution parameters by default)
    params = pybamm.ParameterValues("Marquis2019")

    # Extract the radii values. We will choose these to be the means of our area-weighted distributions
    R_a_n_dim = params["Negative particle radius [m]"]
    R_a_p_dim = params["Positive particle radius [m]"]

    # Standard deviations (dimensional)
    sd_a_n_dim = 0.2 * R_a_n_dim
    sd_a_p_dim = 0.6 * R_a_p_dim

    # Minimum and maximum particle sizes (dimensional)
    R_min_n = 0
    R_min_p = 0
    R_max_n = 2 * R_a_n_dim
    R_max_p = 3 * R_a_p_dim

    # Set the area-weighted particle-size distributions.
    # Choose a lognormal (but any pybamm function could be used)


    def f_a_dist_n_dim(R):
        return pybamm.lognormal(R, R_a_n_dim, sd_a_n_dim)


    def f_a_dist_p_dim(R):
        return pybamm.lognormal(R, R_a_p_dim, sd_a_p_dim)


    # Note: the only argument must be the particle size R

    # input distribution params to the dictionary
    distribution_params = {
        "Negative minimum particle radius [m]": R_min_n,
        "Positive minimum particle radius [m]": R_min_p,
        "Negative maximum particle radius [m]": R_max_n,
        "Positive maximum particle radius [m]": R_max_p,
        "Negative area-weighted " + "particle-size distribution [m-1]": f_a_dist_n_dim,
        "Positive area-weighted " + "particle-size distribution [m-1]": f_a_dist_p_dim,
    }
    params.update(distribution_params, check_already_exists=False)

    sim = pybamm.Simulation(model, parameter_values=params)
    sim.solve(t_eval=[0, 3600])

    sim.plot(output_variables=output_variables)


    # The discrete sizes or "bins" used, and the distributions
    R_p = sim.solution["Positive particle sizes [m]"].entries[
        :, 0
    ]  # const in the current collector direction
    # The distributions
    f_a_p = sim.solution[
        "X-averaged positive area-weighted particle-size distribution [m-1]"
    ].entries[:, 0]
    f_num_p = sim.solution[
        "X-averaged positive number-based particle-size distribution [m-1]"
    ].entries[:, 0]
    f_v_p = sim.solution[
        "X-averaged positive volume-weighted particle-size distribution [m-1]"
    ].entries[:, 0]


    # plot
    width_p = (R_p[-1] - R_p[-2]) / 1e-6
    plt.bar(
        R_p / 1e-6,
        f_a_p * 1e-6,
        width=width_p,
        alpha=0.3,
        color="tab:blue",
        label="area-weighted",
    )
    plt.bar(
        R_p / 1e-6,
        f_num_p * 1e-6,
        width=width_p,
        alpha=0.3,
        color="tab:red",
        label="number-weighted",
    )
    plt.bar(
        R_p / 1e-6,
        f_v_p * 1e-6,
        width=width_p,
        alpha=0.3,
        color="tab:green",
        label="volume-weighted",
    )
    plt.xlim((0, 30))
    plt.xlabel("Particle size $R_{\mathrm{p}}$ [$\mu$m]", fontsize=12)
    plt.ylabel("[$\mu$m$^{-1}$]", fontsize=12)
    plt.legend(fontsize=10)
    plt.title("Discretized distributions (histograms) in positive electrode")
    plt.show()


    # Define standard deviation in negative electrode to vary
    sd_a_p_dim = pybamm.Parameter(
        "Positive electrode area-weighted particle-size standard deviation [m]"
    )

    # Set the area-weighted particle-size distribution


    def f_a_dist_p_dim(R):
        return pybamm.lognormal(R, R_a_p_dim, sd_a_p_dim)


    # input to param dictionary
    distribution_params = {
        "Positive electrode area-weighted particle-size "
        + "standard deviation [m]": "[input]",
        "Positive area-weighted " + "particle-size distribution [m-1]": f_a_dist_p_dim,
    }
    params.update(distribution_params, check_already_exists=False)

    # Experiment with a relaxation period, to see the effect of distribution width
    experiment = pybamm.Experiment(["Discharge at 1 C for 3400 s", "Rest for 1 hours"])

    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    solutions = []
    for sd_a_p in [0.4, 0.6, 0.8]:
        solution = sim.solve(
            inputs={
                "Positive electrode area-weighted particle-size "
                + "standard deviation [m]": sd_a_p * R_a_p_dim
            }
        )
        solutions.append(solution)


    pybamm.dynamic_plot(
        solutions,
        output_variables=output_variables,
        labels=["MPM, sd_a_p=0.4", "MPM, sd_a_p=0.6", "MPM, sd_a_p=0.8"],
    )

    print("The mean of the input lognormal was:", R_a_p_dim)
    print("The means of discretized distributions are:")
    for solution in solutions:
        R = solution["Positive area-weighted mean particle radius [m]"]
        print("Positive area-weighted mean particle radius [m]", R.entries[0])


    print("The standard deviations of the input lognormal were:")
    print(0.4 * R_a_p_dim)
    print(0.6 * R_a_p_dim)
    print(0.8 * R_a_p_dim)
    print("The standard deviations of discretized distributions are:")
    for solution in solutions:
        sd = solution["Positive area-weighted particle-size standard deviation [m]"]
        print("Positive area-weighted particle-size standard deviation [m]", sd.entries[0])

    models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.MPM(), pybamm.lithium_ion.DFN()]

    # solve
    sims = []
    for model in models:
        sim = pybamm.Simulation(model)
        sim.solve(t_eval=[0, 3500])
        sims.append(sim)

    # plot
    pybamm.dynamic_plot(sims)


    model_Fickian = pybamm.lithium_ion.MPM(name="MPM Fickian")
    model_Uniform = pybamm.lithium_ion.MPM(
        name="MPM Uniform", options={"particle": "uniform profile"}
    )

    sim_Fickian = pybamm.Simulation(model_Fickian)
    sim_Uniform = pybamm.Simulation(model_Uniform)

    sim_Fickian.solve(t_eval=[0, 3500])
    sim_Uniform.solve(t_eval=[0, 3500])

    pybamm.dynamic_plot([sim_Fickian, sim_Uniform], output_variables=output_variables)


    # choose model options
    model_cc = pybamm.lithium_ion.MPM(
        options={
            "current collector": "potential pair",
            "dimensionality": 1,
            "particle": "uniform profile",  # to reduce computation time
        }
    )

    # solve
    sim_cc = pybamm.Simulation(model_cc)
    sim_cc.solve(t_eval=[0, 3600])

    # variables to plot
    output_variables = [
        "X-averaged negative particle surface concentration distribution [mol.m-3]",
        "X-averaged positive particle surface concentration distribution [mol.m-3]",
        "X-averaged positive electrode interfacial current density distribution [A.m-2]",
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Voltage [V]",
    ]
    pybamm.dynamic_plot(sim_cc, output_variables=output_variables)


def multi_species_multi_reaction_model():
    model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})

    parameter_values = model.default_parameter_values

    # Loop over domains
    for domain in ["negative", "positive"]:
        Electrode = domain.capitalize()
        # Loop over reactions
        N = int(parameter_values["Number of reactions in " + domain + " electrode"])
        for i in range(N):
            names = [
                f"{Electrode} electrode host site occupancy fraction ({i})",
                f"{Electrode} electrode host site standard potential ({i}) [V]",
                f"{Electrode} electrode host site ideality factor ({i})",
                f"{Electrode} electrode host site charge transfer coefficient ({i})",
                f"{Electrode} electrode host site reference exchange-current density ({i}) [A.m-2]",
            ]
            for name in names:
                print(f"{name} = {parameter_values[name]}")

    # get symbolic parameters
    param = model.param
    param_n = param.n.prim
    param_p = param.p.prim

    num_reactions_n = int(parameter_values["Number of reactions in negative electrode"])
    num_reactions_p = int(parameter_values["Number of reactions in positive electrode"])

    # set up ranges for plotting
    U_n = pybamm.linspace(0.05, 1.1, 1000)
    U_p = pybamm.linspace(2.8, 4.4, 1000)

    # get reference electrolyte concentration and temperature
    c_e = param.c_e_init
    T = param.T_init

    # set up figure
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    colors = ["r", "g", "b", "c", "m", "y"]

    # sto vs potential
    x_n = param_n.x(U_n, T)
    x_p = param_p.x(U_p, T)
    ax[0, 0].plot(parameter_values.evaluate(x_n), parameter_values.evaluate(U_n), "k-")
    ax[0, 1].plot(parameter_values.evaluate(x_p), parameter_values.evaluate(U_p), "k-")
    ax[0, 0].set_xlabel("x_n")
    ax[0, 0].set_ylabel("U_n [V]")
    ax[0, 1].set_xlabel("x_p")
    ax[0, 1].set_ylabel("U_p [V]")

    # fractional occupancy vs potential
    for i in range(num_reactions_n):
        xj = param_n.x_j(U_n, T, i)
        ax[1, 0].plot(
            parameter_values.evaluate(x_n),
            parameter_values.evaluate(xj),
            color=colors[i],
            label=f"x_n_{i}",
        )
    ax[1, 0].set_xlabel("x_n")
    ax[1, 0].set_ylabel("x_n_j")
    ax[1, 0].legend()
    for i in range(num_reactions_p):
        xj = param_p.x_j(U_p, T, i)
        ax[1, 1].plot(
            parameter_values.evaluate(x_p),
            parameter_values.evaluate(xj),
            color=colors[i],
            label=f"x_p_{i}",
        )
    ax[1, 1].set_xlabel("x_p")
    ax[1, 1].set_ylabel("x_p_j")
    ax[1, 1].legend()

    # exchange current density vs potential
    for i in range(num_reactions_n):
        xj = param_n.x_j(U_n, T, i)
        j0 = param_n.j0_j(c_e, U_n, T, i)
        ax[2, 0].plot(
            parameter_values.evaluate(x_n),
            parameter_values.evaluate(j0),
            color=colors[i],
            label=f"j0_n_{i}",
        )
    ax[2, 0].set_xlabel("x_n")
    ax[2, 0].set_ylabel("j0_n_j [A.m-2]")
    ax[2, 0].legend()
    for i in range(num_reactions_p):
        xj = param_p.x_j(U_p, T, i)
        j0 = param_p.j0_j(c_e, U_p, T, i)
        ax[2, 1].plot(
            parameter_values.evaluate(x_p),
            parameter_values.evaluate(j0),
            color=colors[i],
            label=f"j0_p_{i}",
        )
    ax[2, 1].set_ylim([0, 0.5])
    ax[2, 1].set_xlabel("x_p")
    ax[2, 1].set_ylabel("j0_p_j [A.m-2]")
    ax[2, 1].legend()

    plt.tight_layout()
    plt.show()

    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 1C for 1 hour or until 3 V",
                "Rest for 1 hour",
                "Charge at C/3 until 4.2 V",
                "Hold at 4.2 V until 10 mA",
                "Rest for 1 hour",
            ),
        ],
    )
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve()

    sim.plot(
        [
            "Negative particle stoichiometry",
            "Positive particle stoichiometry",
            "X-averaged negative electrode open-circuit potential [V]",
            "X-averaged positive electrode open-circuit potential [V]",
            "Negative particle potential [V]",
            "Positive particle potential [V]",
            "Current [A]",
            "Voltage [V]",
        ],
        variable_limits="tight",  # make axes tight to plot at each timestep
    )

    xns = [
        f"Average x_n_{i}" for i in range(num_reactions_n)
    ]  # negative electrode reactions: x_n_0, x_n_1, ..., x_n_5
    xps = [
        f"Average x_p_{i}" for i in range(num_reactions_p)
    ]  # positive electrode reactions: x_p_0, x_p_1, ..., x_p_3
    sim.plot(
        [
            xns,
            xps,
            "Current [A]",
            "Negative electrode stoichiometry",
            "Positive electrode stoichiometry",
            "Voltage [V]",
        ]
    )

    sol = sim.solution
    time = sol["Time [h]"].data
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].plot(time, sol["Average negative particle stoichiometry"].data, "k-", label="x_n")
    bottom = 0
    for xn in xns:
        top = bottom + sol[xn].data
        ax[0].fill_between(time, bottom, top, label=xn[-4:])
        bottom = top
    ax[0].set_xlabel("Time [h]")
    ax[0].set_ylabel("x_n [-]")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax[1].plot(time, sol["Average positive particle stoichiometry"].data, "k-", label="x_p")
    bottom = 0
    for xp in xps:
        top = bottom + sol[xp].data
        ax[1].fill_between(time, bottom, top, label=xp[-4:])
        bottom = top
    ax[1].set_xlabel("Time [h]")
    ax[1].set_ylabel("x_p [-]")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.show()


def pouch_cell_model():
    cc_model = pybamm.current_collector.EffectiveResistance({"dimensionality": 1})
    dfn_av = pybamm.lithium_ion.DFN({"thermal": "lumped"}, name="Average DFN")
    dfn = pybamm.lithium_ion.DFN(
        {"current collector": "potential pair", "dimensionality": 1, "thermal": "x-lumped"},
        name="1+1D DFN",
    )

    models = {"Current collector": cc_model, "Average DFN": dfn_av, "1+1D DFN": dfn}

    param = dfn.default_parameter_values
    I_1C = param[
        "Nominal cell capacity [A.h]"
    ]  # 1C current is cell capacity multipled by 1 hour
    param.update(
        {
            "Current function [A]": I_1C * 3,
            "Negative particle diffusivity [m2.s-1]": 3.9 * 10 ** (-14),
            "Positive particle diffusivity [m2.s-1]": 10 ** (-13),
            "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 10,
            "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 10,
            "Negative tab heat transfer coefficient [W.m-2.K-1]": 10,
            "Positive tab heat transfer coefficient [W.m-2.K-1]": 10,
            "Edge heat transfer coefficient [W.m-2.K-1]": 10,
            "Total heat transfer coefficient [W.m-2.K-1]": 10,
        },
        check_already_exists=False,
    )

    npts = 16
    var_pts = {
        "x_n": npts,
        "x_s": npts,
        "x_p": npts,
        "r_n": npts,
        "r_p": npts,
        "z": npts,
    }

    data_loader = pybamm.DataLoader()
    comsol_results_path = pybamm.get_parameters_filepath(
        f"{data_loader.get_data('comsol_1plus1D_3C.json')}"
    )
    comsol_variables = json.load(open(comsol_results_path))

    simulations = {}
    solutions = {}  # store solutions in a separate dict for easy access later
    for name, model in models.items():
        sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts)
        simulations[name] = sim  # store simulation for later
        if name == "Current collector":
            # model is independent of time, so just solve arbitrarily at t=0 using
            # the default algebraic solver
            t_eval = np.array([0])
            solutions[name] = sim.solve(t_eval=t_eval)
        else:
            # solve at COMSOL times using Casadi solver in "fast" mode
            t_eval = np.array(comsol_variables["time"])
            solutions[name] = sim.solve(
                solver=pybamm.CasadiSolver(mode="fast"), t_eval=t_eval
            )

    # set up times
    comsol_t = np.array(comsol_variables["time"])
    pybamm_t = comsol_t
    # set up space
    mesh = simulations["1+1D DFN"].mesh
    L_z = param.evaluate(dfn.param.L_z)
    pybamm_z = mesh["current collector"].nodes
    z_interp = pybamm_z


    def get_interp_fun_curr_coll(variable_name):
        """
        Create a :class:`pybamm.Function` object using the variable (interpolate in space
        to match nodes, and then create function to interpolate in time)
        """

        comsol_z = np.array(comsol_variables[variable_name + "_z"])
        variable = np.array(comsol_variables[variable_name])
        variable = interp.interp1d(comsol_z, variable, axis=0, kind="linear")(z_interp)

        # Make sure to use dimensional time
        fun = pybamm.Interpolant(
            comsol_t, variable.T, pybamm.t, name=variable_name + "_comsol"
        )
        fun.domains = {"primary": "current collector"}
        fun.mesh = mesh.combine_submeshes("current collector")
        fun.secondary_mesh = None

        return fun

    comsol_voltage = pybamm.Interpolant(
        comsol_t,
        np.array(comsol_variables["voltage"]),
        pybamm.t,
        name="voltage_comsol",
    )
    comsol_phi_s_cn = get_interp_fun_curr_coll("phi_s_cn")
    comsol_phi_s_cp = get_interp_fun_curr_coll("phi_s_cp")
    comsol_current = get_interp_fun_curr_coll("current")
    comsol_temperature = get_interp_fun_curr_coll("temperature")

    comsol_model = pybamm.BaseModel()
    comsol_model._geometry = pybamm.battery_geometry(options={"dimensionality": 1})
    comsol_model.variables = {
        "Voltage [V]": comsol_voltage,
        "Negative current collector potential [V]": comsol_phi_s_cn,
        "Positive current collector potential [V]": comsol_phi_s_cp,
        "Current collector current density [A.m-2]": comsol_current,
        "X-averaged cell temperature [K]": comsol_temperature,
        # Add spatial variables to match pybamm model
        "z [m]": simulations["1+1D DFN"].built_model.variables["z [m]"],
    }

    comsol_solution = pybamm.Solution(
        solutions["1+1D DFN"].t, solutions["1+1D DFN"].y, comsol_model, {}
    )

    V_av = solutions["Average DFN"]["Voltage [V]"]
    I_av = solutions["Average DFN"]["Total current density [A.m-2]"]

    dfncc_vars = cc_model.post_process(solutions["Current collector"], param, V_av, I_av)

    def plot(
        t_plot,
        z_plot,
        t_slices,
        var_name,
        units,
        comsol_var_fun,
        dfn_var_fun,
        dfncc_var_fun,
        param,
        cmap="viridis",
    ):
        fig, ax = plt.subplots(2, 2, figsize=(13, 7))
        fig.subplots_adjust(
            left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0.4, hspace=0.8
        )
        # plot comsol var
        comsol_var = comsol_var_fun(t=t_plot, z=z_plot)
        comsol_var_plot = ax[0, 0].pcolormesh(
            z_plot * 1e3, t_plot, np.transpose(comsol_var), shading="gouraud", cmap=cmap
        )
        if "cn" in var_name:
            format = "%.0e"
        elif "cp" in var_name:
            format = "%.0e"
        else:
            format = None
        fig.colorbar(
            comsol_var_plot,
            ax=ax,
            format=format,
            location="top",
            shrink=0.42,
            aspect=20,
            anchor=(0.0, 0.0),
        )

        # plot slices
        ccmap = plt.get_cmap("inferno")
        for ind, t in enumerate(t_slices):
            color = ccmap(float(ind) / len(t_slices))
            comsol_var_slice = comsol_var_fun(t=t, z=z_plot)
            dfn_var_slice = dfn_var_fun(t=t, z=z_plot)
            dfncc_var_slice = dfncc_var_fun(t=np.array([t]), z=z_plot)
            ax[0, 1].plot(
                z_plot * 1e3, comsol_var_slice, "o", fillstyle="none", color=color
            )
            ax[0, 1].plot(
                z_plot * 1e3,
                dfn_var_slice,
                "-",
                color=color,
                label=f"{t_slices[ind]:.0f} s",
            )
            ax[0, 1].plot(z_plot * 1e3, dfncc_var_slice, ":", color=color)
        # add dummy points for legend of styles
        (comsol_p,) = ax[0, 1].plot(np.nan, np.nan, "ko", fillstyle="none")
        (pybamm_p,) = ax[0, 1].plot(np.nan, np.nan, "k-", fillstyle="none")
        (dfncc_p,) = ax[0, 1].plot(np.nan, np.nan, "k:", fillstyle="none")

        # compute errors
        dfn_var = dfn_var_fun(t=t_plot, z=z_plot)
        dfncc_var = dfncc_var_fun(t=t_plot, z=z_plot)
        error = np.abs(comsol_var - dfn_var)
        error_bar = np.abs(comsol_var - dfncc_var)

        # plot time averaged error
        ax[1, 0].plot(z_plot * 1e3, np.nanmean(error, axis=1), "k-", label=r"$1+1$D")
        ax[1, 0].plot(z_plot * 1e3, np.nanmean(error_bar, axis=1), "k:", label="DFNCC")

        # plot z averaged error
        ax[1, 1].plot(t_plot, np.nanmean(error, axis=0), "k-", label=r"$1+1$D")
        ax[1, 1].plot(t_plot, np.nanmean(error_bar, axis=0), "k:", label="DFNCC")

        # set ticks
        ax[0, 0].tick_params(which="both")
        ax[0, 1].tick_params(which="both")
        ax[1, 0].tick_params(which="both")
        if var_name in ["$\mathcal{I}^*$"]:
            ax[1, 0].set_yscale("log")
            ax[1, 0].set_yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1]
        else:
            ax[1, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
        ax[1, 1].tick_params(which="both")
        if var_name in ["$\phi^*_{\mathrm{s,cn}}$", "$\phi^*_{\mathrm{s,cp}} - V^*$"]:
            ax[1, 0].ticklabel_format(style="sci", scilimits=(-2, 2), axis="y")
        else:
            ax[1, 1].set_yscale("log")
            ax[1, 1].set_yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1]

        # set labels
        ax[0, 0].set_xlabel(r"$z^*$ [mm]")
        ax[0, 0].set_ylabel(r"$t^*$ [s]")
        ax[0, 0].set_title(rf"{var_name} {units}", y=1.5)
        ax[0, 1].set_xlabel(r"$z^*$ [mm]")
        ax[0, 1].set_ylabel(rf"{var_name}")
        ax[1, 0].set_xlabel(r"$z^*$ [mm]")
        ax[1, 0].set_ylabel("Time-averaged" + "\n" + rf"absolute error {units}")
        ax[1, 1].set_xlabel(r"$t^*$ [s]")
        ax[1, 1].set_ylabel("Space-averaged" + "\n" + rf"absolute error {units}")

        ax[0, 0].text(-0.1, 1.6, "(a)", transform=ax[0, 0].transAxes)
        ax[0, 1].text(-0.1, 1.6, "(b)", transform=ax[0, 1].transAxes)
        ax[1, 0].text(-0.1, 1.2, "(c)", transform=ax[1, 0].transAxes)
        ax[1, 1].text(-0.1, 1.2, "(d)", transform=ax[1, 1].transAxes)

        leg1 = ax[0, 1].legend(
            bbox_to_anchor=(0, 1.1, 1.0, 0.102),
            loc="lower left",
            borderaxespad=0.0,
            ncol=3,
            mode="expand",
        )

        ax[0, 1].legend(
            [comsol_p, pybamm_p, dfncc_p],
            ["COMSOL", r"$1+1$D", "DFNCC"],
            bbox_to_anchor=(0, 1.5, 1.0, 0.102),
            loc="lower left",
            borderaxespad=0.0,
            ncol=3,
            mode="expand",
        )
        ax[0, 1].add_artist(leg1)

        ax[1, 0].legend(
            bbox_to_anchor=(0.0, 1.1, 1.0, 0.102),
            loc="lower right",
            borderaxespad=0.0,
            ncol=3,
        )
        ax[1, 1].legend(
            bbox_to_anchor=(0.0, 1.1, 1.0, 0.102),
            loc="lower right",
            borderaxespad=0.0,
            ncol=3,
        )

        plt.show()

    t_plot = comsol_t
    z_plot = z_interp
    t_slices = np.array([600, 1200, 1800, 2400, 3000]) / 3

    var = "Negative current collector potential [V]"
    comsol_var_fun = comsol_solution[var]
    dfn_var_fun = solutions["1+1D DFN"][var]

    dfncc_var_fun = dfncc_vars[var]
    plot(
        t_plot,
        z_plot,
        t_slices,
        "$\phi^*_{\mathrm{s,cn}}$",
        "[V]",
        comsol_var_fun,
        dfn_var_fun,
        dfncc_var_fun,
        param,
        cmap="cividis",
    )


    var = "Positive current collector potential [V]"
    comsol_var = comsol_solution[var]
    V_comsol = comsol_solution["Voltage [V]"]


    def comsol_var_fun(t, z):
        return comsol_var(t=t, z=z) - V_comsol(t=t)


    dfn_var = solutions["1+1D DFN"][var]
    V = solutions["1+1D DFN"]["Voltage [V]"]


    def dfn_var_fun(t, z):
        return dfn_var(t=t, z=z) - V(t=t)


    dfncc_var = dfncc_vars[var]
    V_dfncc = dfncc_vars["Voltage [V]"]


    def dfncc_var_fun(t, z):
        return dfncc_var(t=t, z=z) - V_dfncc(t)


    plot(
        t_plot,
        z_plot,
        t_slices,
        "$\phi^*_{\mathrm{s,cp}} - V^*$",
        "[V]",
        comsol_var_fun,
        dfn_var_fun,
        dfncc_var_fun,
        param,
        cmap="viridis",
    )

    var = "Current collector current density [A.m-2]"
    comsol_var_fun = comsol_solution[var]
    dfn_var_fun = solutions["1+1D DFN"][var]

    I_av = solutions["Average DFN"][var]


    def dfncc_var_fun(t, z):
        "In the DFNCC the current is just the average current"
        return np.transpose(np.repeat(I_av(t)[:, np.newaxis], len(z), axis=1))


    plot(
        t_plot,
        z_plot,
        t_slices,
        "$\mathcal{I}^*$",
        "[A/m${}^2$]",
        comsol_var_fun,
        dfn_var_fun,
        dfncc_var_fun,
        param,
        cmap="plasma",
    )


    T_ref = param.evaluate(dfn.param.T_ref)
    var = "X-averaged cell temperature [K]"
    comsol_var = comsol_solution[var]


    def comsol_var_fun(t, z):
        return comsol_var(t=t, z=z) - T_ref


    dfn_var = solutions["1+1D DFN"][var]


    def dfn_var_fun(t, z):
        return dfn_var(t=t, z=z) - T_ref


    T_av = solutions["Average DFN"][var]


    def dfncc_var_fun(t, z):
        "In the DFNCC the temperature is just the average temperature"
        return np.transpose(np.repeat(T_av(t)[:, np.newaxis], len(z), axis=1)) - T_ref


    plot(
        t_plot,
        z_plot,
        t_slices,
        "$\\bar{T}^* - \\bar{T}_0^*$",
        "[K]",
        comsol_var_fun,
        dfn_var_fun,
        dfncc_var_fun,
        param,
        cmap="inferno",
    )


def generate_rate_capability_plots():
    model = pybamm.lithium_ion.SPMe()

    C_rates = np.linspace(0.05, 5, 20)
    capacities = np.zeros_like(C_rates)
    currents = np.zeros_like(C_rates)
    voltage_av = np.zeros_like(C_rates)

    for i, C_rate in enumerate(C_rates):
        experiment = pybamm.Experiment([f"Discharge at {C_rate:.4f}C until 3.2V"])
        sim = pybamm.Simulation(model, experiment=experiment)
        sim.solve()

        time = sim.solution["Time [s]"].entries
        capacity = sim.solution["Discharge capacity [A.h]"]
        current = sim.solution["Current [A]"]
        voltage = sim.solution["Voltage [V]"]

        capacities[i] = capacity(time[-1])
        currents[i] = current(time[-1])
        voltage_av[i] = np.mean(voltage(time))

    plt.figure(1)
    plt.scatter(C_rates, capacities)
    plt.xlabel("C-rate")
    plt.ylabel("Capacity [Ah]")

    plt.figure(2)
    plt.scatter(currents * voltage_av, capacities * voltage_av)
    plt.xlabel("Power [W]")
    plt.ylabel("Energy [Wh]")

    plt.show()


def modelling_SEI_growth_on_particle_cracks():

    model1 = pybamm.lithium_ion.DFN(
        {"SEI": "solvent-diffusion limited", "particle mechanics": "swelling only"}
    )
    model2 = pybamm.lithium_ion.DFN(
        {
            "particle mechanics": "swelling and cracking",
            "SEI": "solvent-diffusion limited",
            "SEI on cracks": "true",
        }
    )

    param = pybamm.ParameterValues("OKane2022")
    var_pts = {
        "x_n": 20,  # negative electrode
        "x_s": 20,  # separator
        "x_p": 20,  # positive electrode
        "r_n": 26,  # negative particle
        "r_p": 26,  # positive particle
    }

    exp = pybamm.Experiment(
        ["Hold at 4.2 V until C/100", "Rest for 1 hour", "Discharge at 1C until 2.5 V"]
    )
    sim1 = pybamm.Simulation(
        model1, parameter_values=param, experiment=exp, var_pts=var_pts
    )
    sol1 = sim1.solve(calc_esoh=False)
    sim2 = pybamm.Simulation(
        model2, parameter_values=param, experiment=exp, var_pts=var_pts
    )
    sol2 = sim2.solve(calc_esoh=False)

    t1 = sol1["Time [s]"].entries
    V1 = sol1["Voltage [V]"].entries
    SEI1 = sol1["Loss of lithium to negative SEI [mol]"].entries
    lithium_neg1 = sol1["Total lithium in negative electrode [mol]"].entries
    lithium_pos1 = sol1["Total lithium in positive electrode [mol]"].entries
    t2 = sol2["Time [s]"].entries
    V2 = sol2["Voltage [V]"].entries
    SEI2 = (
        sol2["Loss of lithium to negative SEI [mol]"].entries
        + sol2["Loss of lithium to negative SEI on cracks [mol]"].entries
    )
    lithium_neg2 = sol2["Total lithium in negative electrode [mol]"].entries
    lithium_pos2 = sol2["Total lithium in positive electrode [mol]"].entries

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
    ax1.plot(t1, V1, label="without cracking")
    ax1.plot(t2, V2, label="with cracking", linestyle="dashed")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]")
    ax1.legend()
    ax2.plot(t1, SEI1, label="without cracking")
    ax2.plot(t2, SEI2, label="with cracking", linestyle="dashed")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Loss of lithium to SEI [mol]")
    ax2.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(t2, lithium_neg2 + lithium_pos2)
    ax.plot(t2, lithium_neg2[0] + lithium_pos2[0] - SEI2, linestyle="dashed")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total lithium in electrodes [mol]")
    plt.show()


def simulate_3E_cell():
    model = pybamm.lithium_ion.DFN()

    L_n = model.param.n.L  # Negative electrode thickness [m]
    L_s = model.param.s.L  # Separator thickness [m]
    L_ref = L_n + L_s / 2  # Reference electrode position [m]

    model.insert_reference_electrode(L_ref)

    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])

    sim.plot(
        [
            [
                "Negative electrode surface potential difference at separator interface [V]",
                "Negative electrode 3E potential [V]",
            ],
            [
                "Positive electrode surface potential difference at separator interface [V]",
                "Positive electrode 3E potential [V]",
            ],
            "Voltage [V]",
        ]
    )


def LG_M50_ORegan2022():

    # DFN + lumped thermal
    options = {"thermal": "lumped", "dimensionality": 0, "cell geometry": "arbitrary"}
    model = pybamm.lithium_ion.DFN(options=options)

    # O'Regan 2022 parameter set
    param = pybamm.ParameterValues("ORegan2022")

    # Choose CasADI fast (we do a short discharge so there are no events, if events are needed choose "fast with events")
    solver = pybamm.IDAKLUSolver()

    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 40, "r_p": 40}

    submesh_types = model.default_submesh_types
    submesh_types["negative particle"] = pybamm.MeshGenerator(
        pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
    )
    submesh_types["positive particle"] = pybamm.MeshGenerator(
        pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
    )

    # Define the simulation
    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        C_rate=1,
        solver=solver,
        var_pts=var_pts,
        submesh_types=submesh_types,
    )
    sim.solve(
        [0, 1800]
    )  # solving time kept short for testing purposes, feel free to extend it
    sim.plot()

    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        C_rate=1,
        solver=solver,
        # var_pts=var_pts,
        # submesh_types=submesh_types,
    )
    sim.solve(
        [0, 3600]
    )  # solving time kept short for testing purposes, feel free to extend it
    sim.plot()


def DFN_model_for_sodium_ion_batteries():
    model = pybamm.sodium_ion.BasicDFN()

    C_rates = [1 / 12, 5 / 12, 10 / 12, 1]
    solutions = []

    for C_rate in C_rates:
        sim = pybamm.Simulation(model, C_rate=C_rate)
        sol = sim.solve([0, 4000 / C_rate])
        solutions.append(sol)

    pybamm.dynamic_plot(solutions)

    for solution, C_rate in zip(solutions, C_rates, strict=False):
        capacity = [i * 1000 for i in solution["Discharge capacity [A.h]"].entries]
        voltage = solution["Voltage [V]"].entries
        plt.plot(capacity, voltage, label=f"{(12 * C_rate)} A.m-2")

    plt.xlabel("Discharge Capacity [mA.h]")
    plt.ylabel("Voltage [V]")
    plt.show()


def Single_Particle_Model():
    model = pybamm.lithium_ion.SPM()

    variable = list(model.rhs.keys())[1]
    equation = list(model.rhs.values())[1]
    print("rhs equation for variable '", variable, "' is:")
    equation.visualise("spm1.png")

    geometry = model.default_geometry
    print("SPM domains:")
    for i, (k, v) in enumerate(geometry.items()):
        print(str(i + 1) + ".", k, "with variables:")
        for var, rng in v.items():
            if "min" in rng:
                print("  -(", rng["min"], ") <=", var, "<= (", rng["max"], ")")
            else:
                print(var, "=", rng["position"])

    param = model.default_parameter_values

    param.process_model(model)
    param.process_geometry(geometry)

    for k, t in model.default_submesh_types.items():
        print(k, "is of type", t.__name__)
    for var, npts in model.default_var_pts.items():
        print(var, "has", npts, "mesh points")

    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    for k, method in model.default_spatial_methods.items():
        print(k, "is discretised using", method.__class__.__name__, "method")

    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    model.concatenated_rhs.children[1].visualise("spm2.png")

    # Solve the model at the given time points (in seconds)
    solver = model.default_solver
    n = 250
    t_eval = np.linspace(0, 3600, n)
    print("Solving using", type(solver).__name__, "solver...")
    solution = solver.solve(model, t_eval)
    print("Finished.")

    print("SPM model variables:")
    for v in model.variables.keys():
        print("\t-", v)

    voltage = solution["Voltage [V]"]
    c_s_n_surf = solution["Negative particle surface concentration"]
    c_s_p_surf = solution["Positive particle surface concentration"]

    t = solution["Time [s]"].entries
    x = solution["x [m]"].entries[:, 0]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

    ax1.plot(t, voltage(t))
    ax1.set_xlabel(r"$Time [s]$")
    ax1.set_ylabel("Voltage [V]")

    ax2.plot(
        t, c_s_n_surf(t=t, x=x[0])
    )  # can evaluate at arbitrary x (single representative particle)
    ax2.set_xlabel(r"$Time [s]$")
    ax2.set_ylabel("Negative particle surface concentration")

    ax3.plot(
        t, c_s_p_surf(t=t, x=x[-1])
    )  # can evaluate at arbitrary x (single representative particle)
    ax3.set_xlabel(r"$Time [s]$")
    ax3.set_ylabel("Positive particle surface concentration")

    plt.tight_layout()
    plt.show()

    c_s_n = solution["Negative particle concentration"]
    c_s_p = solution["Positive particle concentration"]
    r_n = solution["r_n [m]"].entries[:, 0]
    r_p = solution["r_p [m]"].entries[:, 0]

    c_s_n = solution["Negative particle concentration"]
    c_s_p = solution["Positive particle concentration"]
    r_n = solution["r_n [m]"].entries[:, 0, 0]
    r_p = solution["r_p [m]"].entries[:, 0, 0]


    def plot_concentrations(t):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        (plot_c_n,) = ax1.plot(
            r_n, c_s_n(r=r_n, t=t, x=x[0])
        )  # can evaluate at arbitrary x (single representative particle)
        (plot_c_p,) = ax2.plot(
            r_p, c_s_p(r=r_p, t=t, x=x[-1])
        )  # can evaluate at arbitrary x (single representative particle)
        ax1.set_ylabel("Negative particle concentration")
        ax2.set_ylabel("Positive particle concentration")
        ax1.set_xlabel(r"$r_n$ [m]")
        ax2.set_xlabel(r"$r_p$ [m]")
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        plt.show()

    quick_plot = pybamm.QuickPlot(solution)
    quick_plot.dynamic_plot()


def Single_Particle_Model_with_Electrolyte():
    # load model
    model = pybamm.lithium_ion.SPMe()

    # create simulation
    simulation = pybamm.Simulation(model)

    # solve simulation
    simulation.solve([0, 3600])  # time interval in seconds

    simulation.plot()


def Using_crack_submodels():

    model = pybamm.lithium_ion.DFN(
        options={
            "particle": "Fickian diffusion",
            "particle mechanics": "swelling and cracking",  # other options are "none", "swelling only"
        }
    )

    param = pybamm.ParameterValues("Ai2020")
    ## It can update the speed of crack propagation using the commands below:
    # param.update({"Negative electrode Cracking rate":3.9e-20*10}, check_already_exists=False)

    var_pts = {
        "x_n": 20,  # negative electrode
        "x_s": 20,  # separator
        "x_p": 20,  # positive electrode
        "r_n": 26,  # negative particle
        "r_p": 26,  # positive particle
    }

    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        var_pts=var_pts,
    )
    solution = sim.solve(t_eval=[0, 3600], inputs={"C-rate": 1})
    # plot
    quick_plot = pybamm.QuickPlot(solution)
    quick_plot.dynamic_plot()

    # extract voltage
    E_n = param["Negative electrode Young's modulus [Pa]"]
    stress_t_n_surf = solution["Negative particle surface tangential stress [Pa]"]
    x = solution["x [m]"].entries[0:19, 0]
    c_s_n = solution["Negative particle concentration"]
    r_n = solution["r_n [m]"].entries[:, 0, 0]

    # plot


    def plot_concentrations(t):
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4))
        ax1.plot(x, stress_t_n_surf(t=t, x=x) / E_n)
        ax1.set_xlabel(r"$x_n$ [m]")
        ax1.set_ylabel("$\sigma_t/E_n$")

        (plot_c_n,) = ax2.plot(
            r_n, c_s_n(r=r_n, t=t, x=x[0])
        )  # can evaluate at arbitrary x (single representative particle)
        ax2.set_ylabel("Negative particle concentration")
        ax2.set_xlabel(r"$r_n$ [m]")
        ax2.set_ylim(0, 1)
        ax2.set_title("Close to current collector")
        ax2.grid()

        (plot_c_n,) = ax3.plot(
            r_n, c_s_n(r=r_n, t=t, x=x[10])
        )  # can evaluate at arbitrary x (single representative particle)
        ax3.set_ylabel("Negative particle concentration")
        ax3.set_xlabel(r"$r_n$ [m]")
        ax3.set_ylim(0, 1)
        ax3.set_title("In the middle")
        ax3.grid()

        (plot_c_n,) = ax4.plot(
            r_n, c_s_n(r=r_n, t=t, x=x[-1])
        )  # can evaluate at arbitrary x (single representative particle)
        ax4.set_ylabel("Negative particle concentration")
        ax4.set_xlabel(r"$r_n$ [m]")
        ax4.set_ylim(0, 1)
        ax4.set_title("Close to separator")
        ax4.grid()
        plt.show()

    label = ["Crack model"]
    output_variables = [
        "Negative particle crack length [m]",
        "Positive particle crack length [m]",
        "X-averaged negative particle crack length [m]",
        "X-averaged positive particle crack length [m]",
    ]
    quick_plot = pybamm.QuickPlot(
        solution, output_variables, label, variable_limits="tight"
    )
    quick_plot.dynamic_plot()


def Loss_of_active_material_submodels():
    model = pybamm.lithium_ion.DFN(
        options={
            "SEI": "solvent-diffusion limited",
            "SEI porosity change": "false",
            "particle mechanics": "swelling only",
            "loss of active material": "stress-driven",
        }
    )
    param = pybamm.ParameterValues("Ai2020")
    param.update({"Negative electrode LAM constant proportional term [s-1]": 1e-4 / 3600})
    param.update({"Positive electrode LAM constant proportional term [s-1]": 1e-4 / 3600})
    experiment = pybamm.Experiment(
        [
            "Discharge at 1C until 3 V",
            "Rest for 600 seconds",
            "Charge at 1C until 4.2 V",
            "Hold at 4.199 V for 600 seconds",
        ]
    )
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=param,
        discretisation_kwargs={"remove_independent_variables_from_rhs": True},
    )
    solution = sim.solve(calc_esoh=False)

    sim.plot(
        [
            "Voltage [V]",
            "Current [A]",
            "Sum of x-averaged positive electrode volumetric interfacial current densities [A.m-3]",
            "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
            "X-averaged positive electrode active material volume fraction",
            "X-averaged negative electrode active material volume fraction",
            "X-averaged positive particle surface tangential stress [Pa]",
            "X-averaged negative particle surface tangential stress [Pa]",
        ]
    )

    ks = [1e-4, 1e-3, 1e-2]
    solutions = []

    for k in ks:
        param.update({"Positive electrode LAM constant proportional term [s-1]": k / 3600})
        param.update({"Negative electrode LAM constant proportional term [s-1]": k / 3600})

        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=param,
            discretisation_kwargs={"remove_independent_variables_from_rhs": True},
        )
        solution = sim.solve(calc_esoh=False)
        solutions.append(solution)

    pybamm.dynamic_plot(
        solutions,
        output_variables=[
            "Voltage [V]",
            "Current [A]",
            "Sum of x-averaged positive electrode volumetric interfacial current densities [A.m-3]",
            "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
            "X-averaged positive electrode active material volume fraction",
            "X-averaged negative electrode active material volume fraction",
            "X-averaged positive electrode surface area to volume ratio [m-1]",
            "X-averaged negative electrode surface area to volume ratio [m-1]",
        ],
        labels=[f"k={k:.0e}" for k in ks],
    )

    experiment = pybamm.Experiment(
        [
            "Discharge at 1C until 3 V",
            "Rest for 600 seconds",
            "Charge at 1C until 4.2 V",
            "Hold at 4.199 V for 600 seconds",
        ]
    )
    model = pybamm.lithium_ion.DFN(
        options={
            "SEI": "solvent-diffusion limited",
            "loss of active material": "reaction-driven",
        }
    )
    param = pybamm.ParameterValues("Chen2020")
    param.update({"Negative electrode reaction-driven LAM factor [m3.mol-1]": 1e-4})
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=param,
    )
    solution = sim.solve(calc_esoh=False)

    sim.plot(
        [
            "Voltage [V]",
            "Current [A]",
            "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
            "X-averaged negative electrode active material volume fraction",
            "Negative total SEI thickness [m]",
            "X-averaged negative total SEI thickness [m]",
        ]
    )

    def current_LAM(i, T):
        return -1e-10 * (abs(i) + 1e3 * abs(i) ** 0.5)


    model = pybamm.lithium_ion.DFN(
        options={
            "loss of active material": "current-driven",
        }
    )
    param = pybamm.ParameterValues("Chen2020")
    param.update(
        {
            "Positive electrode current-driven LAM rate": current_LAM,
            "Negative electrode current-driven LAM rate": current_LAM,
        },
        check_already_exists=False,
    )
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=param,
    )
    solution = sim.solve(calc_esoh=False)

    sim.plot(
        [
            "Voltage [V]",
            "Current [A]",
            "X-averaged positive electrode active material volume fraction",
            "X-averaged negative electrode active material volume fraction",
        ]
    )

    # Volume change functions from Ai2020 parameters


    def graphite_volume_change_Ai2020(sto):
        p1 = 145.907
        p2 = -681.229
        p3 = 1334.442
        p4 = -1415.710
        p5 = 873.906
        p6 = -312.528
        p7 = 60.641
        p8 = -5.706
        p9 = 0.386
        p10 = -4.966e-05
        t_change = (
            p1 * sto**9
            + p2 * sto**8
            + p3 * sto**7
            + p4 * sto**6
            + p5 * sto**5
            + p6 * sto**4
            + p7 * sto**3
            + p8 * sto**2
            + p9 * sto
            + p10
        )
        return t_change


    def lico2_volume_change_Ai2020(sto):
        omega = pybamm.Parameter("Positive electrode partial molar volume [m3.mol-1]")
        c_s_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
        t_change = omega * c_s_max * sto
        return t_change

    options = {
        "particle phases": ("2", "1"),
        "open-circuit potential": (("single", "current sigmoid"), "single"),
        "loss of active material": "stress-driven",
    }

    model = pybamm.lithium_ion.SPM(options)
    parameter_values = pybamm.ParameterValues("Chen2020_composite")
    second = 0.1
    parameter_values.update(
        {
            "Primary: Negative electrode reference concentration for free of deformation [mol.m-3]": 0.0,
            "Secondary: Negative electrode reference concentration for free of deformation [mol.m-3]": 0.0,
            "Primary: Negative electrode LAM constant proportional term [s-1]": 1e-4 / 3600,
            "Secondary: Negative electrode LAM constant proportional term [s-1]": 1e-4
            / 3600
            * second,
            "Positive electrode LAM constant proportional term [s-1]": 1e-4 / 3600,
            "Primary: Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
            "Primary: Negative electrode Young's modulus [Pa]": 15000000000.0,
            "Primary: Negative electrode Poisson's ratio": 0.3,
            "Primary: Negative electrode critical stress [Pa]": 60000000.0,
            "Secondary: Negative electrode critical stress [Pa]": 60000000.0,
            "Primary: Negative electrode LAM constant exponential term": 2.0,
            "Secondary: Negative electrode LAM constant exponential term": 2.0,
            "Secondary: Negative electrode partial molar volume [m3.mol-1]": 3.1e-06
            * second,
            "Secondary: Negative electrode Young's modulus [Pa]": 15000000000.0 * second,
            "Secondary: Negative electrode Poisson's ratio": 0.3 * second,
            "Negative electrode reference concentration for free of deformation [mol.m-3]": 0.0,
            "Primary: Negative electrode volume change": graphite_volume_change_Ai2020,
            "Secondary: Negative electrode volume change": graphite_volume_change_Ai2020,
            "Positive electrode partial molar volume [m3.mol-1]": -7.28e-07,
            "Positive electrode Young's modulus [Pa]": 375000000000.0,
            "Positive electrode Poisson's ratio": 0.2,
            "Positive electrode critical stress [Pa]": 375000000.0,
            "Positive electrode LAM constant exponential term": 2.0,
            "Positive electrode reference concentration for free of deformation [mol.m-3]": 0.0,
            "Positive electrode volume change": lico2_volume_change_Ai2020,
        },
        check_already_exists=False,
    )

    # sim = pybamm.Simulation(model, parameter_values=parameter_values)
    # sim.solve([0, 4500])
    experiment = pybamm.Experiment(
        [
            "Discharge at 1C until 3 V",
            "Rest for 600 seconds",
            "Charge at 1C until 4.2 V",
            "Hold at 4.199 V for 600 seconds",
        ]
    )
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=parameter_values,
        discretisation_kwargs={"remove_independent_variables_from_rhs": True},
    )
    solution = sim.solve(calc_esoh=False)

    pybamm.dynamic_plot(
        sim,
        [
            "Voltage [V]",
            "Current [A]",
            [
                "Average negative primary particle concentration",
                "Average negative secondary particle concentration",
                "Average positive particle concentration",
            ],
            "X-averaged negative electrode primary active material volume fraction",
            "X-averaged positive electrode active material volume fraction",
            "X-averaged negative electrode secondary active material volume fraction",
            "Sum of x-averaged positive electrode volumetric interfacial current densities [A.m-3]",
            "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
            "X-averaged positive particle surface tangential stress [Pa]",
            "X-averaged negative primary particle surface tangential stress [Pa]",
            "X-averaged negative secondary particle surface tangential stress [Pa]",
        ],
    )

    options = {
        "particle phases": ("2", "1"),
        "open-circuit potential": (("single", "current sigmoid"), "single"),
        "SEI": "solvent-diffusion limited",
        "loss of active material": "reaction-driven",
    }

    model = pybamm.lithium_ion.SPM(options)
    parameter_values = pybamm.ParameterValues("Chen2020_composite")
    second = 0.9

    parameter_values.update(
        {
            "Primary: Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
            "Primary: Negative electrode Young's modulus [Pa]": 15000000000.0,
            "Primary: Negative electrode Poisson's ratio": 0.3,
            "Negative electrode critical stress [Pa]": 60000000.0,
            "Negative electrode LAM constant exponential term": 2.0,
            "Secondary: Negative electrode partial molar volume [m3.mol-1]": 3.1e-06
            * second,
            "Secondary: Negative electrode Young's modulus [Pa]": 15000000000.0 * second,
            "Secondary: Negative electrode Poisson's ratio": 0.3 * second,
            "Negative electrode reference concentration for free of deformation [mol.m-3]": 0.0,
            "Primary: Negative electrode volume change": graphite_volume_change_Ai2020,
            "Secondary: Negative electrode volume change": graphite_volume_change_Ai2020,
            "Positive electrode partial molar volume [m3.mol-1]": -7.28e-07,
            "Positive electrode Young's modulus [Pa]": 375000000000.0,
            "Positive electrode Poisson's ratio": 0.2,
            "Positive electrode critical stress [Pa]": 375000000.0,
            "Positive electrode LAM constant exponential term": 2.0,
            "Positive electrode reference concentration for free of deformation [mol.m-3]": 0.0,
            "Positive electrode volume change": lico2_volume_change_Ai2020,
            "Primary: Negative electrode reaction-driven LAM factor [m3.mol-1]": 1e-9,
            "Secondary: Negative electrode reaction-driven LAM factor [m3.mol-1]": 10,
        },
        check_already_exists=False,
    )

    # Changing secondary SEI solvent diffusivity to show different degradation between phases
    parameter_values.update(
        {
            "Secondary: SEI solvent diffusivity [m2.s-1]": 2.5e-24,
        }
    )

    # sim = pybamm.Simulation(model, parameter_values=parameter_values)
    # sim.solve([0, 4100])
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=parameter_values,
    )
    solution = sim.solve(calc_esoh=False)

    sim.plot(
        [
            "Voltage [V]",
            "Current [A]",
            "Sum of x-averaged negative electrode volumetric interfacial current densities [A.m-3]",
            "X-averaged negative electrode primary active material volume fraction",
            "X-averaged negative electrode secondary active material volume fraction",
            "Negative total primary SEI thickness [m]",
            "Negative total secondary SEI thickness [m]",
        ]
    )


def thermal_models():
    options = {"cell geometry": "arbitrary", "thermal": "lumped"}
    arbitrary_lumped_model = pybamm.lithium_ion.DFN(options)
    # OR
    options = {"cell geometry": "pouch", "thermal": "lumped"}
    pouch_lumped_model = pybamm.lithium_ion.DFN(options)

    options = {"thermal": "lumped"}
    model = pybamm.lithium_ion.DFN(options)
    print("Cell geometry:", model.options["cell geometry"])

    options = {"thermal": "lumped", "contact resistance": "true"}
    model = pybamm.lithium_ion.DFN(options)

    options = {"thermal": "lumped"}
    model = pybamm.lithium_ion.DFN(options)
    print("contact resistance:", model.options["contact resistance"])

    options = {"thermal": "x-full"}
    model = pybamm.lithium_ion.DFN(options)

    options = {
        "current collector": "potential pair",
        "dimensionality": 2,
        "thermal": "x-lumped",
    }
    model = pybamm.lithium_ion.DFN(options)

    full_thermal_model = pybamm.lithium_ion.SPMe(
        {"thermal": "x-full"}, name="full thermal model"
    )
    lumped_thermal_model = pybamm.lithium_ion.SPMe(
        {"thermal": "lumped"}, name="lumped thermal model"
    )
    models = [full_thermal_model, lumped_thermal_model]

    parameter_values = pybamm.ParameterValues("Marquis2019")

    full_params = parameter_values.copy()
    full_params.update(
        {
            "Negative current collector"
            + " surface heat transfer coefficient [W.m-2.K-1]": 5,
            "Positive current collector"
            + " surface heat transfer coefficient [W.m-2.K-1]": 5,
            "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
            "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
            "Edge heat transfer coefficient [W.m-2.K-1]": 0,
        }
    )

    A = parameter_values["Electrode width [m]"] * parameter_values["Electrode height [m]"]
    lumped_params = parameter_values.copy()
    lumped_params.update(
        {
            "Total heat transfer coefficient [W.m-2.K-1]": 5,
            "Cell cooling surface area [m2]": 2 * A,
        }
    )

    params = [full_params, lumped_params]
    # loop over the models and solve
    sols = []
    for model, param in zip(models, params, strict=False):
        param["Current function [A]"] = 3 * 0.68
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve([0, 3600])
        sols.append(sim.solution)

    # plot
    output_variables = [
        "Voltage [V]",
        "X-averaged cell temperature [K]",
        "Cell temperature [K]",
    ]
    pybamm.dynamic_plot(sols, output_variables)

    # plot the results
    pybamm.dynamic_plot(
        sols,
        [
            "Volume-averaged cell temperature [K]",
            "Volume-averaged total heating [W.m-3]",
            "Current [A]",
            "Voltage [V]",
        ],
    )

    model_no_contact_resistance = pybamm.lithium_ion.SPMe(
        {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"},
        name="lumped thermal model",
    )
    model_contact_resistance = pybamm.lithium_ion.SPMe(
        {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "true"},
        name="lumped thermal model with contact resistance",
    )
    models = [model_no_contact_resistance, model_contact_resistance]

    parameter_values = pybamm.ParameterValues("Marquis2019")
    lumped_params = parameter_values.copy()
    lumped_params_contact_resistance = parameter_values.copy()

    lumped_params_contact_resistance.update(
        {
            "Contact resistance [Ohm]": 0.05,
        }
    )

    params = [lumped_params, lumped_params_contact_resistance]
    sols = []
    for model, param in zip(models, params, strict=False):
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve([0, 3600])
        sols.append(sim.solution)


    output_variables = [
        "Voltage [V]",
        "X-averaged cell temperature [K]",
        "Cell temperature [K]",
    ]
    pybamm.dynamic_plot(sols, output_variables)


def transport_efficiency_and_models_for_tortuosity_factor():

    sols = []
    te_opts = pybamm.BatteryModelOptions({}).possible_options["transport efficiency"]
    parameter_values = pybamm.ParameterValues("Marquis2019")
    print(te_opts)

    parameter_values.search("porosity")

    parameter_values.search("Bruggeman")

    parameter_values.update(
        {
            "Negative electrode tortuosity factor (electrolyte)": 0.3 ** (-0.5),
            "Positive electrode tortuosity factor (electrolyte)": 0.3 ** (-0.5),
            "Negative electrode tortuosity factor (electrode)": 0.7 ** (-0.5),
            "Positive electrode tortuosity factor (electrode)": 0.7 ** (-0.5),
            "Separator tortuosity factor (electrolyte)": 1.0,
        },
        check_already_exists=False,
    )

    for t_label in te_opts:
        model = pybamm.lithium_ion.DFN(
            options={"transport efficiency": t_label}
        )  # Doyle-Fuller-Newman model
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sols.append(sim.solve([0, 3600]))  # solve for 1 hour

    pybamm.dynamic_plot(sols, labels=te_opts)

    np.allclose(sols[0]["Terminal voltage [V]"].data, sols[4]["Terminal voltage [V]"].data)

    parameter_values.update(
        {
            "Negative electrode tortuosity factor (electrolyte)": 4.0,
            "Positive electrode tortuosity factor (electrolyte)": 4.0,
            "Negative electrode tortuosity factor (electrode)": 3.0,
            "Positive electrode tortuosity factor (electrode)": 3.0,
            "Separator tortuosity factor (electrolyte)": 1.5,
        },
        check_already_exists=False,
    )

    model = pybamm.lithium_ion.DFN(
        options={"transport efficiency": "tortuosity factor"}
    )  # Doyle-Fuller-Newman model
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sols.append(sim.solve([0, 3600]))

    pybamm.dynamic_plot(sols, labels=[*te_opts, "higher tortuosity factor"])


def solve_the_heat_equation():
    model = pybamm.BaseModel()

    x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
    T = pybamm.Variable("Temperature", domain="rod")
    k = pybamm.Parameter("Thermal diffusivity")

    N = -k * pybamm.grad(T)  # Heat flux
    Q = 1 - pybamm.Function(np.abs, x - 1)  # Source term
    dTdt = -pybamm.div(N) + Q  # The right hand side of the PDE
    model.rhs = {T: dTdt}  # Add to model

    model.boundary_conditions = {
        T: {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }
    }

    model.initial_conditions = {T: 2 * x - x**2}

    model.variables = {"Temperature": T, "Heat flux": N, "Heat source": Q}

    geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}}}

    param = pybamm.ParameterValues({"Thermal diffusivity": 0.75})

    param.process_model(model)
    param.process_geometry(geometry)

    submesh_types = {"rod": pybamm.Uniform1DSubMesh}
    var_pts = {x: 30}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {"rod": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)

    disc.process_model(model)

    solver = pybamm.ScipySolver()
    t = np.linspace(0, 1, 100)
    solution = solver.solve(model, t)

    T_out = solution["Temperature"]

    N = 100  # number of Fourier modes to sum
    k_val = param[
        "Thermal diffusivity"
    ]  # extract value of diffusivity from the parameters dictionary


    # Fourier coefficients
    def q(n):
        return (8 / (n**2 * np.pi**2)) * np.sin(n * np.pi / 2)


    def c(n):
        return (16 / (n**3 * np.pi**3)) * (1 - np.cos(n * np.pi))


    def b(n):
        return c(n) - 4 * q(n) / (k_val * n**2 * np.pi**2)


    def T_n(t, n):
        return (4 * q(n) / (k_val * n**2 * np.pi**2)) + b(n) * np.exp(
            -k_val * (n * np.pi / 2) ** 2 * t
        )


    # Sum series to get the temperature
    def T_exact(x, t):
        out = 0
        for n in np.arange(1, N):
            out += T_n(t, n) * np.sin(n * np.pi * x / 2)
        return out

    x_nodes = mesh["rod"].nodes  # numerical gridpoints
    xx = np.linspace(0, 2, 101)  # fine mesh to plot exact solution
    plot_times = np.linspace(0, 1, 5)  # times at which to plot

    plt.figure(figsize=(15, 8))
    cmap = plt.get_cmap("inferno")
    for i, t in enumerate(plot_times):
        color = cmap(float(i) / len(plot_times))
        plt.plot(
            x_nodes,
            T_out(t, x=x_nodes),
            "o",
            color=color,
            label="Numerical" if i == 0 else "",
        )
        plt.plot(
            xx,
            T_exact(xx, t),
            "-",
            color=color,
            label=f"Exact (t={plot_times[i]})",
        )
    plt.xlabel("x", fontsize=16)
    plt.ylabel("T", fontsize=16)
    plt.legend()
    plt.show()


def Using_model_options():
    options = {"particle": "quadratic profile", "thermal": "lumped"}
    model = pybamm.lithium_ion.SPMe(options)
    param = pybamm.ParameterValues("Chen2020")
    simulation = pybamm.Simulation(model, parameter_values=param)
    simulation.solve([0, 3600])

    simulation.plot(
        [
            "Voltage [V]",
            "X-averaged cell temperature [K]",
        ]
    )


def Using_submodels():
    model = pybamm.lithium_ion.SPM()

    for name, submodel in model.submodels.items():
        print(name, submodel)

    model = pybamm.lithium_ion.SPM(build=False)

    model.submodels["negative primary particle"] = (
        pybamm.particle.XAveragedPolynomialProfile(
            model.param,
            "negative",
            options={**model.options, "particle": "uniform profile"},
        )
    )

    for name, submodel in model.submodels.items():
        print(name, submodel)

    print(model.rhs)

    model.build_model()

    print(model.rhs)

    simulation = pybamm.Simulation(model)
    simulation.solve([0, 3600])
    simulation.plot()


    model = pybamm.lithium_ion.BaseModel()

    model.submodels["external circuit"] = pybamm.external_circuit.ExplicitCurrentControl(
        model.param, model.options
    )

    model.submodels["current collector"] = pybamm.current_collector.Uniform(model.param)
    model.submodels["thermal"] = pybamm.thermal.isothermal.Isothermal(model.param)
    model.submodels["porosity"] = pybamm.porosity.Constant(model.param, model.options)
    model.submodels["negative active material"] = pybamm.active_material.Constant(
        model.param, "negative", model.options
    )
    model.submodels["positive active material"] = pybamm.active_material.Constant(
        model.param, "positive", model.options
    )

    model.submodels["negative electrode potentials"] = pybamm.electrode.ohm.LeadingOrder(
        model.param, "negative"
    )
    model.submodels["positive electrode potentials"] = pybamm.electrode.ohm.LeadingOrder(
        model.param, "positive"
    )

    options = {**model.options, "particle": "uniform profile"}
    model.submodels["negative primary particle"] = (
        pybamm.particle.XAveragedPolynomialProfile(model.param, "negative", options)
    )
    model.submodels["positive primary particle"] = (
        pybamm.particle.XAveragedPolynomialProfile(model.param, "positive", options)
    )

    model.submodels["negative total particle concentration"] = (
        pybamm.particle.TotalConcentration(model.param, "negative", options)
    )
    model.submodels["positive total particle concentration"] = (
        pybamm.particle.TotalConcentration(model.param, "positive", options)
    )

    model.submodels["negative open-circuit potential"] = (
        pybamm.open_circuit_potential.SingleOpenCircuitPotential(
            model.param, "negative", "lithium-ion main", options=model.options
        )
    )
    model.submodels["positive open-circuit potential"] = (
        pybamm.open_circuit_potential.SingleOpenCircuitPotential(
            model.param, "positive", "lithium-ion main", options=model.options
        )
    )
    model.submodels["negative interface"] = pybamm.kinetics.InverseButlerVolmer(
        model.param, "negative", "lithium-ion main", options=model.options
    )
    model.submodels["positive interface"] = pybamm.kinetics.InverseButlerVolmer(
        model.param, "positive", "lithium-ion main", options=model.options
    )
    model.submodels["negative interface current"] = (
        pybamm.kinetics.CurrentForInverseButlerVolmer(
            model.param, "negative", "lithium-ion main"
        )
    )
    model.submodels["positive interface current"] = (
        pybamm.kinetics.CurrentForInverseButlerVolmer(
            model.param, "positive", "lithium-ion main"
        )
    )
    model.submodels["negative interface utilisation"] = pybamm.interface_utilisation.Full(
        model.param, "negative", model.options
    )
    model.submodels["positive interface utilisation"] = pybamm.interface_utilisation.Full(
        model.param, "positive", model.options
    )


    model.submodels["Negative particle mechanics"] = pybamm.particle_mechanics.NoMechanics(
        model.param, "negative", model.options
    )
    model.submodels["Positive particle mechanics"] = pybamm.particle_mechanics.NoMechanics(
        model.param, "positive", model.options
    )
    model.submodels["Negative sei"] = pybamm.sei.NoSEI(
        model.param, "negative", model.options
    )
    model.submodels["Positive sei"] = pybamm.sei.NoSEI(
        model.param, "positive", model.options
    )
    model.submodels["Negative sei on cracks"] = pybamm.sei.NoSEI(
        model.param, "negative", model.options, cracks=True
    )
    model.submodels["Positive sei on cracks"] = pybamm.sei.NoSEI(
        model.param, "positive", model.options, cracks=True
    )
    model.submodels["Negative lithium plating"] = pybamm.lithium_plating.NoPlating(
        model.param, "negative"
    )
    model.submodels["Positive lithium plating"] = pybamm.lithium_plating.NoPlating(
        model.param, "positive"
    )

    model.submodels["electrolyte diffusion"] = (
        pybamm.electrolyte_diffusion.ConstantConcentration(model.param)
    )
    model.submodels["electrolyte conductivity"] = (
        pybamm.electrolyte_conductivity.LeadingOrder(model.param)
    )

    model.build_model()

    simulation = pybamm.Simulation(model)
    simulation.solve([0, 3600])
    simulation.plot()


def parameter_set_of_the_Enertech_cells():

    model = pybamm.lithium_ion.DFN(
        options={
            "particle": "Fickian diffusion",
            "cell geometry": "arbitrary",
            "thermal": "lumped",
            "particle mechanics": "swelling only",
        }
    )

    # update parameters, making C-rate and input
    param = pybamm.ParameterValues("Ai2020")
    capacity = param["Nominal cell capacity [A.h]"]
    param.update({"Current function [A]": capacity * pybamm.InputParameter("C-rate")})

    # update the mesh
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 50,
        var.x_s: 50,
        var.x_p: 50,
        var.r_n: 21,
        var.r_p: 21,
    }

    # define the simulation
    sim = pybamm.Simulation(
        model,
        var_pts=var_pts,
        parameter_values=param,
    )

    # solve for different C-rates
    Crates = [0.5, 1, 2]
    solutions = []
    for Crate in Crates:
        print(f"{Crate} C")
        sol = sim.solve(t_eval=[0, 3600 / Crate * 1.05], inputs={"C-rate": Crate})
        solutions.append(sol)

    # unpack solutions
    solution05C, solution1C, solution2C = solutions

    # load experimental results
    data_loader = pybamm.DataLoader()

    data_Disp_01C = pd.read_csv(
        data_loader.get_data("0.1C_discharge_displacement.txt"),
        delimiter="\s+",
        header=None,
    )
    data_Disp_05C = pd.read_csv(
        data_loader.get_data("0.5C_discharge_displacement.txt"),
        delimiter="\s+",
        header=None,
    )
    data_Disp_1C = pd.read_csv(
        data_loader.get_data("1C_discharge_displacement.txt"), delimiter="\s+", header=None
    )
    data_Disp_2C = pd.read_csv(
        data_loader.get_data("2C_discharge_displacement.txt"), delimiter="\s+", header=None
    )
    data_V_01C = pd.read_csv(
        data_loader.get_data("0.1C_discharge_U.txt"), delimiter="\s+", header=None
    )
    data_V_05C = pd.read_csv(
        data_loader.get_data("0.5C_discharge_U.txt"), delimiter="\s+", header=None
    )
    data_V_1C = pd.read_csv(
        data_loader.get_data("1C_discharge_U.txt"), delimiter="\s+", header=None
    )
    data_V_2C = pd.read_csv(
        data_loader.get_data("2C_discharge_U.txt"), delimiter="\s+", header=None
    )
    data_T_05C = pd.read_csv(
        data_loader.get_data("0.5C_discharge_T.txt"), delimiter="\s+", header=None
    )
    data_T_1C = pd.read_csv(
        data_loader.get_data("1C_discharge_T.txt"), delimiter="\s+", header=None
    )
    data_T_2C = pd.read_csv(
        data_loader.get_data("2C_discharge_T.txt"), delimiter="\s+", header=None
    )


    t_all2C = solution2C["Time [h]"].entries
    V_n2C = solution2C["Voltage [V]"].entries
    T_n2C = (
        solution2C["Volume-averaged cell temperature [K]"].entries
        - param["Initial temperature [K]"]
    )
    L_x2C = solution2C["Cell thickness change [m]"].entries

    t_all1C = solution1C["Time [h]"].entries
    V_n1C = solution1C["Voltage [V]"].entries
    T_n1C = (
        solution1C["Volume-averaged cell temperature [K]"].entries
        - param["Initial temperature [K]"]
    )
    L_x1C = solution1C["Cell thickness change [m]"].entries

    t_all05C = solution05C["Time [h]"].entries
    V_n05C = solution05C["Voltage [V]"].entries
    T_n05C = (
        solution05C["Volume-averaged cell temperature [K]"].entries
        - param["Initial temperature [K]"]
    )
    L_x05C = solution05C["Cell thickness change [m]"].entries

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.plot(t_all2C, V_n2C, "r-", label="Simulation")
    ax1.plot(
        data_V_2C.values[::30, 0] / 3600,
        data_V_2C.values[::30, 1],
        "ro",
        markerfacecolor="none",
        label="Experiment",
    )
    ax1.plot(t_all05C, V_n05C, "g-")
    ax1.plot(
        data_V_05C.values[::100, 0] / 3600,
        data_V_05C.values[::100, 1],
        "go",
        markerfacecolor="none",
    )
    ax1.plot(t_all1C, V_n1C, "b-")
    ax1.plot(
        data_V_1C.values[::50, 0] / 3600,
        data_V_1C.values[::50, 1],
        "bo",
        markerfacecolor="none",
    )
    ax1.legend()
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("Voltage [V]")
    ax1.text(0.1, 3.2, r"2 C", {"color": "r", "fontsize": 14})
    ax1.text(1.1, 3.2, r"1 C", {"color": "b", "fontsize": 14})
    ax1.text(1.6, 3.2, r"0.5 C", {"color": "g", "fontsize": 14})

    ax2.plot(t_all2C, T_n2C, "r-", label="Simulation")
    ax2.plot(
        data_T_2C.values[0:1754:50, 0] / 3600,
        data_T_2C.values[0:1754:50, 1],
        "ro",
        markerfacecolor="none",
        label="Experiment",
    )
    ax2.plot(t_all05C, T_n05C, "g-")
    ax2.plot(
        data_T_05C.values[0:7301:200, 0] / 3600,
        data_T_05C.values[0:7301:200, 1],
        "go",
        markerfacecolor="none",
    )
    ax2.plot(t_all1C, T_n1C, "b-")
    ax2.plot(
        data_T_1C.values[0:3598:100, 0] / 3600,
        data_T_1C.values[0:3598:100, 1],
        "bo",
        markerfacecolor="none",
    )
    ax2.legend()
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Temperature rise [K]")
    ax2.text(0.5, 8, r"2 C", {"color": "r", "fontsize": 14})
    ax2.text(0.8, 4.4, r"1 C", {"color": "b", "fontsize": 14})
    ax2.text(1.5, 2, r"0.5 C", {"color": "g", "fontsize": 14})

    ax3.plot(t_all2C, L_x2C, "r-", label="Simulation")
    ax3.plot(
        data_Disp_2C.values[0:1754:5, 0] / 3600,
        data_Disp_2C.values[0:1754:5, 1] - data_Disp_2C.values[0, 1],
        "ro",
        markerfacecolor="none",
        label="Experiment",
    )
    ax3.plot(t_all05C, L_x05C, "g-")
    ax3.plot(
        data_Disp_05C.values[0:1754:10, 0] / 3600,
        data_Disp_05C.values[0:1754:10, 1] - data_Disp_05C.values[0, 1],
        "go",
        markerfacecolor="none",
    )
    ax3.plot(t_all1C, L_x1C, "b-")
    ax3.plot(
        data_Disp_1C.values[0:1754:10, 0] / 3600,
        data_Disp_1C.values[0:1754:10, 1] - data_Disp_1C.values[0, 1],
        "bo",
        markerfacecolor="none",
    )
    ax3.legend()
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel("Thickness change [m]")
    ax3.text(0.1, -0.0001, r"2 C", {"color": "r", "fontsize": 14})
    ax3.text(0.9, -0.0001, r"1 C", {"color": "b", "fontsize": 14})
    ax3.text(1.8, -0.0001, r"0.5 C", {"color": "g", "fontsize": 14})

    plt.tight_layout()

    E_n = param["Negative electrode Young's modulus [Pa]"]
    E_p = param["Positive electrode Young's modulus [Pa]"]

    cs_n_xav = solution2C["X-averaged negative particle concentration [mol.m-3]"].entries
    cs_p_xav = solution2C["X-averaged positive particle concentration [mol.m-3]"].entries
    st_surf_n = solution2C["Negative particle surface tangential stress [Pa]"].entries / E_n
    st_surf_p = solution2C["Positive particle surface tangential stress [Pa]"].entries / E_p

    data_st_n_2C = pd.read_csv(data_loader.get_data("stn_2C.txt"), delimiter=",", header=3)
    data_st_p_2C = pd.read_csv(data_loader.get_data("stp_2C.txt"), delimiter=",", header=3)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(t_all2C, st_surf_n[-1, :], "ro", markerfacecolor="none", label="Current")
    ax1.plot(
        data_st_n_2C.values[:, 0] / 3600,
        data_st_n_2C.values[:, 1],
        "r-",
        label="Ai et al. 2020",
    )
    ax1.legend()
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("$\sigma_{t,n}/E_n$")

    ax2.plot(t_all2C, st_surf_p[0, :], "ro", markerfacecolor="none", label="Current")
    ax2.plot(
        data_st_p_2C.values[0:3601, 0] / 3600,
        data_st_p_2C.values[0:3601, 1],
        "r-",
        label="Ai et al. 2020",
    )
    ax2.legend()
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("$\sigma_{t,p}/E_p$")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # compare_PyBaMM_with_COMSOL_curves()
    # compare_with_experimental_data()
    # compare_lithium_ion_battery_models()
    # compare_particle_diffusion_models()
    # composite_electrode_particle_model()
    # model_coupled_degradation_mechanisms()
    # Doyler_Fuller_Newman_model_with_particle_size_distributions()
    # electrode_state_of_health()
    # simulate_graded_electrode()
    # half_cell_models()
    # hysteresis_state_models()
    # jelly_roll_model()
    # use_latexify()
    # lead_acid_models()
    # modelling_lithium_plating()
    # modelling_lithium_plating_on_composite_electrodes()
    # many_particle_model()
    # multi_species_multi_reaction_model()
    # pouch_cell_model()
    # generate_rate_capability_plots()
    # modelling_SEI_growth_on_particle_cracks()
    # simulate_3E_cell()
    # LG_M50_ORegan2022()
    # DFN_model_for_sodium_ion_batteries()
    # Single_Particle_Model()
    # Single_Particle_Model_with_Electrolyte()
    # Using_crack_submodels()
    # Loss_of_active_material_submodels()
    # thermal_models()
    # transport_efficiency_and_models_for_tortuosity_factor()
    # solve_the_heat_equation()
    # Using_model_options()
    # Using_submodels()
    parameter_set_of_the_Enertech_cells()
