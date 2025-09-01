import numpy as np
import matplotlib.pyplot as plt
import pybamm
import os
import pandas as pd
import json
import timeit
import scipy.interpolate as interp
from pybamm import constants, exp
import scipy
from datetime import datetime
# --------------------------------------------------------------------

def Callbacks():
    model = pybamm.lithium_ion.DFN()
    experiment = pybamm.Experiment(
        [
            (
                "Discharge at C/5 for 10 hours or until 3.3 V",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 10 mA",
            ),
        ]
        * 3
    )
    sim = pybamm.Simulation(model, experiment=experiment)

    pybamm.set_logging_level("NOTICE")
    sim.solve()


    # callback = pybamm.callbacks.LoggingCallback("output.log")
    # sim.solve(callbacks=callback)
    #
    # # Read the file that has been written, which was saved to callback.logfile
    # with open(callback.logfile) as f:
    #     print(f.read())
    #
    # # Remove the log file
    # os.remove(callback.logfile)

    class CustomCallback(pybamm.callbacks.Callback):
        def on_experiment_end(self, logs):
            print(f"We are at the end of the simulation. Logs are {logs}")

    # Note the default `LoggingCallback` is also called
    sim.solve(callbacks=CustomCallback())


def custom_experiments():
    # Set up model and parameters
    model = pybamm.lithium_ion.DFN()
    # add anode potential as a variable
    # we use the potential at the separator interface since that is the minimum potential
    # during charging (plating is most likely to occur first at the separator interface)
    model.variables["Anode potential [V]"] = model.variables[
        "Negative electrode surface potential difference at separator interface [V]"
    ]
    parameter_values = pybamm.ParameterValues("Chen2020")


    # Create a custom termination event for the anode potential cut-off at 0.02V
    # We use 0.02V instead of 0V to give a safety factor
    def anode_potential_cutoff(variables):
        return variables["Anode potential [V]"] - 0.02


    # The CustomTermination class takes a name and function
    anode_potential_termination = pybamm.step.CustomTermination(
        name="Anode potential cut-off [V]", event_function=anode_potential_cutoff
    )

    # Provide a list of termination events, each step will stop whenever the first
    # termination event is reached
    terminations = [anode_potential_termination, "4.2V"]

    # Set up multi-step CC experiment with the customer terminations followed
    # by a voltage hold
    experiment = pybamm.Experiment(
        [
            (
                pybamm.step.c_rate(-1, termination=terminations),
                pybamm.step.c_rate(-0.5, termination=terminations),
                pybamm.step.c_rate(-0.25, termination=terminations),
                "Hold at 4.2V until C/50",
            )
        ]
    )

    # Set up simulation
    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)

    # for a charge we start as SOC 0
    sim.solve(initial_soc=0)


    # Plot
    plot = pybamm.QuickPlot(
        sim.solution, ["Current [A]", "Voltage [V]", "Anode potential [V]"]
    )
    plot.plot(0)

    # Plot the limits used in the termination events to check they are not surpassed
    plot.axes.by_variable("Voltage [V]").axhline(4.2, color="k", linestyle=":")
    plot.axes.by_variable("Anode potential [V]").axhline(0.02, color="k", linestyle=":")

    for i, step in enumerate(sim.solution.cycles[0].steps):
        print(f"Step {i}: {step.termination}")

    def custom_step_power(variables):
        target_power = 4
        voltage = variables["Voltage [V]"]
        return target_power / voltage


    # Run for 10 minutes and plot
    step = pybamm.step.CustomStepExplicit(custom_step_power, duration=600)
    sol = pybamm.Simulation(
        model, experiment=step, parameter_values=parameter_values
    ).solve()
    pybamm.QuickPlot(sol, ["Current [A]", "Voltage [V]", "Power [W]"]).plot(0)


    step = pybamm.step.CustomStepExplicit(
        custom_step_power, termination="2.5V", direction="discharge"
    )
    sol = pybamm.Simulation(
        model, experiment=step, parameter_values=parameter_values
    ).solve()
    pybamm.QuickPlot(sol, ["Current [A]", "Voltage [V]", "Power [W]"]).plot(0)

    def constant_voltage(variables):
        return variables["Voltage [V]"] - 3.8


    step = pybamm.step.CustomStepImplicit(constant_voltage, duration=600)
    sol = pybamm.Simulation(
        model, experiment=step, parameter_values=parameter_values
    ).solve()
    pybamm.QuickPlot(sol, ["Current [A]", "Voltage [V]"]).plot(0)

    def custom_voltage(variables):
        return 100 * (variables["Voltage [V]"] - 3.8)


    step = pybamm.step.CustomStepImplicit(
        custom_voltage, duration=600, control="differential"
    )
    sol = pybamm.Simulation(model, experiment=step).solve()
    pybamm.QuickPlot(sol, ["Current [A]", "Voltage [V]"]).plot(0)

    # Create a function for the anode potential cut-off at 0.02V
    def anode_potential_cutoff(variables):
        return variables["Anode potential [V]"] - 0.02


    # We can reuse the same function to create both the termination event and the step
    anode_potential_step = pybamm.step.CustomStepImplicit(
        anode_potential_cutoff, direction="charge", termination="4.2V"
    )
    anode_potential_termination = pybamm.step.CustomTermination(
        name="Anode potential cut-off [V]", event_function=anode_potential_cutoff
    )


    # Charge with constant current, then constant anode potential, then constant voltage
    def run_experiment(c_rate):
        # Create the experiment
        experiment = pybamm.Experiment(
            [
                (
                    # include the 4.2V termination event in case the anode potential cut-off is not reached
                    pybamm.step.c_rate(
                        -c_rate, termination=[anode_potential_termination, "4.2V"]
                    ),
                    anode_potential_step,
                    "Hold at 4.2V until C/50",
                )
            ]
        )

        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )

        # for a charge we start as SOC 0
        sim.solve(initial_soc=0)

        # Plot
        pybamm.QuickPlot(
            sim.solution, ["Current [A]", "Voltage [V]", "Anode potential [V]"]
        ).plot(0)


    run_experiment(c_rate=1)

    run_experiment(c_rate=0.2)
    plt.show()


def experiments_with_start_time():
    model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(
        ["Discharge at 1C for 20 minutes", "Charge at C/3 for 10 minutes"]
    )
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve()
    sim.plot()

    s = pybamm.step.string

    experiment = pybamm.Experiment(
        [
            s("Discharge at 1C for 1 hour", start_time=datetime(1, 1, 1, 8, 0, 0)),
            s("Charge at C/3 for 10 minutes", start_time=datetime(1, 1, 1, 8, 30, 0)),
            s("Discharge at C/2 for 30 minutes", start_time=datetime(1, 1, 1, 9, 0, 0)),
            s("Rest for 1 hour"),
        ]
    )
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve()
    sim.plot()

    datetime.strptime("2023-01-02 8:30:00", "%Y-%m-%d %H:%M:%S")


def degradation_experiments_with_reference_performance_tests():

    model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
    parameter_values = pybamm.ParameterValues("Mohtat2020")
    parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-14})

    N = 10
    cccv_experiment = pybamm.Experiment(
        [
            (
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
            )
        ]
        * N
    )
    charge_experiment = pybamm.Experiment(
        [
            (
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",
            )
        ]
    )
    rpt_experiment = pybamm.Experiment([("Discharge at C/3 until 3V",)])

    sim = pybamm.Simulation(
        model, experiment=cccv_experiment, parameter_values=parameter_values
    )
    cccv_sol = sim.solve()
    sim = pybamm.Simulation(
        model, experiment=charge_experiment, parameter_values=parameter_values
    )
    charge_sol = sim.solve(starting_solution=cccv_sol)
    sim = pybamm.Simulation(
        model, experiment=rpt_experiment, parameter_values=parameter_values
    )
    rpt_sol = sim.solve(starting_solution=charge_sol)

    pybamm.dynamic_plot(rpt_sol.cycles[-1], ["Current [A]", "Voltage [V]"])

    pybamm.plot_summary_variables(rpt_sol)

    cccv_sols = []
    charge_sols = []
    rpt_sols = []
    M = 5
    for i in range(M):
        if i != 0:  # skip the first set of ageing cycles because it's already been done
            sim = pybamm.Simulation(
                model, experiment=cccv_experiment, parameter_values=parameter_values
            )
            cccv_sol = sim.solve(starting_solution=rpt_sol)
            sim = pybamm.Simulation(
                model, experiment=charge_experiment, parameter_values=parameter_values
            )
            charge_sol = sim.solve(starting_solution=cccv_sol)
            sim = pybamm.Simulation(
                model, experiment=rpt_experiment, parameter_values=parameter_values
            )
            rpt_sol = sim.solve(starting_solution=charge_sol)
        cccv_sols.append(cccv_sol)
        charge_sols.append(charge_sol)
        rpt_sols.append(rpt_sol)

    pybamm.dynamic_plot(rpt_sols[-1].cycles[-1], ["Current [A]", "Voltage [V]"])


    cccv_cycles = []
    cccv_capacities = []
    rpt_cycles = []
    rpt_capacities = []
    for i in range(M):
        for j in range(N):
            cccv_cycles.append(i * (N + 2) + j + 1)
            start_capacity = (
                rpt_sol.cycles[i * (N + 2) + j]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[0]
            )
            end_capacity = (
                rpt_sol.cycles[i * (N + 2) + j]
                .steps[2]["Discharge capacity [A.h]"]
                .entries[-1]
            )
            cccv_capacities.append(end_capacity - start_capacity)
        rpt_cycles.append((i + 1) * (N + 2))
        start_capacity = rpt_sol.cycles[(i + 1) * (N + 2) - 1][
            "Discharge capacity [A.h]"
        ].entries[0]
        end_capacity = rpt_sol.cycles[(i + 1) * (N + 2) - 1][
            "Discharge capacity [A.h]"
        ].entries[-1]
        rpt_capacities.append(end_capacity - start_capacity)
    plt.scatter(cccv_cycles, cccv_capacities, label="Ageing cycles")
    plt.scatter(rpt_cycles, rpt_capacities, label="RPT cycles")
    plt.xlabel("Cycle number")
    plt.ylabel("Discharge capacity [A.h]")
    plt.legend()

    pybamm.plot_summary_variables(rpt_sol)
    plt.show()


def Simulate_long_experiments():
    parameter_values = pybamm.ParameterValues("Mohtat2020")
    parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-14})
    spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})

    # Calculate stoichiometries at 100% SOC
    parameter_values.set_initial_stoichiometries(1)

    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",
            )
        ]
    )
    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)
    sol = sim.solve()

    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",
            )
        ]
        * 500,
        termination="80% capacity",
    )
    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)
    sol = sim.solve()

    sol.plot(["Current [A]", "Voltage [V]"])

    sorted(sol.summary_variables.all_variables)

    print(spm.summary_variables)

    pybamm.plot_summary_variables(sol)

    # With integer
    sol_int = sim.solve(save_at_cycles=5)
    # With list
    sol_list = sim.solve(save_at_cycles=[30, 45, 55])

    print(sol_int.cycles)

    print(sol_list.cycles)

    sol_list.cycles[44].plot(["Current [A]", "Voltage [V]"])


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for cycle in sol_int.cycles:
        if cycle is not None:
            t = cycle["Time [h]"].data - cycle["Time [h]"].data[0]
            ax[0].plot(t, cycle["Current [A]"].data)
            ax[0].set_xlabel("Time [h]")
            ax[0].set_title("Current [A]")
            ax[1].plot(t, cycle["Voltage [V]"].data)
            ax[1].set_xlabel("Time [h]")
            ax[1].set_title("Voltage [V]")

    pybamm.plot_summary_variables(sol_list)

    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",
            )
        ]
        * 10,
        termination="80% capacity",
    )
    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)
    sol = sim.solve()

    sol2 = sim.solve(starting_solution=sol)

    print(len(sol2.cycles))

    plt.show()


if __name__ == '__main__':
    # Callbacks()
    # custom_experiments()
    # experiments_with_start_time()
    # degradation_experiments_with_reference_performance_tests()
    Simulate_long_experiments()
