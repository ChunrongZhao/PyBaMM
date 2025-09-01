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
import matplotlib as mpl
from cycler import cycler
# --------------------------------------------------------------------
def customizing_quickplot():
    models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe(), pybamm.lithium_ion.DFN()]
    sims = []
    for model in models:
        sim = pybamm.Simulation(model)
        sim.solve([0, 3600])
        sims.append(sim)

    pybamm.dynamic_plot(sims)

    print(plt.style.available)
    plt.style.use("ggplot")
    pybamm.settings.max_words_in_line = 3
    pybamm.dynamic_plot(sims)

    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["axes.prop_cycle"] = cycler("color", ["k", "g", "c"])
    pybamm.dynamic_plot(sims)

    pybamm.settings.max_words_in_line = 4

    plot = pybamm.QuickPlot(sims, figsize=(14, 7))
    plot.plot(0.5)  # time in hours

    # Move title to ylabel
    for ax in plot.fig.axes:
        title = ax.get_title()
        ax.set_title("")
        ax.set_ylabel(title)

    # Remove old legend and add a new one in the bottom
    leg = plot.fig.get_children()[-1]
    leg.set_visible(False)
    plot.fig.legend(plot.labels, loc="lower center", ncol=len(plot.labels), fontsize=11)

    # Adjust layout
    plot.gridspec.tight_layout(plot.fig, rect=[0, 0.04, 1, 1])
    plt.show()


def plot_voltage_components():
    model = pybamm.lithium_ion.DFN()

    experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])

    sim = pybamm.Simulation(
        model, experiment=experiment, parameter_values=pybamm.ParameterValues("Chen2020")
    )
    sol = sim.solve()

    sol.plot(
        [
            "Negative electrode bulk open-circuit potential [V]",
            "Positive electrode bulk open-circuit potential [V]",
            "Negative particle concentration overpotential [V]",
            "Positive particle concentration overpotential [V]",
            "X-averaged negative electrode reaction overpotential [V]",
            "X-averaged positive electrode reaction overpotential [V]",
            "X-averaged concentration overpotential [V]",
            "X-averaged electrolyte ohmic losses [V]",
            "X-averaged negative electrode ohmic losses [V]",
            "X-averaged positive electrode ohmic losses [V]",
        ],
    )

    sol.plot_voltage_components(split_by_electrode=True)

    sol.plot_voltage_components()


if __name__ == '__main__':
    # customizing_quickplot()
    plot_voltage_components()
