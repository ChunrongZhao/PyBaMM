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

def finite_volume_discretisation():
    parameter_values = pybamm.ParameterValues(
        values={
            "Negative particle radius [m]": 0.5,
            "Positive particle radius [m]": 0.6,
            "Negative electrode thickness [m]": 0.3,
            "Separator thickness [m]": 0.2,
            "Positive electrode thickness [m]": 0.3,
        }
    )

    geometry = pybamm.battery_geometry()
    parameter_values.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.Uniform1DSubMesh,
        "separator": pybamm.Uniform1DSubMesh,
        "positive electrode": pybamm.Uniform1DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
        "current collector": pybamm.SubMesh0D,
    }

    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 15, var.x_s: 10, var.x_p: 15, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    spatial_methods = {
        "macroscale": pybamm.FiniteVolume(),
        "negative particle": pybamm.FiniteVolume(),
        "positive particle": pybamm.FiniteVolume(),
    }
    disc = pybamm.Discretisation(mesh, spatial_methods)

    # Set up
    macroscale = ["negative electrode", "separator", "positive electrode"]
    x_var = pybamm.SpatialVariable("x", domain=macroscale)
    r_var = pybamm.SpatialVariable("r", domain=["negative particle"])

    # Discretise
    x_disc = disc.process_symbol(x_var)
    r_disc = disc.process_symbol(r_var)
    print(f"x_disc is a {type(x_disc)}")
    print(f"r_disc is a {type(r_disc)}")

    # Evaluate
    x = x_disc.evaluate()
    r = r_disc.evaluate()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(x, "*")
    ax1.set_xlabel("index")
    ax1.set_ylabel(r"$x$")

    ax2.plot(r, "*")
    ax2.set_xlabel("index")
    ax2.set_ylabel(r"$r$")

    plt.tight_layout()
    plt.show()

    y_macroscale = x**3 / 3
    y_microscale = np.cos(r)
    y_scalar = np.array([[5]])

    y = np.concatenate([y_macroscale, y_microscale, y_scalar])

    u = pybamm.Variable(
        "u", domain=macroscale
    )  # u is a variable in the macroscale (e.g. electrolyte potential)
    v = pybamm.Variable(
        "v", domain=["negative particle"]
    )  # v is a variable in the negative particle (e.g. particle concentration)
    w = pybamm.Variable(
        "w"
    )  # w is a variable without a domain (e.g. time, average concentration)

    variables = [u, v, w]

    try:
        u.evaluate()
    except NotImplementedError as e:
        print(e)

    # Pass the list of variables to the discretisation to calculate the slices to be used (order matters here!)
    disc.set_variable_slices(variables)

    # Discretise the variables
    u_disc = disc.process_symbol(u)
    v_disc = disc.process_symbol(v)
    w_disc = disc.process_symbol(w)

    # Print the outcome
    print(f"Discretised u is the StateVector {u_disc}")
    print(f"Discretised v is the StateVector {v_disc}")
    print(f"Discretised w is the StateVector {w_disc}")

    x_fine = np.linspace(x[0], x[-1], 1000)
    r_fine = np.linspace(r[0], r[-1], 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(x_fine, x_fine**3 / 3, x, u_disc.evaluate(y=y), "o")
    ax1.set_xlabel("x")
    ax1.legend(["x^3/3", "u"], loc="best")

    ax2.plot(r_fine, np.cos(r_fine), r, v_disc.evaluate(y=y), "o")
    ax2.set_xlabel("r")
    ax2.legend(["cos(r)", "v"], loc="best")

    plt.tight_layout()
    plt.show()

    grad_u = pybamm.grad(u)
    grad_u_disc = disc.process_symbol(grad_u)
    grad_u_disc.render()


    macro_mesh = mesh.combine_submeshes(*macroscale)
    print("gradient matrix is:\n")
    print(
        f"1/dx *\n{macro_mesh.d_nodes[:, np.newaxis] * grad_u_disc.children[0].entries.toarray()}"
    )

    x_edge = macro_mesh.edges[1:-1]  # note that grad_u_disc is evaluated on the node edges

    fig, ax = plt.subplots()
    ax.plot(x_fine, x_fine**2, x_edge, grad_u_disc.evaluate(y=y), "o")
    ax.set_xlabel("x")
    legend = ax.legend(["x^2", "grad(u).evaluate(y=x**3/3)"], loc="best")

    plt.show()

    print(v.domain)

    grad_v = pybamm.grad(v)
    grad_v_disc = disc.process_symbol(grad_v)
    print("grad(v) tree is:\n")
    grad_v_disc.render()

    micro_mesh = mesh["negative particle"]
    print("\n gradient matrix is:\n")
    print(
        f"1/dr *\n{micro_mesh.d_nodes[:, np.newaxis] * grad_v_disc.children[0].entries.toarray()}"
    )

    r_edge = micro_mesh.edges[1:-1]  # note that grad_u_disc is evaluated on the node edges

    fig, ax = plt.subplots()
    ax.plot(r_fine, -np.sin(r_fine), r_edge, grad_v_disc.evaluate(y=y), "o")
    ax.set_xlabel("x")
    legend = ax.legend(["-sin(r)", "grad(v).evaluate(y=cos(r))"], loc="best")

    plt.show()


    disc.bcs = {
        u: {
            "left": (pybamm.Scalar(1), "Dirichlet"),
            "right": (pybamm.Scalar(2), "Dirichlet"),
        }
    }
    grad_u_disc = disc.process_symbol(grad_u)
    print("The gradient object is:")
    (grad_u_disc.render())
    u_eval = grad_u_disc.evaluate(y=y)
    dx = np.diff(macro_mesh.nodes)[-1]
    print(f"The value of u on the left-hand boundary is {y[0] - dx * u_eval[0] / 2}")
    print(f"The value of u on the right-hand boundary is {y[1] + dx * u_eval[-1] / 2}")


    disc.bcs = {
        u: {"left": (pybamm.Scalar(3), "Neumann"), "right": (pybamm.Scalar(4), "Neumann")}
    }
    grad_u_disc = disc.process_symbol(grad_u)
    print("The gradient object is:")
    (grad_u_disc.render())
    grad_u_eval = grad_u_disc.evaluate(y=y)
    print(f"The gradient on the left-hand boundary is {grad_u_eval[0]}")
    print(f"The gradient of u on the right-hand boundary is {grad_u_eval[-1]}")


    disc.bcs = {
        u: {"left": (pybamm.Scalar(5), "Dirichlet"), "right": (pybamm.Scalar(6), "Neumann")}
    }
    grad_u_disc = disc.process_symbol(grad_u)
    print("The gradient object is:")
    (grad_u_disc.render())
    grad_u_eval = grad_u_disc.evaluate(y=y)
    u_eval = grad_u_disc.children[1].evaluate(y=y)
    print(f"The value of u on the left-hand boundary is {(u_eval[0] + u_eval[1]) / 2}")
    print(f"The gradient on the right-hand boundary is {grad_u_eval[-1]}")

    disc.bcs = {
        u: {"left": (pybamm.Scalar(-1), "Neumann"), "right": (pybamm.Scalar(1), "Neumann")}
    }


    div_grad_u = pybamm.div(grad_u)
    div_grad_u_disc = disc.process_symbol(div_grad_u)
    div_grad_u_disc.render()


    print("div(grad) matrix is:\n")
    print(
        "1/dx^2 * \n{}".format(
            macro_mesh.d_edges[:, np.newaxis] ** 2
            * div_grad_u_disc.right.left.entries.toarray()
        )
    )

    int_u = pybamm.Integral(u, x_var)
    int_u_disc = disc.process_symbol(int_u)
    print(f"int(u) = {int_u_disc.evaluate(y=y)} is approximately equal to 1/12, {1 / 12}")

    # We divide v by r to evaluate the integral more easily
    int_v_over_r2 = pybamm.Integral(v / r_var**2, r_var)
    int_v_over_r2_disc = disc.process_symbol(int_v_over_r2)
    print(
        f"int(v/r^2) = {int_v_over_r2_disc.evaluate(y=y)} is approximately equal to 4 * pi * sin(1), {4 * np.pi * np.sin(1)}"
    )

    print("int(u):\n")
    int_u_disc.render()
    print("\nint(v):\n")
    int_v_over_r2_disc.render()


    int_u_disc.children[0].evaluate() / macro_mesh.d_edges


    model = pybamm.BaseModel()

    c_e = pybamm.Variable("electrolyte concentration", domain=macroscale)
    N_e = pybamm.grad(c_e)
    c_s = pybamm.Variable("particle concentration", domain=["negative particle"])
    N_s = pybamm.grad(c_s)
    model.rhs = {c_e: pybamm.div(N_e) - 5, c_s: pybamm.div(N_s)}
    model.boundary_conditions = {
        c_e: {"left": (np.cos(0), "Neumann"), "right": (np.cos(10), "Neumann")},
        c_s: {"left": (0, "Neumann"), "right": (-1, "Neumann")},
    }
    model.initial_conditions = {c_e: 1 + 0.1 * pybamm.sin(10 * x_var), c_s: 1}

    # Create a new discretisation and process model
    disc2 = pybamm.Discretisation(mesh, spatial_methods)
    disc2.process_model(model)

    c_e_0 = model.initial_conditions[c_e].evaluate()
    c_s_0 = model.initial_conditions[c_s].evaluate()
    y0 = model.concatenated_initial_conditions.evaluate()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    ax1.plot(x_fine, 1 + 0.1 * np.sin(10 * x_fine), x, c_e_0, "o")
    ax1.set_xlabel("x")
    ax1.legend(["1+0.1*sin(10*x)", "c_e_0"], loc="best")

    ax2.plot(x_fine, np.ones_like(r_fine), r, c_s_0, "o")
    ax2.set_xlabel("r")
    ax2.legend(["1", "c_s_0"], loc="best")

    ax3.plot(y0, "*")
    ax3.set_xlabel("index")
    ax3.set_ylabel("y0")

    plt.tight_layout()
    plt.show()


    rhs_c_e = model.rhs[c_e].evaluate(0, y0)
    rhs_c_s = model.rhs[c_s].evaluate(0, y0)
    rhs = model.concatenated_rhs.evaluate(0, y0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    ax1.plot(x_fine, -10 * np.sin(10 * x_fine) - 5, x, rhs_c_e, "o")
    ax1.set_xlabel("x")
    ax1.set_ylabel("rhs_c_e")
    ax1.legend(["1+0.1*sin(10*x)", "c_e_0"], loc="best")

    ax2.plot(r, rhs_c_s, "o")
    ax2.set_xlabel("r")
    ax2.set_ylabel("rhs_c_s")

    ax3.plot(rhs, "*")
    ax3.set_xlabel("index")
    ax3.set_ylabel("rhs")

    plt.tight_layout()
    plt.show()


    model = pybamm.BaseModel()

    # Define concentration and velocity
    c = pybamm.Variable(
        "c", domain=["negative electrode", "separator", "positive electrode"]
    )
    v = pybamm.PrimaryBroadcastToEdges(
        1, ["negative electrode", "separator", "positive electrode"]
    )
    model.rhs = {c: -pybamm.div(c * v) + 1}
    model.initial_conditions = {c: 0}
    model.boundary_conditions = {c: {"left": (0, "Dirichlet")}}
    model.variables = {"c": c}


    def solve_and_plot(model):
        model_disc = disc.process_model(model, inplace=False)

        t_eval = [0, 100]
        solution = pybamm.IDAKLUSolver().solve(model_disc, t_eval)

        # plot
        plot = pybamm.QuickPlot(solution, ["c"], spatial_unit="m")
        plot.dynamic_plot()


    solve_and_plot(model)


    model.rhs = {c: -pybamm.div(pybamm.upwind(c) * v) + 1}
    solve_and_plot(model)

    model.rhs = {c: -pybamm.div(pybamm.downwind(c) * (-v)) + 1}
    model.boundary_conditions = {c: {"right": (0, "Dirichlet")}}
    solve_and_plot(model)


if __name__ == '__main__':
    finite_volume_discretisation()
