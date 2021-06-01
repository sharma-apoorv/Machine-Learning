import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def gd_viz(f, f_plot_param, iterates):
    N = len(iterates)

    x1_min = f_plot_param["x1_min"]
    x1_max = f_plot_param["x1_max"]
    x2_min = f_plot_param["x2_min"]
    x2_max = f_plot_param["x2_max"]
    nb_points = f_plot_param["nb_points"]
    levels = f_plot_param["levels"]
    title = f_plot_param["title"]

    def f_no_vector(x1, x2):
        return f(np.array([x1, x2]))

    x, y = np.meshgrid(
        np.linspace(x1_min, x1_max, nb_points), np.linspace(x2_min, x2_max, nb_points)
    )
    z = f_no_vector(x, y)

    plt.figure()
    plt.contour(x, y, z, levels)

    # Plot iterates.
    for j in range(1, N):
        plt.annotate(
            "",
            xy=iterates[j],
            xytext=iterates[j - 1],
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1},
            va="center",
            ha="center",
        )
    plt.scatter(*zip(*iterates), c="red", s=100, lw=0)
    # plt.plot(3,1,'r*',markersize=15)
    # plt.clabel(graphe,  inline=1, fontsize=10,fmt='%3.2f')
    plt.title(title)
    plt.show()


def custom_3dplot(f, f_plot_param):

    x1_min = f_plot_param["x1_min"]
    x1_max = f_plot_param["x1_max"]
    x2_min = f_plot_param["x2_min"]
    x2_max = f_plot_param["x2_max"]
    nb_points = f_plot_param["nb_points"]
    v_min = f_plot_param["v_min"]
    v_max = f_plot_param["v_max"]

    def f_no_vector(x1, x2):
        return f(np.array([x1, x2]))

    x, y = np.meshgrid(
        np.linspace(x1_min, x1_max, nb_points), np.linspace(x2_min, x2_max, nb_points)
    )
    z = f_no_vector(x, y)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y, z, cmap=cm.hot, vmin=v_min, vmax=v_max)
    ax.set_zlim(v_min, v_max)
    plt.show()


def level_plot(f, f_plot_param):

    x1_min = f_plot_param["x1_min"]
    x1_max = f_plot_param["x1_max"]
    x2_min = f_plot_param["x2_min"]
    x2_max = f_plot_param["x2_max"]
    nb_points = f_plot_param["nb_points"]
    levels = f_plot_param["levels"]
    title = f_plot_param["title"]

    def f_no_vector(x1, x2):
        return f(np.array([x1, x2]))

    x, y = np.meshgrid(
        np.linspace(x1_min, x1_max, nb_points), np.linspace(x2_min, x2_max, nb_points)
    )
    z = f_no_vector(x, y)

    plt.figure()
    plt.contour(x, y, z, levels)
    # plt.plot(3,1,'r*',markersize=15)
    # plt.clabel(graphe,  inline=1, fontsize=10,fmt='%3.2f')
    plt.title(title)
    plt.show()


def rls_viz(L, iterates, beta_true, *args):

    # m = 20
    # theta0_true = 2
    # theta1_true = 0.5
    x = np.linspace(-1, 1, 20)
    y = beta_true[0] + beta_true[1] * x

    # The plot: LHS is the data, RHS will be the loss function.
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].scatter(x, y, marker=".", s=40, color="k")

    # First construct a grid of (theta0, theta1) parameter pairs and their
    # corresponding loss function values.
    num_points = 101
    beta0_grid = np.linspace(-1, 4, num_points)
    beta1_grid = np.linspace(-5, 5, num_points)

    L_grid = np.zeros((num_points, num_points))
    for i, beta0 in enumerate(beta0_grid):
        for j, beta1 in enumerate(beta1_grid):
            beta = np.array([beta0, beta1])
            L_grid[i, j] = L(beta, *args)

    # A labeled contour plot for the RHS loss function
    X, Y = np.meshgrid(beta0_grid, beta1_grid)
    contours = ax[1].contour(X, Y, L_grid, 10)
    ax[1].clabel(contours)
    # The target parameter values indicated on the loss function contour plot
    ax[1].scatter([beta_true[0]] * 2, [beta_true[1]] * 2, s=[50, 10], color=["k", "w"])

    N = len(iterates)

    def hypothesis(x, beta):
        return beta[0] + beta[1] * x

    # Annotate the loss function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    colors = ["b", "g", "m", "c", "orange"]
    ax[0].plot(
        x,
        hypothesis(x, iterates[0]),
        color=colors[0],
        lw=2,
        label=r"$\beta_0 = {:.3f}, \beta_1 = {:.3f}$".format(*iterates[0]),
    )
    for j in range(1, N):
        ax[1].annotate(
            "",
            xy=iterates[j],
            xytext=iterates[j - 1],
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1},
            va="center",
            ha="center",
        )
        ax[0].plot(
            x,
            hypothesis(x, iterates[j]),
            color=colors[j],
            lw=2,
            label=r"$\beta_0 = {:.3f}, \beta_1 = {:.3f}$".format(*iterates[j]),
        )
    ax[1].scatter(*zip(*iterates), c=colors, s=100, lw=0)

    # Labels, titles and a legend.
    ax[1].set_xlabel(r"$\beta_0$")
    ax[1].set_ylabel(r"$\beta_1$")
    ax[1].set_title("Loss Function")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$")
    ax[0].set_title("Data and Fit")
    axbox = ax[0].get_position()
    # Position the legend by hand so that it doesn't cover up any of the lines.
    ax[0].legend(
        loc=(axbox.x0 + 0.5 * axbox.width, axbox.y0 + 0.1 * axbox.height),
        fontsize="small",
    )

    plt.show()
