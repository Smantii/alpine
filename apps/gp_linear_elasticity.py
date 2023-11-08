from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from deap import gp, base
from alpine.data.util import load_dataset
from alpine.data.linear_elasticity.linear_elasticity_dataset import data_path
from dctkit.mesh import util
from alpine.gp import gpsymbreg as gps
from alpine.util import get_LE_boundary_values
from dctkit import config
import dctkit

import ray

import numpy as np
import jax.numpy as jnp
import math
import time
import sys
import yaml
from typing import Tuple, Callable
import numpy.typing as npt
import pygmsh

residual_formulation = False

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


def eval_MSE_sol(func: Callable, indlen: int, X: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, gamma: float,
                 F_0: npt.NDArray) -> Tuple[float, npt.NDArray]:

    num_data, _, _ = X.shape

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(F_flat, curr_bvalues):
        F = F_flat.reshape((2, 2))
        x_init = S.node_coords[:, :-1]
        F_stack = jnp.stack([F]*S.num_nodes)
        deformed_x = jnp.einsum("ijk, ik -> ij", F_stack, x_init)
        x_reshaped = jnp.zeros(S.node_coords.shape)
        x_reshaped = x_reshaped.at[:, :-1].set(deformed_x)
        # x_reshaped = x.reshape(S.node_coords.shape)
        penalty = 0.
        for key in curr_bvalues:
            idx, values = curr_bvalues[key]
            if key == ":":
                penalty += jnp.sum((x_reshaped[idx, :] - values)**2)
            else:
                penalty += jnp.sum((x_reshaped[idx, int(key)] - values)**2)
        penalty *= gamma
        total_energy = func(F[0, 0], F[0, 1], F[1, 0], F[1, 1]) + penalty
        return total_energy

    prb = oc.OptimizationProblem(dim=4,
                                 state_dim=4,
                                 objfun=total_energy)

    total_err = 0.

    best_sols = []

    # for i, x in enumerate(X):
    for i in range(num_data):
        # extract current bvalues
        curr_bvalues = bvalues[i]

        # set current bvalues and vec_y for the Poisson problem
        # args = {'vec_y': vec_y, 'vec_bvalues': vec_bvalues}
        args = {'curr_bvalues': curr_bvalues}
        prb.set_obj_args(args)

        # minimize the objective
        F_flat = prb.solve(x0=F_0.flatten(), maxeval=5000,
                           ftol_abs=1e-12, ftol_rel=1e-12)
        # reshape x to have a tensor
        # x = x_flatten.reshape(S.node_coords.shape)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):
            F = F_flat.reshape((2, 2))
            F_stack = jnp.stack([F]*S.num_nodes)
            deformed_x = jnp.einsum("ijk, ik -> ij", F_stack, S.node_coords[:, :-1])
            x_reshaped = np.zeros(S.node_coords.shape)
            x_reshaped[:, :-1] = deformed_x
            current_err = np.linalg.norm(x_reshaped-X[i, :])**2
            W = jnp.array([[0, jnp.e], [-jnp.e, 0]])
            F_plus_W = F + W
            func_mean_F = func(F[0, 0], F[0, 1], F[1, 0], F[1, 1])
            func_mean_F_plus_W = func(
                F_plus_W[0, 0], F_plus_W[0, 1], F_plus_W[1, 0], F_plus_W[1, 1])
            current_err += (func_mean_F - func_mean_F_plus_W)**2
        else:
            current_err = math.nan

        if math.isnan(current_err):
            total_err = 1e5
            break

        total_err += current_err

        best_sols.append(x_reshaped)

    total_err *= 1/X.shape[0]

    return 1000.*total_err, best_sols


@ray.remote(num_cpus=2)
def eval_best_sols(individual: Callable, indlen: int, X: npt.NDArray,
                   bvalues: dict, S: SimplicialComplex,
                   gamma: float, F_0: npt.NDArray, penalty: dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(individual, indlen, X, bvalues, S,
                                gamma, F_0)

    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(individual: Callable, indlen: int, X: npt.NDArray,
             bvalues: dict, S: SimplicialComplex,
             gamma: float, F_0: npt.NDArray, penalty: dict) -> float:

    MSE, _ = eval_MSE_sol(individual, indlen, X, bvalues, S, gamma, F_0)

    return MSE


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, gamma: float,
                 F_0: npt.NDArray, penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, indlen, X, bvalues, S, gamma, F_0)

    # penalty terms on length
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, bvalues: dict,
             S: SimplicialComplex, gamma: float, F_0: C.CochainP0,
             toolbox: base.Toolbox):

    indfun = toolbox.compile(expr=ind)
    dim = X.shape[0]

    _, u = eval_MSE_sol(indfun, indlen=0, X=X, bvalues=bvalues,
                        S=S, gamma=gamma, F_0=F_0)

    plt.figure(10, figsize=(10, 2))
    plt.clf()
    fig = plt.gcf()
    _, axes = plt.subplots(1, dim, num=10)
    for i in range(dim):
        axes[i].triplot(S.node_coords[:, 0], S.node_coords[:, 1],
                        triangles=S.S[2], color="#e5f5e0")
        axes[i].triplot(u[i][:, 0], u[i][:, 1],
                        triangles=S.S[2], color="#a1d99b")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


def gp_linear_elasticity(config_file, output_path=None):
    # generate mesh
    lc = 0.1
    L = 1.
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        geom.add_physical(p.lines[0], label="down")
        geom.add_physical(p.lines[2], label="up")
        geom.add_physical(p.lines[1], label="right")
        geom.add_physical(p.lines[3], label="left")
        mesh = geom.generate_mesh()

    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPD_weights()
    S.get_flat_DPP_weights()

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_path, "npy")

    # set bc
    ref_node_coords = S.node_coords

    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")
    down_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "down")
    up_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "up")

    # FIXME: just to initialize ref_metric_contravariant.
    # Write a routine in simplex that does it
    _ = S.get_deformation_gradient(ref_node_coords)

    # define a dictionary containing boundary nodes information (needed to set properly
    #  boundary_values)
    boundary_nodes_info = {'left_bnd_nodes_idx': left_bnd_nodes_idx,
                           'right_bnd_nodes_idx': right_bnd_nodes_idx,
                           'up_bnd_nodes_idx': up_bnd_nodes_idx,
                           'down_bnd_nodes_idx': down_bnd_nodes_idx}

    # extract boundary values
    bvalues_train = get_LE_boundary_values(X=X_train,
                                           y=y_train,
                                           ref_node_coords=ref_node_coords,
                                           boundary_nodes_info=boundary_nodes_info)
    bvalues_val = get_LE_boundary_values(X=X_val,
                                         y=y_val,
                                         ref_node_coords=ref_node_coords,
                                         boundary_nodes_info=boundary_nodes_info)
    bvalues_test = get_LE_boundary_values(X=X_test,
                                          y=y_test,
                                          ref_node_coords=ref_node_coords,
                                          boundary_nodes_info=boundary_nodes_info)

    # penalty parameter for the Dirichlet bcs
    gamma = 1000000.

    # initial guess for the solution of the problem
    F_0 = jnp.identity(2)

    # define primitive set and add primitives and terminals
    pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float], float)
    # ones cochain
    pset.addTerminal(C.Cochain(S.num_nodes, True, S, np.ones(
        S.num_nodes, dtype=dctkit.float_dtype)), C.Cochain, name="F")

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(1., float, name="1.")
    pset.addTerminal(2., float, name="2.")
    pset.addTerminal(10., float, name="10.")
    pset.addTerminal(0.1, float, name="0.1")

    # rename arguments
    pset.renameArguments(ARG0="F11", ARG1="F12", ARG2="F21", ARG3="F22")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file)

    penalty = config_file["gp"]["penalty"]

    GPprb.store_eval_common_params({'S': S, 'penalty': penalty,
                                    'gamma': gamma, 'F_0': F_0})

    params_names = ('X', 'bvalues')
    datasets = {'train': [X_train, bvalues_train],
                'val': [X_val, bvalues_val],
                'test': [X_test, bvalues_test]}
    GPprb.store_eval_dataset_params(params_names, datasets)

    GPprb.register_eval_funcs(fitness=eval_fitness.remote, error_metric=eval_MSE.remote,
                              test_sols=eval_best_sols.remote)

    if GPprb.plot_best:
        GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val,
                               bvalues=bvalues_val, S=S, gamma=gamma, F_0=F_0,
                               toolbox=GPprb.toolbox)

    GPprb.register_map([len])

    start = time.perf_counter()

    # opt_string = "AddF(MulF(2.,tr_E_sq), MulF(10., tr_sq_E))"
    # half_tr_F_sq = "MulF(0.5, AddF(SquareF(F11), AddF(MulF(F12, F21),
    # AddF(MulF(F21, F12), SquareF(F22)))))"
    # half_tr_F_FT = "MulF(0.5, AddF(SquareF(F11), AddF(SquareF(F21),
    # AddF(SquareF(F12), SquareF(F22)))))"
    # double_tr_F = "MulF(2., AddF(F11, F22))"
    # tr_E_sq = "AddF(AddF(half_tr_F_sq, SubF(half_tr_F_FT, double_tr_F)), 2.)"
    # tr_E_sq = tr_E_sq.replace("half_tr_F_sq", half_tr_F_sq)
    # tr_E_sq = tr_E_sq.replace("half_tr_F_FT", half_tr_F_FT)
    # tr_E_sq = tr_E_sq.replace("double_tr_F", double_tr_F)
    # tr_sq_E = "SquareF(AddF(F11, SubF(F22, 2.)))"
    # opt_string = opt_string.replace("tr_E_sq", tr_E_sq)
    # opt_string = opt_string.replace("tr_sq_E", tr_sq_E)
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPprb.run(print_log=True, seed=None,
              save_best_individual=True, save_train_fit_history=True,
              save_best_test_sols=True, X_test_param_name="X",
              output_path=output_path)

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))

    # path for output data speficified
    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."

    gp_linear_elasticity(config_file, output_path)
