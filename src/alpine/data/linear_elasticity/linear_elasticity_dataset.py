import numpy as np
import dctkit as dt
import os
from dctkit.mesh import simplex, util
import pygmsh
from typing import Tuple, List
import numpy.typing as npt
import alpine.data.util as u
from dctkit.physics.elasticity import LinearElasticity
from dctkit.math.opt import optctrl


data_path = os.path.dirname(os.path.realpath(__file__))


def get_data(S: simplex.SimplicialComplex, lame_moduli: List[List],
             num_data_per_each_mod_couple: List[List],
             bench_names: List[str], prb_kwargs: List[List]) -> Tuple[npt.NDArray,
                                                                      npt.NDArray]:
    """Generate dataset for linear elasticity.

    Args:
        S: a simplicial complex (reference configuration).
        lame_moduli: list of lame moduli for each benchmark.
        num_data_per_each_mod_couple: list containing the number of data for each
            benchmark.
        bench_names: list containing the name of each benchmark
            (currently supported: pure tension, pure shear).

    Returns:
        a tuple containing the node coordinates of the deformed configuration and
            the benchmark name corresponding to those.
    """
    # number of lame moduli must be equal to the number of data per lame moduli
    assert len(lame_moduli) == len(num_data_per_each_mod_couple)
    # total number of data = sum of the data for each couple of lame moduli
    tot_num_data = sum([sum(num_data) for num_data in num_data_per_each_mod_couple])
    X = np.zeros((tot_num_data,
                 S.num_nodes, 3), dtype=dt.float_dtype)

    # y = []
    # NOTE: setting dtype = str truncates the string
    # y.append(np.empty(tot_num_data, dtype=object))
    y = np.zeros((tot_num_data, S.num_nodes+1, 3), dtype=object)

    # number of data points for the previous couple of lame moduli, initialized to 0
    prec_num_data = 0
    # NOTE: only needed for non homogeneous example
    gamma = 1000000.

    for k, benchmark in enumerate(bench_names):
        for i, moduli in enumerate(lame_moduli[k]):
            lambda_, mu_ = moduli
            num_data = num_data_per_each_mod_couple[k][i]
            iterable = 0.01*np.arange(1, num_data + 1, dtype=dt.float_dtype)
            true_curr_node_coords = np.zeros(
                (num_data, S.num_nodes, 3), dtype=dt.float_dtype)
            f_data = np.zeros((num_data, S.num_nodes, 3), dtype=dt.float_dtype)
            if benchmark == "pure_tension":
                for j, strain_xx in enumerate(iterable):
                    strain_yy = -(lambda_/(2*mu_+lambda_))*strain_xx
                    true_curr_node_coords[j, :, :] = S.node_coords.copy()
                    true_curr_node_coords[j, :, 0] *= 1 + strain_xx
                    true_curr_node_coords[j, :, 1] *= 1 + strain_yy
            elif benchmark == "pure_shear":
                for j, gamma_ in enumerate(iterable):
                    true_curr_node_coords[j, :, :] = S.node_coords.copy()
                    true_curr_node_coords[j, :, 0] += gamma_*S.node_coords[:, 1]
            elif benchmark == "non_homogeneous_example":
                bnodes = prb_kwargs[k][0]
                for j, a in enumerate(iterable):
                    b = 0.1*a

                    anal_node_coords = S.node_coords.copy()
                    anal_node_coords[:, 0] = S.node_coords[:, 0] - \
                        b*S.node_coords[:, 0]**2
                    anal_node_coords[:, 1] = S.node_coords[:, 1] + \
                        a*S.node_coords[:, 1]**2 - a

                    bvalues = anal_node_coords[bnodes, :]
                    boundary_values = {":": (bnodes, bvalues)}

                    f = np.zeros((S.num_nodes, 3))
                    f[:, 0] = 4*mu_*b - 2*lambda_*a + 2*lambda_*b
                    f[:, 1] = -4*mu_*a - 2*lambda_*a + 2*lambda_*b

                    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
                    obj = ela.obj_linear_elasticity_energy
                    x0 = S.node_coords.flatten()
                    obj_args = {'f': f, 'gamma': gamma,
                                'boundary_values': boundary_values}

                    prb = optctrl.OptimizationProblem(
                        dim=len(x0), state_dim=len(x0), objfun=obj)

                    prb.set_obj_args(obj_args)
                    sol = prb.solve(x0=x0, ftol_abs=1e-12,
                                    ftol_rel=1e-12, maxeval=20000)
                    curr_node_coords = sol.reshape(S.node_coords.shape)
                    true_curr_node_coords[j, :, :] = curr_node_coords
                    f_data[j, :, :] = f

            X[prec_num_data:prec_num_data+num_data, :, :] = true_curr_node_coords
            y[prec_num_data:prec_num_data + num_data, 0, :] = benchmark
            y[prec_num_data:prec_num_data + num_data, 1:, :] = f_data
            # update prec_num_data
            prec_num_data = num_data

    return X, y


if __name__ == '__main__':
    lc = 0.1
    L = 1.
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        geom.add_physical(p.lines[0], label="down")
        geom.add_physical(p.lines[1], label="right")
        geom.add_physical(p.lines[2], label="up")
        geom.add_physical(p.lines[3], label="left")
        mesh = geom.generate_mesh()

    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPD_weights()
    S.get_flat_DPP_weights()

    down_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "down")
    up_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "up")
    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")
    bnodes = left_bnd_nodes_idx + right_bnd_nodes_idx + up_bnd_nodes_idx + down_bnd_nodes_idx

    u.save_dataset(data_generator=get_data,
                   data_generator_kwargs={'S': S,
                                          'lame_moduli': [[(10, 1)], [(10, 1)]],
                                          'num_data_per_each_mod_couple': [[10], [10]],
                                          'bench_names': ["pure_shear",
                                                          "non_homogeneous_example"], 'prb_kwargs': [[], [bnodes]]},
                   perc_val=0.3,
                   perc_test=0.2,
                   format="npy")
