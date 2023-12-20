
import pygmo as pg
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from ezc3d import c3d
import time
import pickle
import sys

sys.path.append("../")
sys.path.append("../casadi/")
sys.path.append("../../Dynamique/casadi")
sys.path.append("../../Dynamique/iterative_stabilisation")
sys.path.append("../../Dynamique/data_treatment")
from enums import InitialGuessType
from Optim_35_essais_kM_regul_koblique import Param_fixe, list2tab, Spring_bouts, Spring_bouts_croix, tab2list
from modele_dynamique_nxm_DimensionsReelles import (
    Points_ancrage_repos,
    multiple_shooting_integration,
)
from Optim_multi_essais_kM_regul_koblique import m_bounds, k_bounds, load_training_pool_data
from optim_dynamique_withoutC_casadi import get_list_results_dynamic, Pt_bounds, F_bounds
from iterative_stabilisation import position_the_points_based_on_the_force
from pygmo_optim_with_iterative_stabilisation import global_optimisation, solve

n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


##########################################################################################################################

def main():

    initial_guess = InitialGuessType.SURFACE_INTERPOLATION
    WITH_K_OBLIQUE = False

    Pts_collecte, Pts_interpolated, Pts_ancrage_interpolated, dict_fixed_params, labels, inds_masse, Masse_centre, Pt_repos, Pt_ancrage_repos, trial_names = load_training_pool_data(initial_guess)

    global_optim = global_optimisation(
        Pts_collecte,
        Pts_interpolated,
        Pts_ancrage_interpolated,
        dict_fixed_params,
        labels,
        inds_masse,
        Masse_centre,
        Pt_repos,
        Pt_ancrage_repos,
        None,
        WITH_K_OBLIQUE=WITH_K_OBLIQUE,
    )
    prob = pg.problem(global_optim)
    w_opt, cost = solve(prob, global_optim)


    embed()
    if global_optim.WITH_K_OBLIQUE:
        K = np.array(w_opt[:15])
        offset = 15
    else:
        K = np.array(w_opt[:8])
        offset = 8

    Ma = w_opt[offset:offset+5]
    offset += 5
    for i in range(1, len(trial_names)):
        Ma = np.hstack((Ma, w_opt[offset:offset+5]))
        offset += 5

    data = {"Ma": Ma,
            "K": K,
            "w_opt": w_opt,
            "cost": cost,
            "trial_names": trial_names,
            }
    with open(f"results/static_multi_essai.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
