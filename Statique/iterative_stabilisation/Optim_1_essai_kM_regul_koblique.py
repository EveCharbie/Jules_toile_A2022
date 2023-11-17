"""
"""

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
sys.path.append("../../Dynamique/")
from enums import InitialGuessType
from Optim_35_essais_kM_regul_koblique import Param_fixe, list2tab, Spring_bouts, Spring_bouts_croix, tab2list
from modele_dynamique_nxm_DimensionsReelles import (
    Points_ancrage_repos,
    multiple_shooting_integration,
)
from Optim_multi_essais_kM_regul_koblique import m_bounds, k_bounds
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

    # RÉSULTATS COLLECTE :
    frame = 700
    participant = 0  # 0 #1 #2
    nb_disques = 8  # entre 1 et 11
    trial_name = "labeled_statique_centrefront_D" + str(nb_disques)
    empty_trial_name = "labeled_statique_centrefront_vide"
    if "front" not in trial_name:
        empty_trial_name = "labeled_statique_vide"

    # MASSE
    if participant != 0:
        masses = [64.5, 87.2]
        Masse_centre = masses[participant - 1]
        print("masse appliquée pour le participant " + str(participant) + " = " + str(Masse_centre) + " kg")

    if participant == 0:
        masses = [0, 27.0, 47.1, 67.3, 87.4, 102.5, 122.6, 142.8, 163.0, 183.1, 203.3, 228.6]
        Masse_centre = masses[nb_disques]
        print("masse appliquée pour " + str(nb_disques) + " disques = " + str(Masse_centre) + " kg")
        print("essai à vide : " + str(empty_trial_name))

    ##########################################################################################################################

    start_main = time.time()

    dict_fixed_params = Param_fixe()
    Fs_totale_collecte, Pts_collecte, labels, ind_masse, Pts_ancrage = get_list_results_dynamic(
        participant, empty_trial_name, trial_name, [frame, frame+1]
    )
    Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)
    initial_guess = InitialGuessType.SURFACE_INTERPOLATION
    _, _, _, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess,
                                                                  Pts_collecte[0, :, :],
                                                                  Pts_ancrage[0, :, :],
                                                                  Pt_repos,
                                                                  Pt_ancrage_repos,
                                                                  dict_fixed_params,
                                                                  trial_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(0, 0, -1.2, "ow")

    ax.plot(
        Pt_ancrage_interpolated[:, 0],
        Pt_ancrage_interpolated[:, 1],
        Pt_ancrage_interpolated[:, 2],
        "ok",
        mfc="none",
        alpha=0.5,
        markersize=4,
        label="Model Frame",
    )
    ax.plot(
        Pts_ancrage[0, :, 0],
        Pts_ancrage[0, :, 1],
        Pts_ancrage[0, :, 2],
        ".k",
        label="Pts_ancrage",
    )

    ax.plot(
        Pt_interpolated[0, :],
        Pt_interpolated[1, :],
        Pt_interpolated[2, :],
        "or",
        mfc="none",
        markersize=4,
        label="Pt_interpolated",
    )
    ax.plot(
        Pts_collecte[0, :, 0],
        Pts_collecte[0, :, 1],
        Pts_collecte[0, :, 2],
        ".r",
        label="Pts_collecte",
    )

    ax.legend()
    plt.show()

    global_optim = global_optimisation(
        Pts_collecte[0, :, :],
        Pt_interpolated,
        Pt_ancrage_interpolated,
        dict_fixed_params,
        labels,
        ind_masse,
        Masse_centre,
        Pt_repos,
        Pt_ancrage_repos,
        None,
        WITH_K_OBLIQUE=False,
    )
    prob = pg.problem(global_optim)
    w_opt, cost = solve(prob, global_optim)

    Ma = np.array(w_opt[0:5])
    K = np.array(w_opt[5:-15])


    data = {"Ma": Ma,
            "K": K,
            }
    with open(f"results/static_1essai.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
