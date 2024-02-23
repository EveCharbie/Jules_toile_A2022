
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
# from Optim_35_essais_kM_regul_koblique import Param_fixe, list2tab, Spring_bouts, Spring_bouts_croix, tab2list
# from modele_dynamique_nxm_DimensionsReelles import (
#     Points_ancrage_repos,
#     multiple_shooting_integration,
# )
# from Optim_multi_essais_kM_regul_koblique import m_bounds, k_bounds, load_training_pool_data
# from optim_dynamique_withoutC_casadi import get_list_results_dynamic, Pt_bounds, F_bounds
# from iterative_stabilisation import position_the_points_based_on_the_force
# from pygmo_optim_with_iterative_stabilisation import global_optimisation, solve

sys.path.append("../../")
from utils_data import load_training_pool_data



n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


##########################################################################################################################

def main():

    initial_guess = InitialGuessType.WARM_START
    WITH_K_OBLIQUE = False
    PLOT_FLAG = True

    Pts_collecte, Pts_interpolated, Pts_ancrage_interpolated, dict_fixed_params, labels, inds_masse, Masse_centre, Pt_repos, Pt_ancrage_repos, trial_names = load_training_pool_data(initial_guess)

    mean_error_per_trial = []
    for i_trial in range(len(trial_names)):
        Pt_interpolated = Pts_interpolated[i_trial]
        Pt_ancrage_interpolated = Pts_ancrage_interpolated[i_trial]
        Pt_collecte = Pts_collecte[i_trial]

        mean_error_per_trial += [np.nanmean(np.linalg.norm(Pt_collecte - Pt_interpolated, axis=0), axis=1)]


        if PLOT_FLAG:
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
                Pt_collecte[0, :],
                Pt_collecte[1, :],
                Pt_collecte[2, :],
                ".b",
                markersize=3,
                label="Experimental Trampoline frame + 1"
            )

            ax.legend()
            plt.savefig(f"results/test_pool_output_{trial_names[i_trial]}.png")
            plt.show()


if __name__ == "__main__":
    main()
