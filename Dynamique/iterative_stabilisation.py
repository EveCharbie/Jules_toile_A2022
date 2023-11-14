"""

"""

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import sys

sys.path.append("../")
from enums import InitialGuessType

sys.path.append("../Statique/")
from Optim_35_essais_kM_regul_koblique import Param_fixe, Calcul_Pt_F, list2tab, Spring_bouts, Spring_bouts_croix, tab2list
from modele_dynamique_nxm_DimensionsReelles import (
    Resultat_PF_collecte,
    Point_ancrage,
    Point_toile_init,
    Points_ancrage_repos,
    surface_interpolation_collecte,
    spring_bouts_collecte,
    static_forces_calc,
    static_force_in_each_point,
    multiple_shooting_integration,
)
from Optim_multi_essais_kM_regul_koblique import m_bounds
from Verif_optim_position_k_fixe import Param_variable
from optim_dynamique_withoutC_casadi import get_list_results_dynamic, Pt_bounds, F_bounds


def position_the_points_based_on_the_force(Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params, weight, ind_masse, PLOT_FLAG=False):

    Ma = np.array([weight/5, weight/5, weight/5, weight/5, weight/5])

    max_iter = 1000
    epsilon = 1e-3
    displacement = np.inf
    iteration = 0
    Pts = np.zeros((n*m, 3))
    Pts[:, :] = Pt_interpolated[:, :].T
    Pts_before = np.zeros((n*m, 3))
    Pts_after = np.zeros((n*m, 3))
    while displacement > epsilon and iteration < max_iter:

        Pts_before[:, :] = Pts[:, :]
        K, _, _ = Param_variable(Ma, ind_masse)
        X = tab2list(Pts)
        _, F_point = Calcul_Pt_F(X, Pt_ancrage_interpolated, dict_fixed_params, K, ind_masse, Ma)
        # Spring_bout_1, Spring_bout_2 = Spring_bouts(Pts, Pt_ancrage_interpolated)
        # Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pts)
        # spring_elongation = np.linalg.norm(Spring_bout_2 - Spring_bout_1) - dict_fixed_params["l_repos"]
        # spring_elongation_croix = np.linalg.norm(Spring_bout_croix_2 - Spring_bout_croix_1) - dict_fixed_params[
        #     "l_repos_croix"]
        Pts += 0.00001 * F_point
        Pts_after[:, :] = Pts[:, :]

        displacement = np.linalg.norm(Pts_after - Pts_before, axis=1).sum()
        iteration += 1

        print(f"{iteration} : {displacement}")

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
            markersize=3,
            label="Model Frame",
        )

        ax.plot(
            Pts_before[:, 0],
            Pts_before[:, 1],
            Pts_before[:, 2],
            "or",
            mfc="none",
            markersize=3,
            label="Pts_before",
        )
        ax.plot(
            Pts_after[:, 0],
            Pts_after[:, 1],
            Pts_after[:, 2],
            "ob",
            mfc="none",
            markersize=3,
            label="Pts_after",
        )
        for i in range(m*n):
            plt.plot(np.vstack((Pts_before[i, 0], Pts_after[i, 0])),
                     np.vstack((Pts_before[i, 1], Pts_after[i, 1])),
                     np.vstack((Pts_before[i, 2], Pts_after[i, 2])),
                     "-k")
        ax.legend()
        plt.show()

    return Pts


##########################################################################################################################
def main():

    # SELECTION OF THE RESULTS FROM THE DATA COLLECTION
    participant = 1
    # participant_1: 64.5 kg
    # participant_2: 87.2 kg
    weight = 64.5
    static_trial_name = "labeled_statique_leftfront_D7"
    trial_name = "labeled_p1_sauthaut_01"
    empty_trial_name = "labeled_statique_centrefront_vide"
    jump_frame_index_interval = [
        7101,
        7120,
        # 7170,
    ]  # This range repends on the trial. To find it, one should use the code plateforme_verification_toutesversions.py.
    dt = 1 / 500  # Hz

    # if trial_name is not a folder, create it
    if not os.path.isdir(f"results_multiple_static_optim_in_a_row/{trial_name}"):
        os.mkdir(f"results_multiple_static_optim_in_a_row/{trial_name}")


    dict_fixed_params = Param_fixe()
    Fs_totale_collecte, Pts_collecte, labels, ind_masse, Pts_ancrage = get_list_results_dynamic(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )

    ########################################################################################################################

    for idx, frame in enumerate(list(range(jump_frame_index_interval[0], jump_frame_index_interval[1]))):

        Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)
        initial_guess = InitialGuessType.SURFACE_INTERPOLATION
        _, _, _, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess,
                                                                      Pts_collecte[idx, :, :],
                                                                      Pts_ancrage[idx, :, :],
                                                                      Pt_repos,
                                                                      Pt_ancrage_repos,
                                                                      dict_fixed_params,
                                                                      trial_name)

        Pts = position_the_points_based_on_the_force(Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params, weight, ind_masse)

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
            markersize=3,
            label="Model Frame",
        )

        ax.plot(
            Pts_ancrage[idx, :, 0],
            Pts_ancrage[idx, :, 1],
            Pts_ancrage[idx, :, 2],
            ".k",
            markersize=3,
            label="Experimental Frame",
        )
        ax.plot(
            Pts_collecte[idx, 0, :],
            Pts_collecte[idx, 1, :],
            Pts_collecte[idx, 2, :],
            ".b",
            markersize=3,
            label="Experimental Trampoline"
        )
        ax.plot(
            Pts_collecte[idx, 0, ind_masse],
            Pts_collecte[idx, 1, ind_masse],
            Pts_collecte[idx, 2, ind_masse],
            ".g",
            markersize=3,
            label="Lowest point on the interval",
        )
        ax.plot(
            Pts[:, 0],
            Pts[:, 1],
            Pts[:, 2],
            "ob",
            mfc="none",
            markersize=3,
            label="Optimized point positions",
        )

        ax.legend()
        # plt.savefig(f"results_multiple_static_optim_in_a_row/{trial_name}/solution_frame{frame}_gaco.png")
        plt.show()

    l_repos = dict_fixed_params["l_repos"]
    l_repos_croix = dict_fixed_params["l_repos_croix"]
    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage_interpolated)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    spring_elongation = np.linalg.norm(Spring_bout_2 - Spring_bout_1, axis=1) - l_repos
    spring_croix_elongation = np.linalg.norm(Spring_bout_croix_2 - Spring_bout_croix_1, axis=1) - l_repos_croix
    print(np.sort(spring_elongation))
    print(np.sort(spring_croix_elongation))

    Pt_integres, erreur_relative, erreur_absolue, static_force_in_each_points, v_all = multiple_shooting_integration(
        1, Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params
    )
    print("erreur relative : ", erreur_relative)
    print("erreur absolue : ", erreur_absolue)


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
        markersize=3,
        label="Model Frame",
    )

    ax.plot(
        Pts_collecte[idx+1, 0, :],
        Pts_collecte[idx+1, 1, :],
        Pts_collecte[idx+1, 2, :],
        ".b",
        markersize=3,
        label="Experimental Trampoline frame + 1"
    )

    ax.plot(
        Pt_integres[:, 0],
        Pt_integres[:, 1],
        Pt_integres[:, 2],
        "ob",
        mfc="none",
        markersize=3,
        label="Integrated point positions",
    )

    ax.legend()
    plt.savefig(f"results_multiple_static_optim_in_a_row/{trial_name}/integration_frame{frame}_gaco.png")
    plt.show()



if __name__ == "__main__":
    n = 15
    m = 9
    main()
