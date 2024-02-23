"""
This file contains useful functions to load and treat data.
"""

import numpy as np
import sys
import pickle

from enums import InitialGuessType

sys.path.append("Dynamique/")
sys.path.append("../Dynamique/")
sys.path.append("../../Dynamique/")
from utils_dynamic import get_list_results_dynamic, Points_ancrage_repos, linear_interpolation_collecte, surface_interpolation_collecte

sys.path.append("Statique/")
from utils_static import Param_fixe

sys.path.append("Dynamique/iterative_stabilisation/")
sys.path.append("../Dynamique/iterative_stabilisation/")
sys.path.append("../../Dynamique/iterative_stabilisation/")
from iterative_stabilisation_algo import position_the_points_based_on_the_force



n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)

# trials = {"centrefront_D": [1, 2, 3,       6, 7, 8, 9, 10, 11],
#               "statique_D":    [1,    3, 4, 5, 6,    8, 9, 10, 11],
#               "leftfront_D":   [1, 2,    4, 5, 6, 7, 8,    10, 11],
#               "left_D":        [   2, 3, 4, 5, 6, 7, 8, 9,     11]}

trials = {"centrefront_D": [1, 2, 6, 7, 10, 11],
              "statique_D": [3, 4, 5, 8, 9, 10],
              "leftfront_D": [1, 2, 4, 5, 6, 7, 8, 10, 11],
              "left_D": [2, 3, 4, 6, 9, 11]}


def Pt_bounds(initial_guess, Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, labels, ind_masse, WITH_K_OBLIQUE):
    """
    Returns the bounds on the position of the points of the model based on the interpolation of the missing points.
    """

    x_slack = [0.3, 0.3]
    y_slack = [0.3, 0.3]
    z_slack = [1, 1]
    if initial_guess == InitialGuessType.LINEAR_INTERPOLATION:
        Pt_interpolated, Pt_ancrage_interpolated = linear_interpolation_collecte(Pt_collecte, Pt_ancrage, labels)
    elif initial_guess == InitialGuessType.RESTING_POSITION:
        Pt_interpolated, Pt_ancrage_interpolated = Pt_repos, Pt_ancrage_repos
    elif initial_guess == InitialGuessType.SURFACE_INTERPOLATION:
        Pt_interpolated, Pt_ancrage_interpolated = surface_interpolation_collecte(
            [Pt_collecte], [Pt_ancrage], Pt_repos, Pt_ancrage_repos, labels, True
        )
        Pt_interpolated = Pt_interpolated[0, :, :].T
        Pt_ancrage_interpolated = Pt_ancrage_interpolated[0, :, :]
    elif initial_guess == InitialGuessType.WARM_START:
        """
        First surface interpolation, then iterative stabilisation based on the results from the Global optimisation.
        """
        dict_fixed_params = Param_fixe()
        Pt_interpolated_surface, Pt_ancrage_interpolated = surface_interpolation_collecte(
            [Pt_collecte], [Pt_ancrage], Pt_repos, Pt_ancrage_repos, dict_fixed_params, PLOT_FLAG=False
        )

        Pt_interpolated_surface = Pt_interpolated_surface[0, :, :].T
        Pt_ancrage_interpolated = Pt_ancrage_interpolated[0, :, :]
        with open(f"../iterative_stabilisation/results/static_multi_essai.pkl", "rb") as f:
            data = pickle.load(f)
            Ma = data["Ma"]
            K = data["K"]

        Pt_interpolated, _ = position_the_points_based_on_the_force(Pt_interpolated_surface, Pt_ancrage_interpolated,
                                                                        dict_fixed_params, Ma, None, K, ind_masse,
                                                                        WITH_K_OBLIQUE, PLOT_FLAG=True)
        Pt_interpolated = Pt_interpolated.T
        x_slack = [0.05, 0.05]
        y_slack = [0.05, 0.05]
        z_slack = [0.05, 0.05]
    else:
        raise RuntimeError(f"The interpolation type of the initial guess {initial_guess} is not recognized.")

    # bounds and initial guess
    lbw_Pt = []
    ubw_Pt = []
    w0_Pt = []

    for k in range(n * m * 3):
        if k % 3 == 0:  # limites et guess en x
            lbw_Pt += [Pt_interpolated[0, int(k // 3)] - x_slack[0]]
            ubw_Pt += [Pt_interpolated[0, int(k // 3)] + x_slack[1]]
            w0_Pt += [Pt_interpolated[0, int(k // 3)]]
        if k % 3 == 1:  # limites et guess en y
            lbw_Pt += [Pt_interpolated[1, int(k // 3)] - y_slack[0]]
            ubw_Pt += [Pt_interpolated[1, int(k // 3)] + y_slack[1]]
            w0_Pt += [Pt_interpolated[1, int(k // 3)]]
        if k % 3 == 2:  # limites et guess en z
            lbw_Pt += [Pt_interpolated[2, int(k // 3)] - z_slack[0]]
            ubw_Pt += [Pt_interpolated[2, int(k // 3)] + z_slack[1]]
            w0_Pt += [Pt_interpolated[2, int(k // 3)]]

    return w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated, Pt_ancrage_interpolated


def load_data(trial_name, nb_disques, initial_guess):

    empty_trial_name = ["labeled_statique_centrefront_vide", "labeled_statique_vide"]

    participants = [0] * len(trial_name)
    frame = []
    participant = []
    trial_names = []
    Masse_centre = []
    inds_masse = []
    essai_vide = []
    Pts_interpolated = np.zeros((1, 3, m * n))
    Pts_ancrage_interpolated = np.zeros((1, 2 * (m + n), 3))
    Pts_collecte = np.zeros((1, 3, m * n))
    Pts_ancrage = np.zeros((1, 2 * (m + n), 3))
    w0_Pt = []
    lbw_Pt = []
    ubw_Pt = []
    for i in range(len(trial_name)):
        trial_names.append(trial_name[i])
        participant.append(participants[i - 1])
        print(trial_names[i])
        if "front" not in trial_names[i]:
            essai_vide += [empty_trial_name[1]]
        else:
            essai_vide += [empty_trial_name[0]]

        if participant[i] != 0:  # si humain choisi
            masses = [64.5, 87.2]
            Masse_centre.append(
                masses[participants[i] - 1]
            )  # on recupere dans la liste au-dessus, attention aux indices ...(-1)
            print("masse appliquée pour le participant " + str(participant[i]) + " = " + str(Masse_centre[i]) + " kg")
            frame += [3000]

        if participant[i] == 0:  # avec des poids
            masses = [0, 27.0, 47.1, 67.3, 87.4, 102.5, 122.6, 142.8, 163.0, 183.1, 203.3, 228.6]
            Masse_centre.append(masses[nb_disques[i]])
            print("masse appliquée pour " + str(nb_disques[i]) + " disques = " + str(Masse_centre[i]) + " kg")
            frame += [700]

        dict_fixed_params = Param_fixe()
        Fs_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage = get_list_results_dynamic(
            participant[i], essai_vide[i], trial_names[i], [frame[i], frame[i] + 1]
        )
        Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)
        current_w0_Pt, current_lbw_Pt, current_ubw_Pt, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess,
                                                                      Pt_collecte[0, :, :],
                                                                      Pt_ancrage[0, :, :],
                                                                      Pt_repos,
                                                                      Pt_ancrage_repos,
                                                                      labels,
                                                                      ind_masse,
                                                                      WITH_K_OBLIQUE=False)

        Pts_interpolated = np.concatenate((Pts_interpolated, Pt_interpolated.reshape(1, 3, m * n)), axis=0)
        Pts_ancrage_interpolated = np.concatenate(
            (Pts_ancrage_interpolated, Pt_ancrage_interpolated.reshape(1, 2 * (m + n), 3)), axis=0)

        inds_masse += [ind_masse]
        Pts_collecte = np.concatenate((Pts_collecte, Pt_collecte), axis=0)
        Pts_ancrage = np.concatenate((Pts_ancrage, Pt_ancrage.reshape(1, 2 * (m + n), 3)), axis=0)
        w0_Pt += [current_w0_Pt]
        lbw_Pt += [current_lbw_Pt]
        ubw_Pt += [current_ubw_Pt]

    Pts_interpolated = Pts_interpolated[1:, :, :]
    Pts_ancrage_interpolated = Pts_ancrage_interpolated[1:, :, :]
    Pts_collecte = Pts_collecte[1:, :, :]
    Pts_ancrage = Pts_ancrage[1:, :, :]

    return Pts_collecte, Pts_interpolated, Pts_ancrage_interpolated, dict_fixed_params, labels, inds_masse, Masse_centre, Pt_repos, Pt_ancrage_repos, trial_name, w0_Pt, lbw_Pt, ubw_Pt


def load_training_pool_data(initial_guess):
    """
    This function allows to load the experimental data from the training pool and computes the bounds and initial guess.
    """

    trial_name = []
    nb_disques = []
    for key in trials.keys():
        for i in range(len(trials[key])):
            trial_name += [f"labeled_{key}{trials[key][i]}"]
            nb_disques += [trials[key][i]]

    return load_data(trial_name, nb_disques, initial_guess)


def load_testing_pool_data(initial_guess):
    """
    This function allows to load the experimental data from the testing pool and computes the bounds and initial guess.
    """

    trial_name = []
    nb_disques = []
    for key in trials.keys():
        for i in range(12):
            if i not in trials[key]:
                trial_name += [f"labeled_{key}{trials[key][i]}"]
                nb_disques += [trials[key][i]]

    return load_data(trial_name, nb_disques, initial_guess)