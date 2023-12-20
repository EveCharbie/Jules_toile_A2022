
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

    initial_guess = InitialGuessType.SURFACE_INTERPOLATION
    frame = []
    Nb_essais_a_optimiser = 36
    essais = []
    empty_trial_name = ["labeled_statique_centrefront_vide", "labeled_statique_vide"]
    participants = [0] * Nb_essais_a_optimiser
    nb_disques = (
        [1, 2, 3, 6, 7, 8, 9, 10, 11]
        + [1, 3, 4, 5, 6, 8, 9, 10, 11]
        + [1, 4, 5, 6, 7, 8, 9, 10, 11]
        + [2, 3, 5, 6, 7, 8, 9, 10, 11]
    )


    for i in range(0, 9):  # 9 essais par zone
        essais += ["labeled_statique_centrefront_D" + str(nb_disques[i])]
    for i in range(9, 18):  # 9 essais par zone
        essais += ["labeled_statique_D" + str(nb_disques[i])]
    for i in range(18, 27):  # 9 essais par zone
        essais += ["labeled_statique_leftfront_D" + str(nb_disques[i])]
    for i in range(27, 36):  # 9 essais par zone
        essais += ["labeled_statique_left_D" + str(nb_disques[i])]

    participant = []
    trial_names = []
    Masse_centre = []
    inds_masse = []
    essai_vide = []
    Pts_interpolated = np.zeros((1, 3, m*n))
    Pts_ancrage_interpolated = np.zeros((1, 2*(m+n), 3))
    Pts_collecte = np.zeros((1, 3, m*n))
    Pts_ancrage = np.zeros((1, 2*(m+n), 3))

    for i in range(len(essais)):
        trial_names.append(essais[i])
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
            participant[i], essai_vide[i], trial_names[i], [frame[i], frame[i]+1]
        )
        Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)
        _, _, _, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess,
                                                                      Pt_collecte[0, :, :],
                                                                      Pt_ancrage[0, :, :],
                                                                      Pt_repos,
                                                                      Pt_ancrage_repos,
                                                                      dict_fixed_params,
                                                                      trial_names[i])
        Pts_interpolated = np.concatenate((Pts_interpolated, Pt_interpolated.reshape(1, 3, m*n)), axis=0)
        Pts_ancrage_interpolated = np.concatenate((Pts_ancrage_interpolated, Pt_ancrage_interpolated.reshape(1, 2*(m+n), 3)), axis=0)

        inds_masse += [ind_masse]
        Pts_collecte = np.concatenate((Pts_collecte, Pt_collecte), axis=0)
        Pts_ancrage = np.concatenate((Pts_ancrage, Pt_ancrage.reshape(1, 2*(m+n), 3)), axis=0)

    Pts_interpolated = Pts_interpolated[1:, :, :]
    Pts_ancrage_interpolated = Pts_ancrage_interpolated[1:, :, :]
    Pts_collecte = Pts_collecte[1:, :, :]
    Pts_ancrage = Pts_ancrage[1:, :, :]

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
        WITH_K_OBLIQUE=False,
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
    for i in range(1, len(essais)):
        Ma = np.hstack((Ma, w_opt[offset:offset+5]))
        offset += 5

    data = {"Ma": Ma,
            "K": K,
            "w_opt": w_opt,
            "cost": cost,
            "essais": essais,
            }
    with open(f"results/static_multi_essai.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
