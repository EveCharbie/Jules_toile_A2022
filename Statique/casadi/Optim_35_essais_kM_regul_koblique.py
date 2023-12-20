"""
Optimization of the elasticity coefficents (K) and of the repartition of the applied mass (M) on the 5 points of the trampoline bed.

First, 35 c3d files containing experimental positions are loaded (these 35 trials compose the learning pool; 80% of the trials)
Then the K are optimized to match all the trials at once.

Optimization variables:
    - K (elasticity coefficients)
    - M (applied mass on the 5 points of the trampoline bed)
    - X (position of the n*m points of the trampoline bed)

Objectives:
    - Minimize the forces on the n*m points of the trampoline bed
    - Minimize the difference between the position of the points and the model
    - Regularization of the diagonal springs (redondant model)

Constraints:
    - The 5 masses must equal the total mass applied on the trampoline using mass plates

The evaluation of the k on the test pool is done in the file: Statique/Verif_optim_position_k_fixe.py
"""

import casadi as cas
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from ezc3d import c3d
from mpl_toolkits import mplot3d
import time
from scipy.interpolate import interp1d
import pickle

import sys

sys.path.append("../")
from enums import InitialGuessType
from utils_static import Param_fixe, list2tab, tab2list, Spring_bouts, Spring_bouts_croix, Calcul_Pt_F

sys.path.append("../../Dynamique/")
from utils_dynamic import surface_interpolation_collecte, Resultat_PF_collecte, linear_interpolation_collecte, Points_ancrage_repos


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


#####################################################################################################################

# def cost_function(X, K, Ma, Pt_collecte, Pt_ancrage, Pt_interpolated, dict_fixed_params, labels, ind_masse, optimize_static_mass):
#
#     if type(X) == cas.MX:
#         zero_fcn = cas.MX.zeros
#     elif type(X) == cas.DM:
#         zero_fcn = cas.DM.zeros
#     elif type(X) == np.ndarray:
#         zero_fcn = np.zeros
#
#     F_totale, F_point = Calcul_Pt_F(X, Pt_ancrage, dict_fixed_params, K, ind_masse, Ma)
#     Pt = list2tab(X)
#
#     Difference = zero_fcn((1, 1))
#     for i in range(3):
#         for ind in range(n * m):
#
#             if "t" + str(ind) in labels:
#                 ind_collecte = labels.index("t" + str(ind))  # ATTENTION gérer les nans
#                 if np.isnan(Pt_collecte[i, ind_collecte]):  # gérer les nans
#                     Difference += (
#                         0.01 * (Pt[ind, i] - Pt_interpolated[i, ind]) ** 2
#                     )  # on donne un poids moins important aux données interpolées
#                 elif ind in [ind_masse, ind_masse-1, ind_masse+1, ind_masse-15, ind_masse+15]:
#                     if optimize_static_mass:
#                         Difference += 500 * (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2
#                     else:
#                         Difference += (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2
#                 else:
#                     Difference += (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2
#             else:
#                 Difference += 0.01 * (Pt[ind, i] - Pt_interpolated[i, ind]) ** 2
#
#             # if not optimize_static_mass:
#             #     if ind not in [ind_masse, ind_masse-1, ind_masse+1, ind_masse-15, ind_masse+15]:
#             #         Difference += (F_point[ind, i]) ** 2
#             # else:
#                 Difference += (F_point[ind, i]) ** 2
#
#     if type(K) == cas.MX:
#         regul_k = K[8] ** 2 + K[9] ** 2 + K[10] ** 2 + K[11] ** 2
#         output = (1e4) * Difference + (1e-6) * regul_k
#         obj = cas.Function("f", [X, K, Ma], [output]).expand()
#     else:
#         obj = cas.Function("f", [X, Ma], [1e-6 * Difference]).expand()
#
#     return obj


def cost_function(X, K, Ma, Pt_collecte, Pt_ancrage, dict_fixed_params, ind_masse):

    if type(X) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(X) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(X) == np.ndarray:
        zero_fcn = np.zeros

    F_total, _ = Calcul_Pt_F(X, Pt_ancrage, dict_fixed_params, K, ind_masse, Ma, WITH_K_OBLIQUE=False)
    Pt = list2tab(X)

    Difference = zero_fcn((1, 1))
    for i_component in range(3):
        for i_spring in range(n * m):
            if not np.isnan(Pt_collecte[i_component, i_spring]):
                Difference += 500 * (Pt[i_spring, i_component] - Pt_collecte[i_component, i_spring]) ** 2

    Difference += cas.norm_fro(F_total) / 100000

    return Difference


def longueur_ressort(dict_fixed_params, Pt, Pt_ancrage):

    if type(Pt) == cas.MX:
        zero_fcn = cas.MX.zeros
        norm_fcn = cas.norm_fro
    elif type(Pt) == cas.DM:
        zero_fcn = cas.DM.zeros
        norm_fcn = cas.norm_fro
    elif type(Pt) == np.ndarray:
        zero_fcn = np.zeros
        norm_fcn = np.linalg.norm

    Pt = list2tab(Pt)
    l_repos = dict_fixed_params["l_repos"]
    l_repos_croix = dict_fixed_params["l_repos_croix"]

    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)

    delta = zero_fcn((Nb_ressorts_croix + Nb_ressorts, 1))
    for i in range(Nb_ressorts):
        delta[i] = norm_fcn(Spring_bout_2[i, :] - Spring_bout_1[i, :]) - l_repos[i]
    for i in range(Nb_ressorts, Nb_ressorts_croix):
        delta[i] = norm_fcn(Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :]) - l_repos_croix[i]

    return delta


def get_list_results_static(participant, trial_name, frame, dict_fixed_params):
    F_totale_collecte = []
    Pt_collecte = []
    ind_masse = []
    labels = []
    Pt_ancrage = []

    for i in range(len(trial_name)):
        empty_trial_name = "labeled_statique_centrefront_vide"
        if "front" not in trial_name[i]:
            empty_trial_name = "labeled_statique_vide"

        Resultat_PF_collecte_total = Resultat_PF_collecte(participant[i], empty_trial_name, trial_name[i], frame)
        F_totale_collecte.append(Resultat_PF_collecte_total[0])
        Pt_collecte.append(Resultat_PF_collecte_total[1])
        labels.append(Resultat_PF_collecte_total[2])
        ind_masse.append(Resultat_PF_collecte_total[3])
        Pt_ancrage.append(Points_ancrage_repos(dict_fixed_params)[0])

    return F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage

    def k_bounds():
        k1 = 3381.540529105023  # un type de coin (ressort horizontal)
        k2 = 4094.5093125505978  # ressorts horizontaux du bord (bord vertical) : relient le cadre et la toile
        k3 = 682.2755546467131  # un type de coin (ressort vertical)
        k4 = 2686.7591230959133  # ressorts verticaux du bord (bord horizontal) : relient le cadre et la toile
        k5 = 16716.81679597304  # ressorts horizontaux du bord horizontal de la toile
        k6 = 67742.77745786862  # ressorts horizontaux
        k7 = 11848.276828389258  # ressorts verticaux du bord vertical de la toile
        k8 = 42804.97154361123  # ressorts verticaux

        k_oblique1 = 1040.5626570596532  # 4 ressorts des coins
        k_oblique2 = 1502.9109243815935  # ressorts des bords verticaux
        k_oblique3 = 3814.722559052384  # ressorts des bords horizontaux
        k_oblique4 = 18726.727599761303  # ressorts obliques quelconques

        w0_k = [k1, k2, k3, k4, k5, k6, k7, k8, k_oblique1, k_oblique2, k_oblique3, k_oblique4]
        for i in range(len(w0_k)):
            w0_k[i] = 1 * w0_k[i]

        lbw_k = [1e-4] * 12
        ubw_k = [1e7] * 12  # bornes très larges

        return w0_k, lbw_k, ubw_k

    def m_bounds(masse_essai):
        lbw_m, ubw_m = [], []

        M1 = masse_essai / 5  # masse centre
        M2 = masse_essai / 5  # masse centre +1
        M3 = masse_essai / 5  # masse centre -1
        M4 = masse_essai / 5  # masse centre +15
        M5 = masse_essai / 5  # masse centre -15

        w0_m = [M1, M2, M3, M4, M5]
        lbw_m += [0.5 * masse_essai / 5] * 5
        ubw_m += [1.5 * masse_essai / 5] * 5

        return w0_m, lbw_m, ubw_m

    def Pt_bounds(initial_guess, Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, labels):
        """
        Returns the bounds on the position of the points of the model based on the interpolation of the missing points.
        Do not use this, instead see Optim_multi_essais_kM_regul_koblique.py
        """

        if isinstance(initial_guess, InitialGuessType.LINEAR_INTERPOLATION):
            Pt_interpolated = linear_interpolation_collecte(Pt_collecte, Pt_ancrage, labels)
        elif isinstance(initial_guess, InitialGuessType.RESTING_POSITION):
            Pt_interpolated = Pt_repos
        elif isinstance(initial_guess, InitialGuessType.SURFACE_INTERPOLATION):
            Pt_interpolated = surface_interpolation_collecte(
                Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, labels
            )
        else:
            raise RuntimeError(f"The interpolation type of the initial guess {initial_guess} is not recognized.")

        # bounds and initial guess
        lbw_Pt = []
        ubw_Pt = []
        w0_Pt = []

        for k in range(n * m * 3):
            if k % 3 == 0:  # limites et guess en x
                lbw_Pt += [Pt_interpolated[0, int(k // 3)] - 0.3]
                ubw_Pt += [Pt_interpolated[0, int(k // 3)] + 0.3]
                w0_Pt += [Pt_interpolated[0, int(k // 3)]]
            if k % 3 == 1:  # limites et guess en y
                lbw_Pt += [Pt_interpolated[1, int(k // 3)] - 0.3]
                ubw_Pt += [Pt_interpolated[1, int(k // 3)] + 0.3]
                w0_Pt += [Pt_interpolated[1, int(k // 3)]]
            if k % 3 == 2:  # limites et guess en z
                lbw_Pt += [-2]
                ubw_Pt += [0.5]
                w0_Pt += [Pt_interpolated[2, int(k // 3)]]

        return w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated


def Optimisation(
    F_totale_collecte,
    Pt_collecte,
    labels,
    ind_masse,
    Pt_ancrage,
    Masse_centre,
    trial_name,
    initial_guess,
    optimize_static_mass,
    dict_fixed_paramss,
):
    # PARAM FIXES
    n = 15
    m = 9
    Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_paramss)

    # OPTIMISATION :
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    # K
    K = cas.MX.sym("K", 12)
    w0_k, lbw_k, ubw_k = k_bounds()
    w0 += w0_k
    lbw += lbw_k
    ubw += ubw_k
    w += [K]

    obj = 0
    for i in range(len(trial_name)):
        masse_essai = Masse_centre[i]

        # NLP VALUES
        Ma = cas.MX.sym("Ma", 5)
        X = cas.MX.sym("X", 135 * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...) puis Fxyz

        # Ma
        w0_m, lbw_m, ubw_m = m_bounds(masse_essai)
        w0 += w0_m
        lbw += lbw_m
        ubw += ubw_m
        w += [Ma]

        # X
        w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated = Pt_bounds(
            initial_guess, Pt_collecte[i], Pt_ancrage, Pt_repos, Pt_ancrage_repos, labels[i]
        )
        lbw += lbw_Pt
        ubw += ubw_Pt
        w0 += w0_Pt
        w += [X]

        # fonction contrainte :
        g += [Ma[0] + Ma[1] + Ma[2] + Ma[3] + Ma[4] - Masse_centre[i]]
        lbg += [0]
        ubg += [0]

        # en statique on ne fait pas de boucle sur le temps :
        J = cost_function(
            X,
            K,
            Ma,
            Pt_collecte[i],
            Pt_ancrage[i],
            Pt_interpolated,
            dict_fixed_params,
            labels[i],
            ind_masse[i],
            optimize_static_mass,
        )
        obj += J(X, K, Ma)

    # Create an NLP solver
    prob = {"f": obj, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    opts = {"ipopt": {"max_iter": 100000, "linear_solver": "ma57"}}
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(
        x0=cas.vertcat(*w0), lbx=cas.vertcat(*lbw), ubx=cas.vertcat(*ubw), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg)
    )
    w_opt = sol["x"].full().flatten()

    return w_opt, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, sol.get("f")


##########################################################################################################################
def main():
    initial_guess = InitialGuessType.RESTING_POSITION  ### to be tested with SURFACE_INTERPOLATION
    optimize_static_mass = True

    # Liste des deux essais a optimiser
    essais = []
    participants = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    nb_disques = [
        2,
        3,
        4,
        5,
        6,
        8,
        8,
        9,
        10,
        11,
        11,
        6,
        9,
        2,
        7,
        8,
        3,
        10,
        1,
        2,
        7,
        6,
        5,
        10,
        1,
        7,
        6,
        3,
        8,
        1,
        9,
        7,
        11,
        4,
        10,
    ]  # choix des 35 masses
    # premier essai
    frame = 700

    # juste faire attention aux doublons avec la liste des disques au dessus
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[0])]
    essais += ["labeled_statique_D" + str(nb_disques[1])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[2])]
    essais += ["labeled_statique_left_D" + str(nb_disques[3])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[4])]
    essais += ["labeled_statique_D" + str(nb_disques[5])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[6])]
    essais += ["labeled_statique_left_D" + str(nb_disques[7])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[8])]
    essais += ["labeled_statique_D" + str(nb_disques[9])]

    essais += ["labeled_statique_leftfront_D" + str(nb_disques[10])]
    essais += ["labeled_statique_left_D" + str(nb_disques[11])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[12])]
    essais += ["labeled_statique_D" + str(nb_disques[13])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[14])]
    essais += ["labeled_statique_left_D" + str(nb_disques[15])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[16])]
    essais += ["labeled_statique_D" + str(nb_disques[17])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[18])]
    essais += ["labeled_statique_left_D" + str(nb_disques[19])]

    essais += ["labeled_statique_centrefront_D" + str(nb_disques[20])]
    essais += ["labeled_statique_D" + str(nb_disques[21])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[22])]
    essais += ["labeled_statique_left_D" + str(nb_disques[23])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[24])]
    essais += ["labeled_statique_D" + str(nb_disques[25])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[26])]
    essais += ["labeled_statique_left_D" + str(nb_disques[27])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[28])]
    essais += ["labeled_statique_D" + str(nb_disques[29])]

    essais += ["labeled_statique_leftfront_D" + str(nb_disques[30])]
    essais += ["labeled_statique_left_D" + str(nb_disques[31])]
    essais += ["labeled_statique_centrefront_D" + str(nb_disques[32])]
    essais += ["labeled_statique_D" + str(nb_disques[33])]
    essais += ["labeled_statique_leftfront_D" + str(nb_disques[34])]

    participant = []  # creation d une liste pour gerer les participants
    trial_name = []  # creation d une liste pour gerer les essais
    empty_trial_name = [
        "labeled_statique_centrefront_vide",
        "labeled_statique_vide",
    ]  # creation d une liste pour les essais a vide
    Masse_centre = []
    for i in range(len(essais)):  # ici 2 essais seulement
        trial_name.append(essais[i])
        participant.append(participants[i - 1])
        essai_vide = empty_trial_name[0]
        print(trial_name[i])
        if "front" not in trial_name[i]:
            essai_vide = empty_trial_name[1]
        print(essai_vide)

        if participant[i] != 0:  # si humain choisi
            masses = [64.5, 87.2]
            Masse_centre.append(
                masses[participants[i] - 1]
            )  # on recupere dans la liste au-dessus, attention aux indices (-1)
            print("masse appliquée pour le participant " + str(participant[i]) + " = " + str(Masse_centre[i]) + " kg")

        if participant[i] == 0:  # avec des poids
            masses = [0, 27.0, 47.1, 67.3, 87.4, 102.5, 122.6, 142.8, 163.0, 183.1, 203.3, 228.6]
            Masse_centre.append(masses[nb_disques[i]])
            print("masse appliquée pour " + str(nb_disques[i]) + " disques = " + str(Masse_centre[i]) + " kg")

    dict_fixed_params = Param_fixe()
    F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage = get_list_results_static(
        participant, trial_name, frame, dict_fixed_params
    )

    ########################################################################################################################

    start_main = time.time()

    Solution, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, f = Optimisation(
        F_totale_collecte,
        Pt_collecte,
        labels,
        ind_masse,
        Pt_ancrage,
        Masse_centre,
        trial_name,
        initial_guess,
        optimize_static_mass,
        dict_fixed_params,
    )

    # recuperation et affichage
    k = np.array(Solution[:12])
    M = []
    Pt = []
    F_totale = []
    F_point = []

    for i in range(len(essais)):
        M.append(np.array(Solution[12 + n * m * 3 * i + 5 * i : 17 + 405 * i + 5 * i]))
        Pt.append(np.reshape(Solution[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i], (135, 3)))

        F_totale.append(
            Calcul_Pt_F(
                Solution[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                k,
                ind_masse[i],
                Solution[12 + 405 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[0]
        )
        F_point.append(
            Calcul_Pt_F(
                Solution[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                k,
                ind_masse[i],
                Solution[12 + 405 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[1]
        )

        F_totale[i] = cas.evalf(F_totale[i])
        F_point[i] = cas.evalf(F_point[i])
        F_point[i] = np.array(F_point[i])

        Pt_collecte[i] = np.array(
            Pt_collecte[i]
        )  # permet de mettre pt collecte sous la bonne forme pour l'utiliser apres
        Pt_ancrage[i] = np.array(Pt_ancrage[i])  # permet de mettre pt ancrage sous la bonne forme pour l'utiliser apres

    end_main = time.time()
    temps_min = (end_main - start_main) / 60

    print("**************************************************************************")
    print("Temps total : " + str(temps_min) + " min")
    print("**************************************************************************")

    ############################################################################################################
    # Comparaison entre collecte et points optimisés :
    fig = plt.figure()
    for i in range(len(essais)):
        ax = plt.subplot(5, 7, i + 1, projection="3d")
        ax.set_box_aspect([1.1, 1.8, 1])
        ax.plot(Pt[i][:, 0], Pt[i][:, 1], Pt[i][:, 2], "+r", label="Points de la toile optimisés")
        ax.plot(Pt_ancrage[i][:, 0], Pt_ancrage[i][:, 1], Pt_ancrage[i][:, 2], ".k", label="Points d'ancrage simulés")
        ax.plot(
            Pt[i][ind_masse[i], 0],
            Pt[i][ind_masse[i], 1],
            Pt[i][ind_masse[i], 2],
            "+y",
            label="Point optimisés le plus bas d'indice " + str(ind_masse[0]),
        )
        ax.plot(Pt_collecte[i][0, :], Pt_collecte[i][1, :], Pt_collecte[i][2, :], ".b", label="Points collecte")
        label_masse = labels[i].index("t" + str(ind_masse[i]))
        ax.plot(
            Pt_collecte[i][0, label_masse],
            Pt_collecte[i][1, label_masse],
            Pt_collecte[i][2, label_masse],
            "og",
            label="Point collecte le plus bas " + labels[i][label_masse],
        )
        plt.title("Fusion optim " + str(trial_name[i]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    plt.legend()

    # calcul de l'erreur :
    # sur la position/force
    erreur_position = []
    erreur_force = []
    for p in range(len(essais)):
        err_pos = 0
        err_force = 0
        for ind in range(2 * n * m):
            if "t" + str(ind) in labels[p]:
                ind_collecte1 = labels[p].index("t" + str(ind))  # ATTENTION gérer les nans
                for i in range(3):
                    if np.isnan(Pt_collecte[p][i, ind_collecte1]) == False:  # gérer les nans
                        err_pos += (Pt[p][ind, i] - Pt_collecte[p][i, ind_collecte1]) ** 2
        erreur_position.append(err_pos)

        for ind in range(n * m):
            for i in range(3):
                err_force += (F_point[p][ind, i]) ** 2
        erreur_force.append(err_force)

        print(
            "-Erreur sur la position-  " + str(trial_name[p]) + " = " + str(erreur_position[p]) + " m" + " // "
            "-Erreur sur la force-  " + str(trial_name[p]) + " = " + str(erreur_force[p]) + " N"
        )

    # ENREGISTREMENT PICKLE#
    path = "results/result_multi_essais/" + "optim_sur_35_essais_corr" + ".pkl"
    with open(path, "wb") as file:
        pickle.dump(Solution, file)
        pickle.dump(labels, file)
        pickle.dump(Pt_collecte, file)
        pickle.dump(Pt_ancrage, file)
        pickle.dump(ind_masse, file)
        pickle.dump(erreur_position, file)
        pickle.dump(erreur_force, file)
        pickle.dump(f, file)
        pickle.dump(dict_fixed_params, file)
        pickle.dump(trial_name, file)

    plt.show()  # on affiche tous les graphes


if __name__ == "__main__":
    main()
