"""
Optimisation des k en single shooting, et des masses du centre pour voir la répartition de la masse du disque sur les 5 points concernés

On load 36 fichier c3d pour avoir les informations de 80% des essais collectés, puis a partir de ca on définit tous les parametres fixes du trampoline.
On considere que les parametres variables sont ;
-les raideurs des ressorts
-les masses des 5 points sur lesquels le disque est posé
L'objectif :
-minimiser la résultante des forces en chacun des n*m points des 35 essais
-minimiser la différence entre la position des points de collecte et la position du modele des 35 essais
-reguler les k obliques

Les contraintes :
-la somme des 5 masses appliquée au niveau des points en contact avec le disque est égale a la masse du disque (pour chacun des essais)

L'optimisation renvoie alors :
-la valeur des k et des 5 masses
-les coordonnées des n*m points des 35 essais
-le label du point sur lequel le disque est posé

"""

import casadi as cas

# from IPython import embed
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

from Optim_35_essais_kM_regul_koblique import interpolation_collecte, a_minimiser, Param_fixe

sys.path.append("../Dynamique/")
from modele_dynamique_nxm_DimensionsReelles import surface_interpolation_collecte, Points_ancrage_repos


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


def k_bounds():
    k1 = 71141.43138667523  # un type de coin (ressort horizontal)
    k2 = 49736.39530405858  # ressorts horizontaux du bord (bord vertical) : relient le cadre et la toile
    k3 = 32719.304620783536  # un type de coin (ressort vertical)
    k4 = 55555.8880837324  # ressorts verticaux du bord (bord horizontal) : relient le cadre et la toile
    k5 = 206089.58358212537  # ressorts horizontaux du bord horizontal de la toile
    k6 = 172374.60990475505  # ressorts horizontaux
    k7 = 130616.02962104743  # ressorts verticaux du bord vertical de la toile
    k8 = 262394.061698019  # ressorts verticaux
    # VALEURS INVENTÉES :
    k_oblique1 = 171417.87643722596  # 4 ressorts des coins
    k_oblique2 = 143529.8253725852  # ressorts des bords verticaux
    k_oblique3 = 200282.72442280647  # ressorts des bords horizontaux
    k_oblique4 = 395528.32183421426  # ressorts obliques quelconques
    k_croix = 3000  # je sais pas

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
    lbw_m += [0.6 * masse_essai / 5] * 5  # diff here
    ubw_m += [1.4 * masse_essai / 5] * 5  # diff here

    return w0_m, lbw_m, ubw_m


def Pt_bounds(initial_guess, Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, labels):
    """
    Returns the bounds on the position of the points of the model based on the interpolation of the missing points.
    """

    if initial_guess == InitialGuessType.LINEAR_INTERPOLATION:
        Pt_interpolated, Pt_ancrage_interpolated = interpolation_collecte(Pt_collecte, Pt_ancrage, labels)
    elif initial_guess == InitialGuessType.RESTING_POSITION:
        Pt_interpolated, Pt_ancrage_interpolated = Pt_repos, Pt_ancrage_repos
    elif initial_guess == InitialGuessType.SURFACE_INTERPOLATION:
        Pt_interpolated, Pt_ancrage_interpolated = surface_interpolation_collecte(
            [Pt_collecte], [Pt_ancrage], Pt_repos, Pt_ancrage_repos, labels, True
        )
        Pt_interpolated = Pt_interpolated[0,:,:].T
        Pt_ancrage_interpolated = Pt_ancrage_interpolated[0, :, :]
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

    return w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated, Pt_ancrage_interpolated


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
    dict_fixed_params,
):
    # PARAM FIXES
    n = 15
    m = 9
    Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)

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
    for i in range(len(essais)):
        masse_essai = Masse_centre[i]

        # NLP VALUES
        Ma = cas.MX.sym("Ma", 5)
        X = cas.MX.sym("X", n * m * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...) puis Fxyz

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
        J = a_minimiser(
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

    frame = 700
    Nb_essais_a_optimiser = 36
    essais = []
    vide_name = ["labeled_statique_centrefront_vide", "labeled_statique_vide"]
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

    participant = []  # creation d une liste pour gerer les participants
    trial_name = []  # creation d une liste pour gerer les essais
    Masse_centre = []  # creation d une liste pour gerer les masses

    for i in range(len(essais)):
        trial_name.append(essais[i])
        participant.append(participants[i - 1])
        essai_vide = vide_name[0]
        print(trial_name[i])
        if "front" not in trial_name[i]:
            essai_vide = vide_name[1]
        print("essai a vide : " + str(essai_vide))

        if participant[i] != 0:  # si humain choisi
            masses = [64.5, 87.2]
            Masse_centre.append(
                masses[participants[i] - 1]
            )  # on recupere dans la liste au-dessus, attention aux indices ...(-1)
            print("masse appliquée pour le participant " + str(participant[i]) + " = " + str(Masse_centre[i]) + " kg")
            frame = 3000

        if participant[i] == 0:  # avec des poids
            masses = [0, 27.0, 47.1, 67.3, 87.4, 102.5, 122.6, 142.8, 163.0, 183.1, 203.3, 228.6]
            Masse_centre.append(masses[nb_disques[i]])
            print("masse appliquée pour " + str(nb_disques[i]) + " disques = " + str(Masse_centre[i]) + " kg")
            frame = 700
    print(vide_name)

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

    ####recuperation et affichage####
    M_centrefront = []
    Pt_centrefront = []
    F_totale_centrefront = []
    F_point_centrefront = []
    Pt_collecte_centrefront = []
    Pt_ancrage_centrefront = []

    M_statique = []
    Pt_statique = []
    F_totale_statique = []
    F_point_statique = []
    Pt_collecte_statique = []
    Pt_ancrage_statique = []

    M_leftfront = []
    Pt_leftfront = []
    F_totale_leftfront = []
    F_point_leftfront = []
    Pt_collecte_leftfront = []
    Pt_ancrage_leftfront = []

    M_left = []
    Pt_left = []
    F_totale_left = []
    F_point_left = []
    Pt_collecte_left = []
    Pt_ancrage_left = []

    # Raideurs#
    k = np.array(Solution[:12])
    print("k = " + str(k))

    # Masses#
    for i in range(0, 9):  # nombre essais centrefront
        M_centrefront.append(np.array(Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_centrefront_" + str(i) + " = " + str(M_centrefront[i]))

    for i in range(9, 18):  # nb essais statique : 9
        M_statique.append(np.array(Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_statique_" + str(i) + " = " + str(M_statique[i]))

    for i in range(18, 27):  # nb essais leftfront: 9
        M_leftfront.append(np.array(Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_leftfront_" + str(i) + " = " + str(M_leftfront[i]))

    for i in range(27, 36):  # nb essais left: 9
        M_left.append(np.array(Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_left_" + str(i) + " = " + str(M_left[i]))

    # Points#
    ###centrefront###
    for i in range(0, 9):
        Pt_centrefront.append(
            np.reshape(Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i], (n * m, 3))
        )
        F_totale_centrefront.append(
            Calcul_Pt_F(
                Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[0]
        )
        F_point_centrefront.append(
            Calcul_Pt_F(
                Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[1]
        )

        F_totale_centrefront[i] = cas.evalf(F_totale_centrefront[i])
        F_point_centrefront[i] = cas.evalf(F_point_centrefront[i])
        F_point_centrefront[i] = np.array(F_point_centrefront[i])

        Pt_collecte_centrefront.append(np.array(Pt_collecte[i]))
        Pt_ancrage_centrefront.append(np.array(Pt_ancrage[i]))
    ###statique###
    for i in range(9, 18):
        Pt_statique.append(np.reshape(Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i], (n * m, 3)))
        F_totale_statique.append(
            Calcul_Pt_F(
                Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[0]
        )
        F_point_statique.append(
            Calcul_Pt_F(
                Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[1]
        )

        F_totale_statique[i - 9] = cas.evalf(F_totale_statique[i - 9])
        F_point_statique[i - 9] = cas.evalf(F_point_statique[i - 9])
        F_point_statique[i - 9] = np.array(F_point_statique[i - 9])

        Pt_collecte_statique.append(np.array(Pt_collecte[i]))
        Pt_ancrage_statique.append(np.array(Pt_ancrage[i]))
    ###leftfront###
    for i in range(18, 27):
        Pt_leftfront.append(np.reshape(Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i], (n * m, 3)))
        F_totale_leftfront.append(
            Calcul_Pt_F(
                Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[0]
        )
        F_point_leftfront.append(
            Calcul_Pt_F(
                Solution[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + n * m * 3 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[1]
        )

        F_totale_leftfront[i - 18] = cas.evalf(F_totale_leftfront[i - 18])
        F_point_leftfront[i - 18] = cas.evalf(F_point_leftfront[i - 18])
        F_point_leftfront[i - 18] = np.array(F_point_leftfront[i - 18])

        Pt_collecte_leftfront.append(np.array(Pt_collecte[i]))
        Pt_ancrage_leftfront.append(np.array(Pt_ancrage[i]))
    ###left###
    for i in range(27, 36):
        Pt_left.append(np.reshape(Solution[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i], (n * m, 3)))
        F_totale_left.append(
            Calcul_Pt_F(
                Solution[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + 405 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[0]
        )
        F_point_left.append(
            Calcul_Pt_F(
                Solution[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                Solution[:12],
                ind_masse[i],
                Solution[12 + 405 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[1]
        )

        F_totale_left[i - 27] = cas.evalf(F_totale_left[i - 27])
        F_point_left[i - 27] = cas.evalf(F_point_left[i - 27])
        F_point_left[i - 27] = np.array(F_point_left[i - 27])

        Pt_collecte_left.append(np.array(Pt_collecte[i]))
        Pt_ancrage_left.append(np.array(Pt_ancrage[i]))

    end_main = time.time()
    print("**************************************************************************")
    print("Temps total : " + str(end_main - start_main))
    print("**************************************************************************")

    #######################################################################################################################

    # ENREGISTREMENT PICKLE#
    path = "results/result_multi_essais/" + "multi_essais_corrigé" + ".pkl"
    with open(path, "wb") as file:
        pickle.dump(Solution, file)
        pickle.dump(labels, file)
        pickle.dump(Pt_collecte, file)
        pickle.dump(Pt_ancrage, file)
        pickle.dump(ind_masse, file)
        pickle.dump(f, file)
        pickle.dump(dict_fixed_params, file)
        pickle.dump(trial_name, file)

    # Comparaison entre collecte et points optimisés des essais choisis

    # CENTREFRONT#
    fig = plt.figure()
    for i in range(0, 9):
        ax = plt.subplot(3, 3, i + 1, projection="3d")
        ax.set_box_aspect([1.1, 1.8, 1])
        ax.plot(
            Pt_centrefront[i][:, 0],
            Pt_centrefront[i][:, 1],
            Pt_centrefront[i][:, 2],
            "+r",
            label="Points de la toile optimisés",
        )
        ax.plot(
            Pt_ancrage_centrefront[i][:, 0],
            Pt_ancrage_centrefront[i][:, 1],
            Pt_ancrage_centrefront[i][:, 2],
            ".k",
            label="Points d'ancrage simulés",
        )
        ax.plot(
            Pt_centrefront[i][ind_masse[i], 0],
            Pt_centrefront[i][ind_masse[i], 1],
            Pt_centrefront[i][ind_masse[i], 2],
            "+y",
            label="Point optimisés le plus bas d'indice " + str(ind_masse[i]),
        )
        ax.plot(
            Pt_collecte_centrefront[i][0, :],
            Pt_collecte_centrefront[i][1, :],
            Pt_collecte_centrefront[i][2, :],
            ".b",
            label="Points collecte",
        )
        label_masse = labels[i].index("t" + str(ind_masse[i]))
        ax.plot(
            Pt_collecte_centrefront[i][0, label_masse],
            Pt_collecte_centrefront[i][1, label_masse],
            Pt_collecte_centrefront[i][2, label_masse],
            "og",
            label="Point collecte le plus bas " + labels[i][label_masse],
        )
        plt.legend()
        plt.title("ESSAI" + str(trial_name[i]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    # STATIQUE#
    fig = plt.figure()
    for i in range(0, 9):
        ax = plt.subplot(3, 3, i + 1, projection="3d")
        ax.set_box_aspect([1.1, 1.8, 1])
        ax.plot(
            Pt_statique[i][:, 0], Pt_statique[i][:, 1], Pt_statique[i][:, 2], "+r", label="Points de la toile optimisés"
        )
        ax.plot(
            Pt_ancrage_statique[i][:, 0],
            Pt_ancrage_statique[i][:, 1],
            Pt_ancrage_statique[i][:, 2],
            ".k",
            label="Points d'ancrage simulés",
        )
        ax.plot(
            Pt_statique[i][ind_masse[i + 9], 0],
            Pt_statique[i][ind_masse[i + 9], 1],
            Pt_statique[i][ind_masse[i + 9], 2],
            "+y",
            label="Point optimisés le plus bas d'indice " + str(ind_masse[i + 9]),
        )
        ax.plot(
            Pt_collecte_statique[i][0, :],
            Pt_collecte_statique[i][1, :],
            Pt_collecte_statique[i][2, :],
            ".b",
            label="Points collecte",
        )
        label_masse = labels[i + 9].index("t" + str(ind_masse[i + 9]))
        ax.plot(
            Pt_collecte_statique[i][0, label_masse],
            Pt_collecte_statique[i][1, label_masse],
            Pt_collecte_statique[i][2, label_masse],
            "og",
            label="Point collecte le plus bas " + labels[i + 9][label_masse],
        )
        plt.legend()
        plt.title("ESSAI" + str(trial_name[i + 9]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    # LEFTFRONT#
    fig = plt.figure()
    for i in range(0, 9):
        ax = plt.subplot(3, 3, i + 1, projection="3d")
        ax.set_box_aspect([1.1, 1.8, 1])
        ax.plot(
            Pt_leftfront[i][:, 0],
            Pt_leftfront[i][:, 1],
            Pt_leftfront[i][:, 2],
            "+r",
            label="Points de la toile optimisés",
        )
        ax.plot(
            Pt_ancrage_leftfront[i][:, 0],
            Pt_ancrage_leftfront[i][:, 1],
            Pt_ancrage_leftfront[i][:, 2],
            ".k",
            label="Points d'ancrage simulés",
        )
        ax.plot(
            Pt_leftfront[i][ind_masse[i + 18], 0],
            Pt_leftfront[i][ind_masse[i + 18], 1],
            Pt_leftfront[i][ind_masse[i + 18], 2],
            "+y",
            label="Point optimisés le plus bas d'indice " + str(ind_masse[i + 18]),
        )
        ax.plot(
            Pt_collecte_leftfront[i][0, :],
            Pt_collecte_leftfront[i][1, :],
            Pt_collecte_leftfront[i][2, :],
            ".b",
            label="Points collecte",
        )
        label_masse = labels[i + 18].index("t" + str(ind_masse[i + 18]))
        ax.plot(
            Pt_collecte_leftfront[i][0, label_masse],
            Pt_collecte_leftfront[i][1, label_masse],
            Pt_collecte_leftfront[i][2, label_masse],
            "og",
            label="Point collecte le plus bas " + labels[i + 18][label_masse],
        )
        plt.legend()
        plt.title("ESSAI" + str(trial_name[i + 18]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    # LEFT#
    fig = plt.figure()
    for i in range(0, 9):
        ax = plt.subplot(3, 3, i + 1, projection="3d")
        ax.set_box_aspect([1.1, 1.8, 1])
        ax.plot(Pt_left[i][:, 0], Pt_left[i][:, 1], Pt_left[i][:, 2], "+r", label="Points de la toile optimisés")
        ax.plot(
            Pt_ancrage_left[i][:, 0],
            Pt_ancrage_left[i][:, 1],
            Pt_ancrage_left[i][:, 2],
            ".k",
            label="Points d'ancrage simulés",
        )
        ax.plot(
            Pt_left[i][ind_masse[i + 27], 0],
            Pt_left[i][ind_masse[i + 27], 1],
            Pt_left[i][ind_masse[i + 27], 2],
            "+y",
            label="Point optimisés le plus bas d'indice " + str(ind_masse[i + 27]),
        )
        ax.plot(
            Pt_collecte_left[i][0, :],
            Pt_collecte_left[i][1, :],
            Pt_collecte_left[i][2, :],
            ".b",
            label="Points collecte",
        )
        label_masse = labels[i + 27].index("t" + str(ind_masse[i + 27]))
        ax.plot(
            Pt_collecte_left[i][0, label_masse],
            Pt_collecte_left[i][1, label_masse],
            Pt_collecte_left[i][2, label_masse],
            "og",
            label="Point collecte le plus bas " + labels[i + 27][label_masse],
        )
        plt.legend()
        plt.title("ESSAI" + str(trial_name[i + 27]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    plt.show()  # on affiche tous les graphes


if __name__ == "__main__":
    main()
