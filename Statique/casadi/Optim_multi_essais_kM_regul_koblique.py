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

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import sys

from Optim_35_essais_kM_regul_koblique import cost_function

sys.path.append("../")
from enums import InitialGuessType
from utils_static import Param_fixe

sys.path.append("../../Dynamique/")
from utils_dynamic import get_list_results_dynamic, linear_interpolation_collecte, surface_interpolation_collecte, Points_ancrage_repos

sys.path.append("../../Dynamique/iterative_stabilisation/")
from iterative_stabilisation_algo import position_the_points_based_on_the_force


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


def k_bounds(initial_guess):

    if initial_guess == InitialGuessType.WARM_START:
        with open(f"../iterative_stabilisation/results/static_multi_essai.pkl", "rb") as f:
            data = pickle.load(f)
            K = data["K"]
            w0_k = list(K)
            lbw_k = list(K * 0.5)
            ubw_k = list(K * 1.5)

    else:
        raise RuntimeError("Verify that you really want to use this !")
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

        w0_k = [k1, k2, k3, k4, k5, k6, k7, k8, k_oblique1, k_oblique2, k_oblique3, k_oblique4]
        for i in range(len(w0_k)):
            w0_k[i] = 1 * w0_k[i]

        lbw_k = [1e-4] * 12
        ubw_k = [1e7] * 12  # bornes très larges

    return w0_k, lbw_k, ubw_k


def m_bounds(masse_essai, initial_guess):

    if initial_guess == InitialGuessType.WARM_START:
        with open(f"../iterative_stabilisation/results/static_multi_essai.pkl", "rb") as f:
            data = pickle.load(f)
            Ma = data["Ma"]
            w0_m = list(Ma)
            lbw_m = list(Ma * 0.8)
            ubw_m = list(Ma * 1.2)
    else:
        raise RuntimeError("Verify that you really want to use this !")
        w0_m = []
        lbw_m = []
        ubw_m = []
        for i in range(len(masse_essai)):
            M1 = masse_essai[i] / 5  # masse centre
            M2 = masse_essai[i] / 5  # masse centre +1
            M3 = masse_essai[i] / 5  # masse centre -1
            M4 = masse_essai[i] / 5  # masse centre +15
            M5 = masse_essai[i] / 5  # masse centre -15
            w0_m += [M1, M2, M3, M4, M5]
            lbw_m += [0.6 * masse_essai / 5] * 5
            ubw_m += [1.4 * masse_essai / 5] * 5

    return w0_m, lbw_m, ubw_m


def Optimisation(
    Pt_collecte,
    ind_masse,
    Pt_ancrage,
    Masse_centre,
    trial_name,
    initial_guess,
    dict_fixed_params,
    w0_Pt,
    lbw_Pt,
    ubw_Pt,
):
    # PARAM FIXES
    n = 15
    m = 9


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
    w0_k, lbw_k, ubw_k = k_bounds(initial_guess)
    w0_m, lbw_m, ubw_m = m_bounds(Masse_centre, initial_guess)

    K = cas.MX.sym("K", 8)
    w0 += w0_k
    lbw += lbw_k
    ubw += ubw_k
    w += [K]

    obj = 0
    for i in range(len(trial_name)):

        # Ma
        Ma = cas.MX.sym("Ma", 5)
        w0 += w0_m[i*5:(i+1)*5]
        lbw += lbw_m[i*5:(i+1)*5]
        ubw += ubw_m[i*5:(i+1)*5]
        w += [Ma]

        # X
        X = cas.MX.sym("X", n * m * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...)
        lbw += lbw_Pt[i]
        ubw += ubw_Pt[i]
        w0 += w0_Pt[i]
        w += [X]

        # Constraints
        g += [Ma[0] + Ma[1] + Ma[2] + Ma[3] + Ma[4] - Masse_centre[i]]
        lbg += [0]
        ubg += [0]

        # Objective
        obj += cost_function(
            X,
            K,
            Ma,
            Pt_collecte[i],
            Pt_ancrage[i],
            dict_fixed_params,
            ind_masse[i],
        )

    # Create an NLP solver
    prob = {"f": obj, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    opts = {"ipopt": {"max_iter": 100000, "linear_solver": "ma57"}}
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(
        x0=cas.vertcat(*w0), lbx=cas.vertcat(*lbw), ubx=cas.vertcat(*ubw), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg)
    )
    w_opt = sol["x"].full().flatten()

    return w_opt, sol


##########################################################################################################################

def main():
    initial_guess = InitialGuessType.WARM_START
    WITH_K_OBLIQUE = False

    Pts_collecte, Pts_interpolated, Pts_ancrage_interpolated, dict_fixed_params, labels, inds_masse, Masse_centre, Pt_repos, Pt_ancrage_repos, trial_names, w0_Pt, lbw_Pt, ubw_Pt = load_training_pool_data(initial_guess)

    w_opt, sol = Optimisation(
        Pts_collecte,
        inds_masse,
        Pts_ancrage_interpolated,
        Masse_centre,
        trial_names,
        initial_guess,
        dict_fixed_params,
        w0_Pt,
        lbw_Pt,
        ubw_Pt
    )

    embed()
    cost = sol.cost

    K = np.array(w_opt[:8])
    offset = 8

    Ma = w_opt[offset:offset+5]
    offset += 5
    X = w_opt[offset:offset+n*m*3]
    offset += n*m*3
    for i in range(1, len(trial_names)):
        Ma = np.hstack((Ma, w_opt[offset:offset+5]))
        offset += 5
        X = np.hstack((X, w_opt[offset:offset+n*m*3]))
        offset += n*m*3

    data = {"Ma": Ma,
            "K": K,
            "w_opt": w_opt,
            "cost": cost,
            "trial_names": trial_names,
            }
    with open(f"results/static_multi_essai_local_refinement.pkl", "wb") as f:
        pickle.dump(data, f)



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
    k = np.array(w_opt[:12])
    print("k = " + str(k))

    # Masses#
    for i in range(0, 9):  # nombre essais centrefront
        M_centrefront.append(np.array(w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_centrefront_" + str(i) + " = " + str(M_centrefront[i]))

    for i in range(9, 18):  # nb essais statique : 9
        M_statique.append(np.array(w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_statique_" + str(i) + " = " + str(M_statique[i]))

    for i in range(18, 27):  # nb essais leftfront: 9
        M_leftfront.append(np.array(w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_leftfront_" + str(i) + " = " + str(M_leftfront[i]))

    for i in range(27, 36):  # nb essais left: 9
        M_left.append(np.array(w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i]))
    for i in range(0, 9):
        print("M_left_" + str(i) + " = " + str(M_left[i]))

    # Points#
    ###centrefront###
    for i in range(0, 9):
        Pt_centrefront.append(
            np.reshape(w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i], (n * m, 3))
        )
        F_totale_centrefront.append(
            Calcul_Pt_F(
                w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[0]
        )
        F_point_centrefront.append(
            Calcul_Pt_F(
                w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[1]
        )

        F_totale_centrefront[i] = cas.evalf(F_totale_centrefront[i])
        F_point_centrefront[i] = cas.evalf(F_point_centrefront[i])
        F_point_centrefront[i] = np.array(F_point_centrefront[i])

        Pt_collecte_centrefront.append(np.array(Pt_collecte[i]))
        Pt_ancrage_centrefront.append(np.array(Pt_ancrage[i]))
    ###statique###
    for i in range(9, 18):
        Pt_statique.append(np.reshape(w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i], (n * m, 3)))
        F_totale_statique.append(
            Calcul_Pt_F(
                w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[0]
        )
        F_point_statique.append(
            Calcul_Pt_F(
                w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[1]
        )

        F_totale_statique[i - 9] = cas.evalf(F_totale_statique[i - 9])
        F_point_statique[i - 9] = cas.evalf(F_point_statique[i - 9])
        F_point_statique[i - 9] = np.array(F_point_statique[i - 9])

        Pt_collecte_statique.append(np.array(Pt_collecte[i]))
        Pt_ancrage_statique.append(np.array(Pt_ancrage[i]))
    ###leftfront###
    for i in range(18, 27):
        Pt_leftfront.append(np.reshape(w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i], (n * m, 3)))
        F_totale_leftfront.append(
            Calcul_Pt_F(
                w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + n * m * 3 * i + 5 * i : 17 + n * m * 3 * i + 5 * i],
            )[0]
        )
        F_point_leftfront.append(
            Calcul_Pt_F(
                w_opt[17 + n * m * 3 * i + 5 * i : 422 + n * m * 3 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + n * m * 3 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[1]
        )

        F_totale_leftfront[i - 18] = cas.evalf(F_totale_leftfront[i - 18])
        F_point_leftfront[i - 18] = cas.evalf(F_point_leftfront[i - 18])
        F_point_leftfront[i - 18] = np.array(F_point_leftfront[i - 18])

        Pt_collecte_leftfront.append(np.array(Pt_collecte[i]))
        Pt_ancrage_leftfront.append(np.array(Pt_ancrage[i]))
    ###left###
    for i in range(27, 36):
        Pt_left.append(np.reshape(w_opt[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i], (n * m, 3)))
        F_totale_left.append(
            Calcul_Pt_F(
                w_opt[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + 405 * i + 5 * i : 17 + 405 * i + 5 * i],
            )[0]
        )
        F_point_left.append(
            Calcul_Pt_F(
                w_opt[17 + 405 * i + 5 * i : 422 + 405 * i + 5 * i],
                Pt_ancrage[i],
                dict_fixed_params,
                w_opt[:12],
                ind_masse[i],
                w_opt[12 + 405 * i + 5 * i : 17 + 405 * i + 5 * i],
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
        pickle.dump(w_opt, file)
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
