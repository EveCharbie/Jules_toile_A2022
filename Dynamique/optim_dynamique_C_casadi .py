"""
Optimization of the damping parameters (C) of the trampoline bed.
The elasticity coefficients are taken from the static optimization (K are constants).

Optimization variables:
    - C
    - X
    - Xdot
    - Force athlete «-» toile

The distance between the model points and markers and the difference between the force and the model force plates are
minimized.
"""

import casadi as cas
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from ezc3d import c3d
import time
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns
from scipy import signal
import pickle

from modele_dynamique_nxm_DimentionsReelles import (
    spring_lengths,
    Params,
    Points_ancrage_fix,
    Points_ancrage_repos,
    Spring_bouts_repos,
    Spring_bouts_cross_repos,
    Spring_bouts,
    Spring_bouts_croix,
    Bouts_ressorts_repos,
    static_forces_calc,
    static_force_in_each_point,
    rotation_points,
    interpolation_collecte,
    interpolation_collecte_nan,
    Point_ancrage,
    Resultat_PF_collecte,
)


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)

###################################################
# --- FONCTIONS AVEC LES PARAMÈTRES VARIABLES --- #
###################################################


def Param_variable(C_symetrie):
    # COEFFICIENTS D'AMORTISSEMENT : la toile est séparéee en 4 quarts dont les C sont les mêmes par symétrie avec le centre
    # C_symetrie = 0.3*np.ones(5*8)
    C = cas.MX.zeros(n * m)

    # coin en bas a droite de la toile : c'est le coin qui définit tous les autres
    C[0:8] = C_symetrie[0:8]
    C[15:23] = C_symetrie[8:16]
    C[30:38] = C_symetrie[16:24]
    C[45:53] = C_symetrie[24:32]
    C[60:68] = C_symetrie[32:40]

    # coin en bas a gauche de la toile :
    C[75:83] = C_symetrie[24:32]
    C[90:98] = C_symetrie[16:24]
    C[105:113] = C_symetrie[8:16]
    C[120:128] = C_symetrie[0:8]

    # coin en haut a droite de la toile :
    C[14:7:-1] = C_symetrie[0:7]
    C[29:22:-1] = C_symetrie[8:15]
    C[44:37:-1] = C_symetrie[16:23]
    C[59:52:-1] = C_symetrie[24:31]
    C[74:67:-1] = C_symetrie[32:39]

    # coin en haut a gauche de la toile :
    C[89:82:-1] = C_symetrie[24:31]
    C[104:97:-1] = C_symetrie[16:23]
    C[119:112:-1] = C_symetrie[8:15]
    C[134:127:-1] = C_symetrie[0:7]

    return C  #  sx


def Param_variable_force(F):
    F_tab = cas.MX.zeros(5, 3)
    F_tab[0, :] = F[:3]
    F_tab[1, :] = F[3:6]
    F_tab[2, :] = F[6:9]
    F_tab[3, :] = F[9:12]
    F_tab[4, :] = F[12:15]
    return F_tab


#
# def Bouts_ressorts_repos(Pt_ancrage, Pos_repos):
#     Spring_bout_1, Spring_bout_2 = Spring_bouts_repos(Pos_repos, Pt_ancrage)
#     Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_cross_repos(Pos_repos)
#     return Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2


def Force_amortissement(Xdot, C):
    C = Param_variable(C)
    F_amortissement = cas.MX.zeros((n * m, 3))
    for point in range(n * m):
        F_amortissement[point, 2] = -C[point] * Xdot[point, 2] ** 2  # turbulent = amortissement quadratique

    return F_amortissement


def Force_totale_points(X, Xdot, C, F, ind_masse):
    Pt = X
    F = Param_variable_force(F)
    # spring_lengths = spring_lengths() -> new
    # k, M = Param() -> new
    # dict_fixed_params = Param_fixe()
    Pt_ancrage = Points_ancrage_fix(dict_fixed_params)
    Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2 = Bouts_ressorts_repos(Pt_ancrage, Pt)

    F_spring, F_spring_croix, F_masses = static_forces_calc(
        Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params
    )
    F_amortissement = Force_amortissement(Xdot, C)

    F_point = static_force_in_each_point(F_spring, F_spring_croix, F_masses, F_amortissement)

    # ajout des forces de l'athlete
    F_point[ind_masse, :] -= F[0, :]
    F_point[ind_masse + 1, :] -= F[1, :]
    F_point[ind_masse - 1, :] -= F[2, :]
    F_point[ind_masse + 15, :] -= F[3, :]
    F_point[ind_masse - 15, :] -= F[4, :]

    F_tot = cas.MX.zeros((1, 3))
    for point in range(n * m):
        F_tot[0, 0] += F_point[point, 0]
        F_tot[0, 1] += F_point[point, 1]
        F_tot[0, 2] += F_point[point, 2]

    return F_tot


def tab2list(tab):
    list = cas.MX.zeros(n * m * 3)
    for i in range(n * m):
        for j in range(3):
            list[j + 3 * i] = tab[i, j]
    return list


def list2tab(list):
    tab = cas.MX.zeros(n * m, 3)
    for ind in range(n * m):
        for i in range(3):
            tab[ind, i] = list[i + 3 * ind]
    return tab


#
# def Integration(X, Xdot, F, C, Masse_centre, ind_masse):
#     dict_fixed_params = Param_fixe(Masse_centre)
#     Pt_ancrage = Points_ancrage_fix(dict_fixed_params)
#     M = dict_fixed_params["M"]
#     dt = 1 / 500
#     Pt = list2tab(X)
#     Vitesse = list2tab(Xdot)
#     F = Param_variable_force(F)  # bonnes dimensions
#
#     # initialisation
#     Pt_integ = cas.MX.zeros((n * m, 3))
#     vitesse_calc = cas.MX.zeros((n * m, 3))
#     accel_calc = cas.MX.zeros((n * m, 3))
#
#     Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2 = Bouts_ressorts_repos(Pt_ancrage, Pt)
#     F_spring, F_spring_croix, F_masses = static_forces_calc(
#         Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params
#     )
#     F_amortissement = Force_amortissement(Vitesse, C)
#     F_point = static_force_in_each_point(F_spring, F_spring_croix, F_masses, F_amortissement)
#     # ajout des forces de l'athlete
#     F_point[ind_masse, :] -= F[0, :]
#     F_point[ind_masse + 1, :] -= F[1, :]
#     F_point[ind_masse - 1, :] -= F[2, :]
#     F_point[ind_masse + 15, :] -= F[3, :]
#     F_point[ind_masse - 15, :] -= F[4, :]
#
#     for i in range(0, n * m):
#         # acceleration
#         accel_calc[i, :] = F_point[i, :] / M[i]
#         # vitesse
#         vitesse_calc[i, :] = dt * accel_calc[i, :] + Vitesse[i, :]
#         # position
#         Pt_integ[i, :] = dt * vitesse_calc[i, :] + Pt[i, :]
#
#     return Pt_integ, vitesse_calc


def Acceleration_cadre(Pt, total_frame):
    masse_cadre = 270
    dt = 1 / 500

    axe = 11  # point du cadre qui existe presque toujours
    time = np.linspace(0, 10, total_frame)
    time2 = np.linspace(0, 10, total_frame - 2)

    x, y, z = [], [], []
    accx, accy, accz = [], [], []
    fx, fy, fz = [], [], []
    for i in Pt:
        x.append(i[axe, 0])
        y.append(i[axe, 1])
        z.append(i[axe, 2])

    a, b = signal.butter(1, 0.15)
    zfil = signal.filtfilt(a, b, z, method="gust")
    yfil = signal.filtfilt(a, b, y, method="gust")
    xfil = signal.filtfilt(a, b, x, method="gust")

    for pos in range(1, len(zfil) - 1):
        accz.append(((zfil[pos + 1] + zfil[pos - 1] - 2 * zfil[pos]) / (dt**2)))
        accy.append(((yfil[pos + 1] + yfil[pos - 1] - 2 * yfil[pos]) / (dt**2)))
        accx.append(((xfil[pos + 1] + xfil[pos - 1] - 2 * xfil[pos]) / (dt**2)))
        fz.append(((zfil[pos + 1] + zfil[pos - 1] - 2 * zfil[pos]) / (dt**2)) * (masse_cadre))
        fy.append(((yfil[pos + 1] + yfil[pos - 1] - 2 * yfil[pos]) / (dt**2)) * (masse_cadre))
        fx.append(((xfil[pos + 1] + xfil[pos - 1] - 2 * xfil[pos]) / (dt**2)) * (masse_cadre))

    fig, ax = plt.subplots(2, 3)
    fig.suptitle("Position sur Z du point d'ancrage")

    ax[0, 0].plot(time[: len(Pt)], x, label="Données collectées")
    ax[0, 0].plot(time[: len(Pt)], xfil, "-r", label="Données filtrées")
    ax[0, 0].set_xlabel("Temps (s)")
    ax[0, 0].set_ylabel("X (m)")
    # ax[1,0].plot(time2, accx, '-g')
    ax[1, 0].plot(time2[: len(fx)], fx, color="lime", marker="o")
    ax[1, 0].set_xlabel("Temps (s)")
    ax[1, 0].set_ylabel("Force cadre X (N)")

    ax[0, 1].plot(time[: len(Pt)], y)
    ax[0, 1].plot(time[: len(Pt)], yfil, "-r")
    ax[0, 1].set_xlabel("Temps (s)")
    ax[0, 1].set_ylabel("Y (m)")
    # ax[1,1].plot(time2, accy, '-g')
    ax[1, 1].plot(time2[: len(fy)], fy, color="lime", marker="o")
    ax[1, 1].set_xlabel("Temps (s)")
    ax[1, 1].set_ylabel("Force cadre Y (N)")

    ax[0, 2].plot(time[: len(Pt)], z)
    ax[0, 2].plot(time[: len(Pt)], zfil, "-r")
    ax[0, 2].set_xlabel("Temps (s)")
    ax[0, 2].set_ylabel("Z (m)")
    # ax[1,2].plot(time2, accz, '-g', label = 'Accélération')
    ax[1, 2].plot(time2[: len(fz)], fz, color="lime", marker="o", label="Force de l'accélération du cadre")
    ax[1, 2].set_xlabel("Temps (s)")
    ax[1, 2].set_ylabel("Force cadre Z (N)")

    fig.legend(shadow=True)
    plt.show()

    return np.array((fx, fy, fz))


def Interpolation_ancrage(liste_point_ancrage, label_ancrage):
    Pt_anc_interp = np.zeros((48, 3))
    Pt_anc_interp[:, :] = np.nan
    for ind in range(48):
        if "C" + str(ind) in label_ancrage:
            Pt_anc_interp[ind, :] = liste_point_ancrage[label_ancrage.index("C" + str(ind)), :]

    return Pt_anc_interp


#####################################################################################################################


def Optimisation():  # main
    def kC_bounds(Uk_C):  # initial guess pour les C
        """
        :param Uk_C: symolique des C, pour la shape
        :return: bounds et init de C
        """

        C = [10] * Uk_C.shape[0]
        w0_C = C
        lbw_C = [1e-3] * Uk_C.shape[0]
        ubw_C = [1e3] * Uk_C.shape[0]

        return lbw_C, ubw_C, w0_C

    def Pt_bounds(x, Pt_collecte, Pt_ancrage, labels):  # bounds and initial guess
        """
        :param x:
        :param Pos:
        :return: bound et init pour les positions
        """
        lbw_Pt = []
        ubw_Pt = []
        w0_Pt = []

        Pt_inter = interpolation_collecte(Pt_collecte, Pt_ancrage, labels)

        for k in range(n * m * 3):
            if k % 3 == 0:  # limites et guess en x
                lbw_Pt += [Pt_inter[0, int(k // 3)] - 0.3]
                ubw_Pt += [Pt_inter[0, int(k // 3)] + 0.3]
                w0_Pt += [Pt_inter[0, int(k // 3)]]
            if k % 3 == 1:  # limites et guess en y
                lbw_Pt += [Pt_inter[1, int(k // 3)] - 0.3]
                ubw_Pt += [Pt_inter[1, int(k // 3)] + 0.3]
                w0_Pt += [Pt_inter[1, int(k // 3)]]
            if k % 3 == 2:  # limites et guess en z
                lbw_Pt += [-2]
                ubw_Pt += [0.5]
                w0_Pt += [Pt_inter[2, int(k // 3)]]

        # pt_trace = np.array(Pt_inter)
        # fig = plt.figure(1)
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect([1.1, 1.8, 1])
        # ax.plot(pt_trace[0, :], pt_trace[1, :], pt_trace[2, :], '.b')
        # plt.show()

        return lbw_Pt, ubw_Pt, w0_Pt

    def Vitesse_bounds(Ptavant, Ptapres):
        """
        :param Ptavant: instant i-1
        :param Ptapres: instant i+1
        :return: bound et init pour les vitesses en i
        """

        lbw_v = [-30] * n * m * 3
        ubw_v = [30] * n * m * 3

        position_imoins1 = interpolation_collecte_nan(Ptavant, labels)
        position_iplus1 = interpolation_collecte_nan(Ptapres, labels)
        distance_xyz = np.abs(position_imoins1 - position_iplus1)
        vitesse_initiale = distance_xyz / (2 * 0.002)
        vitesse_initiale = vitesse_initiale.T

        w0_v = []
        for i in range(n * m):
            for j in range(3):
                w0_v.append(vitesse_initiale[i, j])

        w0_v = [1] * n * m * 3

        return lbw_v, ubw_v, w0_v

    def Force_bounds():
        """
        :return: bounds et init pour la force de l'athlete sur la toile
        on limite la force sur x et y
        """
        ubw_F = [1e5] * 15
        ubw_F[::3] = [1e3] * 5
        ubw_F[1::3] = [1e3] * 5

        lbw_F = [-1e5] * 15
        lbw_F[::3] = [-1e3] * 5
        lbw_F[1::3] = [-1e3] * 5

        w0_F = [200] * 15
        w0_F[::3] = [20] * 5
        w0_F[1::3] = [20] * 5

        return lbw_F, ubw_F, w0_F

    def A_minimiser(X, Xdot, C, F, Masse_centre, Pt_collecte, Force_collecte, force_accel_cadre, labels, ind_masse):
        """
        Fonction objectif, calculée puis évalueée a partir des variables symboliques
        on minimise :
        - la positions entre collecte et simulation
        - la force entre collecte plateforme et simulation
        """

        Difference = cas.MX.zeros(1)
        Pt = list2tab(X)
        vit = list2tab(Xdot)

        # POSITION
        for i in range(3):
            for ind in range(n * m):
                if "t" + str(ind) in labels:
                    ind_collecte = labels.index("t" + str(ind))
                    if np.isnan(Pt_collecte[i, ind_collecte]):
                        do = "on fait rien sur les nan car pas dinterpolation"
                    elif (
                        ind == ind_masse
                        or ind == ind_masse - 1
                        or ind == ind_masse + 1
                        or ind == ind_masse - 15
                        or ind == ind_masse + 15
                    ):
                        Difference += 500 * (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2
                    else:
                        Difference += (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2

        # FORCE
        Force_point = Force_totale_points(Pt, vit, C, F, Masse_centre, ind_masse)
        Force_plateforme = np.zeros((1, 3))
        Force_plateforme[0, 0] = Force_collecte[0] - force_accel_cadre[0]
        Force_plateforme[0, 1] = Force_collecte[1] - force_accel_cadre[1]
        Force_plateforme[0, 2] = Force_collecte[2] + masse_trampo * 9.81 - force_accel_cadre[2]

        for j in range(3):
            Difference += 1000 * (Force_point[0, j] - Force_plateforme[0, j]) ** 2

        output = Difference
        obj = cas.Function("f", [X, Xdot, C, F], [output]).expand()

        return obj

    #######################################
    #### PREPARATION DE L'OPTIMISATION ####
    #######################################

    # -- choix pour le fichier c3d

    participant = 2
    statique_name = "labeled_statique_leftfront_D7"
    vide_name = "labeled_statique_centrefront_vide"
    trial_name = "labeled_p2_troisquartback_01"

    # -- choix de l'intervalle de frame
    total_frame = 7763
    # intervalle_dyna = [5170, 5190]
    intervalle_dyna = [5170, 5173]
    nb_frame = intervalle_dyna[1] - intervalle_dyna[0]

    # -- définition des parametres fixes
    masse_ressorts = 8 * 0.553 + 110 * 0.322
    masse_toile = 5
    masse_trampo = 270
    dt = 1 / 500
    n = 15
    m = 9

    # -- parametres qui peuvent varier
    masses = [64.5, 87.2]
    Masse_centre = masses[0]

    # -- Recuperation parametres du modele
    dict_fixed_params = Param_fixe(Masse_centre)
    Pt_ancrage, Pos_repos = Points_ancrage_repos(dict_fixed_params)

    # -- Recuperation des donnees de la collecte, pour chaque frame de l'intervalle [k, p]
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(
        participant, statique_name, vide_name, trial_name, intervalle_dyna
    )

    # -- on recupere les points d'ancrage, sur l'intervalle dynamique [k, p]
    Pt_ancrage_collecte, label_ancrage = Point_ancrage(Pt_collecte_tab, labels)
    # accélération frame, de la taille dependant de l'intervalle dynamique et dépendant des points d'ancrage collectés
    force_acceleration_cadre = Acceleration_cadre(Pt_ancrage_collecte, total_frame).T

    ########################
    # --- OPTIMISATION --- #
    ########################

    # -- initialisation
    w = []
    w0 = []
    lbw = []
    ubw = []
    objectif = 0
    g = []
    lbg = []
    ubg = []

    # -- variables symbolique a optimiser
    C_sym = cas.MX.sym("C", 40)

    (
        lbw_C,
        ubw_C,
        w0_C,
    ) = kC_bounds(C_sym)

    w += [C_sym]
    lbw += lbw_C
    ubw += ubw_C
    w0 += w0_C

    X_sym = cas.MX.sym("X_0", n * m * 3)
    Xdot_sym = cas.MX.sym("Xdot_0", n * m * 3)
    F_sym = cas.MX.sym("force_0", 5 * 3)

    lbw_X, ubw_X, w0_X = Pt_bounds(X_sym, Pt_collecte_tab[1], Pt_ancrage, labels)
    lbw_Xdot, ubw_Xdot, w0_Xdot = Vitesse_bounds(Pt_collecte_tab[0], Pt_collecte_tab[2])
    lbw_F, ubw_F, w0_F = Force_bounds()

    w += [X_sym]
    w += [Xdot_sym]
    w += [F_sym]

    lbw += lbw_X
    lbw += lbw_Xdot
    lbw += lbw_F

    ubw += ubw_X
    ubw += ubw_Xdot
    ubw += ubw_F

    w0 += w0_X
    w0 += w0_Xdot
    w0 += w0_F

    # -- on boucle sur le nombre de frame apres avoir géré la frame initiale
    for frame in range(1, nb_frame - 1):  # [k+1,p-1]
        # -- Récupérer forces des plateformes
        Force_plateforme_frame = F_totale_collecte[frame, :]  # force plateforme instant i
        Pt_collecte = Pt_collecte_tab[frame]  # pt collecte instant i

        # -- Recuperation force acceleration cadre
        force_accel_cadre = force_acceleration_cadre[
            frame - 1, :
        ]  # force accel instant i, car par la meme taille que les autres array

        # -- on gere l'objectif a l'instant i
        J = A_minimiser(
            X_sym,
            Xdot_sym,
            C_sym,
            F_sym,
            Masse_centre,
            Pt_collecte,
            Force_plateforme_frame,
            force_accel_cadre,
            labels,
            ind_masse,
        )
        objectif += J(X_sym, Xdot_sym, C_sym, F_sym)

        # -- on integre a partir du frame i
        Pt_integres, V_integrees = Integration(X_sym, Xdot_sym, F_sym, C_sym, Masse_centre, ind_masse)

        # -- definition des nouvelles variables symboliques a l'instant i+1
        X_sym = cas.MX.sym(f"X_{frame}", n * m * 3)
        Xdot_sym = cas.MX.sym(f"Xdot_{frame}", n * m * 3)
        F_sym = cas.MX.sym(f"force_{frame}", 5 * 3)

        lbw_X, ubw_X, w0_X = Pt_bounds(X_sym, Pt_collecte_tab[frame + 1], Pt_ancrage, labels)
        lbw_Xdot, ubw_Xdot, w0_Xdot = Vitesse_bounds(Pt_collecte_tab[frame - 1], Pt_collecte_tab[frame + 1])
        (
            lbw_F,
            ubw_F,
            w0_F,
        ) = Force_bounds()

        w += [X_sym]
        w += [Xdot_sym]
        w += [F_sym]

        lbw += lbw_X
        lbw += lbw_Xdot
        lbw += lbw_F

        ubw += ubw_X
        ubw += ubw_Xdot
        ubw += ubw_F

        w0 += w0_X
        w0 += w0_Xdot
        w0 += w0_F

        # -- Contrainte de continuité en tout les points entre la frame i et la frame i+1 (position ET vitesse)
        for i in range(n * m):  # attention aux nan tous les points nexistent pas
            for j in range(3):
                g += [Pt_integres[i, j] - X_sym[j::3][i]]
                g += [V_integrees[i, j] - Xdot_sym[j::3][i]]
        lbg += [0] * (n * m * 3 * 2)
        ubg += [0] * (n * m * 3 * 2)

    # -- Creation du solver
    prob = {"f": objectif, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    # opts={"ipopt" : {"linear_solver" : "ma57", "tol" : 1e-4, "constr_viol_tol" : 1e-4, "constr_inf_tol" : 1e-4, "hessian_approximation" : "limited-memory"}}
    opts = {
        "ipopt": {"max_iter": 10000, "linear_solver": "ma57"}
    }  # , "tol": 1e-4, "hessian_approximation" : "limited-memory"}}
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # -- Resolution
    # sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    sol = solver(
        x0=cas.vertcat(*w0), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg), lbx=cas.vertcat(*lbw), ubx=cas.vertcat(*ubw)
    )
    w_opt = sol["x"].full().flatten()

    path = "/home/lim/Documents/Jules/Jules_toile_A2022/Dynamique/Result/Optim_C.plk"
    with open(path, "wb") as file:
        pickle.dump(sol, file)
        pickle.dump(w0, file)
        pickle.dump(ubw, file)
        pickle.dump(lbw, file)
        pickle.dump(labels, file)
        pickle.dump(Pt_collecte_tab, file)
        pickle.dump(F_totale_collecte, file)
        pickle.dump(sol["f"], file)

    return w_opt


solution = Optimisation()
