"""
Verification of the k values obtained in the training pool on the trial pool.
Here the only optimization variables are the positions of the markers, the k are constants.
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

from Optim_35_essais_kM_regul_koblique import Param_fixe, Spring_bouts, Spring_bouts_croix, Points_ancrage_repos


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


def Param_variable(Masse, ind_masse):
    """
    Dispatch the k parameters on the different types of springs.
    The k values are the values obtained using the static optimization in Optim_35_essais_kM_regul_koblique.py.
    """

    # RESULTS FROM THE STATIC OPTIMIZATION
    k1 = 1.21175669e05
    k2 = 3.20423906e03
    k3 = 4.11963416e03
    k4 = 2.48125477e03
    k5 = 7.56820743e03
    k6 = 4.95811865e05
    k7 = 1.30776275e-03
    k8 = 3.23131678e05
    k_oblique_1 = 7.48735556e02
    k_oblique_2 = 1.08944449e-04
    k_oblique_3 = 3.89409909e03
    k_oblique_4 = 1.04226031e-04

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    # k_bord = np.zeros(Nb_ressorts_cadre)
    k_bord = cas.MX.zeros(Nb_ressorts_cadre)
    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m : 2 * n + m] = k2, k2
    # cotes horizontaux :
    k_bord[n : n + m], k_bord[2 * n + m : 2 * n + 2 * m] = k4, k4
    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    # ressorts horizontaux dans la toile
    k_horizontaux = k6 * cas.MX.ones(n * (m - 1))
    k_horizontaux[0 : n * (m - 1) : n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1 : n * (m - 1) : n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    k_verticaux = k8 * cas.MX.ones(m * (n - 1))
    k_verticaux[0 : m * (n - 1) : m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1 : n * m - m : m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche

    k = cas.vertcat(k_horizontaux, k_verticaux)
    k = cas.vertcat(k_bord, k)

    ######################################################################################################################

    # RESSORTS OBLIQUES
    # milieux :
    k_oblique = cas.MX.zeros(Nb_ressorts_croix)

    # coins :
    k_oblique[0], k_oblique[1] = k_oblique_1, k_oblique_1  # en bas a droite
    k_oblique[2 * (n - 1) - 1], k_oblique[2 * (n - 1) - 2] = k_oblique_1, k_oblique_1  # en haut a droite
    k_oblique[Nb_ressorts_croix - 1], k_oblique[Nb_ressorts_croix - 2] = k_oblique_1, k_oblique_1  # en haut a gauche
    k_oblique[2 * (n - 1) * (m - 2)], k_oblique[2 * (n - 1) * (m - 2) + 1] = k_oblique_1, k_oblique_1  # en bas a gauche

    # côtés verticaux :
    k_oblique[2 : 2 * (n - 1) - 2] = k_oblique_2  # côté droit
    k_oblique[2 * (n - 1) * (m - 2) + 2 : Nb_ressorts_croix - 2] = k_oblique_2  # côté gauche

    # côtés horizontaux :
    k_oblique[28:169:28], k_oblique[29:170:28] = k_oblique_3, k_oblique_3  # en bas
    k_oblique[55:196:28], k_oblique[54:195:28] = k_oblique_3, k_oblique_3  # en haut

    # milieu :
    for j in range(1, 7):
        k_oblique[2 + 2 * j * (n - 1) : 26 + 2 * j * (n - 1)] = k_oblique_4

    ######################################################################################################################
    # masse des points de la toile
    Mtrampo = 5.00
    mressort_bord = 0.322
    mressort_coin = 0.553
    mattache = 0.025  # attache metallique entre trampo et ressort

    mmilieu = Mtrampo / ((n - 1) * (m - 1))  # masse d'un point se trouvant au milieu de la toile
    mpetitbord = 0.5 * (Mtrampo / ((n - 1) * (m - 1))) + 2 * (
        (mressort_bord / 2) + mattache
    )  # masse d un point situé sur le petit bord
    mgrandbord = 0.5 * (Mtrampo / ((n - 1) * (m - 1))) + (33 / 13) * (
        (mressort_bord / 2) + mattache
    )  # masse d un point situé sur le grand bord
    mcoin = (
        0.25 * (Mtrampo / ((n - 1) * (m - 1))) + mressort_coin + 4 * ((mressort_bord / 2) + mattache)
    )  # masse d un point situé dans un coin

    M = mmilieu * cas.MX.ones(n * m)  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n : n * (m - 1) : n] = mpetitbord  # masses du cote bas
    M[2 * n - 1 : n * m - 1 : n] = mpetitbord  # masses du cote haut
    M[1 : n - 1] = mgrandbord  # masse du cote droit
    M[n * (m - 1) + 1 : n * m - 1] = mgrandbord  # masse du cote gauche

    # masse du disque sur les points concernés
    M[ind_masse] += Masse[0]
    M[ind_masse + 1] += Masse[1]
    M[ind_masse - 1] += Masse[2]
    M[ind_masse + 15] += Masse[3]
    M[ind_masse - 15] += Masse[4]

    return k, k_oblique, M

####################################################################################################################

def Optimisation(participant, Masse_centre, trial_name, vide_name, frame, initial_guess, min_energie):  # main
    def m_bounds():  # initial guess pour les k et les m
        """
        Calculer les limites et l'initial guess des k_type
        :return:
        """

        M1 = Masse_centre / 5  # masse centre
        M2 = Masse_centre / 5  # masse centre +1
        M3 = Masse_centre / 5  # masse centre -1
        M4 = Masse_centre / 5  # masse centre +15
        M5 = Masse_centre / 5  # masse centre -15

        w0_m = [M1, M2, M3, M4, M5]
        # for i in range (len(w0_k)) :
        #     w0_k[i] = 1*w0_k[i]

        # lbw_k = [1e-3]*12
        lbw_m = [0.4 * Masse_centre / 5] * 5
        # ubw_k = [1e7]*12 # bornes très larges
        ubw_m = [1.6 * Masse_centre / 5] * 5

        return w0_m, lbw_m, ubw_m

    def Pt_bounds_interp(Pt_collecte, Pt_ancrage, labels, F_totale_collecte):
        """
        Calculer les limites et l'initial guess des coordonnées des points
        :param Pos:
        :return:
        """
        Pt_inter = interpolation_collecte(Pt_collecte, Pt_ancrage, labels)

        # bounds and initial guess
        lbw_Pt = []
        ubw_Pt = []
        w0_Pt = []

        for k in range(405):
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

        # for i in range (3) :
        #     lbw_Pt += [F_totale_collecte[i] - 100]
        #     ubw_Pt += [F_totale_collecte[i] + 100]
        #     w0_Pt += [F_totale_collecte[i]]

        return lbw_Pt, ubw_Pt, w0_Pt

    def Pt_bounds_repos(Pos, Masse_centre):
        """
        Calculer les limites et l'initial guess des coordonnées des points
        :param Pos:
        :return:
        """
        # bounds and initial guess
        lbw_Pt = []
        ubw_Pt = []
        w0_Pt = []

        for k in range(405):
            if k % 3 == 0:  # limites et guess en x
                lbw_Pt += [Pos[int(k // 3), 0] - 0.3]
                ubw_Pt += [Pos[int(k // 3), 0] + 0.3]
                w0_Pt += [Pos[int(k // 3), 0]]
            if k % 3 == 1:  # limites et guess en y
                lbw_Pt += [Pos[int(k // 3), 1] - 0.3]
                ubw_Pt += [Pos[int(k // 3), 1] + 0.3]
                w0_Pt += [Pos[int(k // 3), 1]]
            if k % 3 == 2:  # limites et guess en z
                lbw_Pt += [-2]
                ubw_Pt += [0.5]
                w0_Pt += [Pos[int(k // 3), 2]]

        return lbw_Pt, ubw_Pt, w0_Pt

    # RESULTAT COLLECTE
    F_totale_collecte, Pt_collecte, labels, ind_masse = Resultat_PF_collecte(participant, vide_name, trial_name, frame)

    # PARAM FIXES
    n = 15
    m = 9
    dict_fixed_params = Param_fixe(ind_masse, Masse_centre)
    Pt_ancrage, Pos_repos = Points_ancrage_repos(dict_fixed_params)

    # OPTIMISATION :
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    # NLP VALUES
    M = cas.MX.sym("M", 5)
    X = cas.MX.sym("X", 135 * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...) puis Fxyz
    if initial_guess == "interpolation":
        lbw_Pt, ubw_Pt, w0_Pt = Pt_bounds_interp(Pt_collecte, Pt_ancrage, labels, F_totale_collecte)
    if initial_guess == "repos":
        lbw_Pt, ubw_Pt, w0_Pt = Pt_bounds_repos(Pos_repos, Masse_centre)

    w0_m, lbw_m, ubw_m = m_bounds()

    # w=[k,Pt] :
    w += [M]
    w += [X]
    lbw += lbw_m
    lbw += lbw_Pt
    ubw += ubw_m
    ubw += ubw_Pt
    w0 += w0_m
    w0 += w0_Pt

    # en statique on ne fait pas de boucle sur le temps :
    J = a_minimiser(X, M, F_totale_collecte, Pt_collecte, Pt_ancrage, dict_fixed_params, labels, min_energie, ind_masse)
    obj = J(X, M)

    # fonction contrainte :
    g += [M[0] + M[1] + M[2] + M[3] + M[4] - Masse_centre]
    lbg += [0]
    ubg += [0]

    # Create an NLP solver
    prob = {"f": obj, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    opts = {"ipopt": {"max_iter": 10000, "linear_solver": "ma57"}}
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(
        x0=cas.vertcat(*w0), lbx=cas.vertcat(*lbw), ubx=cas.vertcat(*ubw), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg)
    )
    w_opt = sol["x"].full().flatten()

    return w_opt, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, sol.get("f")


##########################################################################################################################

# PARAM OPTIM :
min_energie = 1  # 0 #1
initial_guess = "interpolation"  #'interpolation' #'repos'

# RÉSULTATS COLLECTE :
frame = 700
participant = 0  # 0 #1 #2
nb_disques = 10  # entre 1 et 11
trial_name = "labeled_statique_D" + str(nb_disques)
vide_name = "labeled_statique_centrefront_vide"
if "front" not in trial_name:
    vide_name = "labeled_statique_vide"

# MASSE
if participant != 0:
    masses = [64.5, 87.2]
    Masse_centre = masses[participant - 1]
    print("masse appliquée pour le participant " + str(participant) + " = " + str(Masse_centre) + " kg")

if participant == 0:
    masses = [0, 27.0, 47.1, 67.3, 87.4, 102.5, 122.6, 142.8, 163.0, 183.1, 203.3, 228.6]
    Masse_centre = masses[nb_disques]
    print("masse appliquée pour " + str(nb_disques) + " disques = " + str(Masse_centre) + " kg")
    print("essai à vide : " + str(vide_name))

##########################################################################################################################

start_main = time.time()

Solution, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, f = Optimisation(
    participant, Masse_centre, trial_name, vide_name, frame, initial_guess, min_energie
)

M = np.array(Solution[0:5])
Pt = np.reshape(Solution[5:], (135, 3))
print("M = " + str(M))
end_main = time.time()

print("**************************************************************************")
print("Temps total : " + str(end_main - start_main))
print("**************************************************************************")

F_totale, F_point = Calcul_Pt_F(Solution[5:], Pt_ancrage, dict_fixed_params, Solution[:5], ind_masse)
F_totale_opt = cas.evalf(F_totale)
F_point = cas.evalf(F_point)
F_point_opt = np.array(F_point)
# delta = longueur_ressort(dict_fixed_params,Solution[12:],Pt_ancrage)
print("force totale optimisée : " + str(F_totale_opt) + " force totale collectée : " + str(F_totale_collecte))
print("différence entre les forces : " + str(F_totale_opt - F_totale_collecte))
print("**************************************************************************")
print("**************************************************************************")


#######################################################################################################################

# Comparaison entre collecte et points optimisés :
Pt_collecte = np.array(Pt_collecte)
Pt_ancrage = np.array(Pt_ancrage)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1.1, 1.8, 1])
ax.plot(Pt[:, 0], Pt[:, 1], Pt[:, 2], "+r", label="Points de la toile optimisés")
ax.plot(Pt_ancrage[:, 0], Pt_ancrage[:, 1], Pt_ancrage[:, 2], ".k", label="Points d'ancrage simulés")
ax.plot(
    Pt[ind_masse, 0],
    Pt[ind_masse, 1],
    Pt[ind_masse, 2],
    "+y",
    label="Point optimisés le plus bas d'indice " + str(ind_masse),
)
ax.plot(Pt_collecte[0, :], Pt_collecte[1, :], Pt_collecte[2, :], ".b", label="Points collecte")
label_masse = labels.index("t" + str(ind_masse))
ax.plot(
    Pt_collecte[0, label_masse],
    Pt_collecte[1, label_masse],
    Pt_collecte[2, label_masse],
    ".g",
    label="Point collecte le plus bas " + labels[label_masse],
)
plt.legend()
plt.title(
    "Essai d'optimisation améliorée : \n Comparaison de l'essai statique "
    + trial_name
    + " \n avec les positions optimisées"
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

# calcul de l'erreur :
# sur la position :
erreur_position = 0
for ind in range(n * m):
    if "t" + str(ind) in labels:
        ind_collecte = labels.index("t" + str(ind))  # ATTENTION gérer les nans
        for i in range(3):
            if np.isnan(Pt_collecte[i, ind_collecte]) == False:  # gérer les nans
                erreur_position += (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2

erreur_force = 0
for ind in range(n * m):
    for i in range(3):
        erreur_force += (F_point_opt[ind, i]) ** 2


print("Erreur sur la position : " + str(erreur_position) + " m")
print("Erreur sur la force : " + str(erreur_force) + " N")


plt.show()
