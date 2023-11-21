"""
This code presents the error made by using a static model to simulate a dynamic trampoline deformation.
The static K coefficients were found using static optimization.
The error is the difference from the integration of the position of the points vs the marker position measured during the data collection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from ezc3d import c3d
import seaborn as sns
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from IPython import embed

from enums import MassType, DisplayType, InitialGuessType


# PARAMETRES :
n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote
Masse_centre = 80
# PARAMETRES POUR LA DYNAMIQUE :
dt = 0.002  # fonctionne pour dt<0.004

# NE PAS CHANGER :
Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


def spring_lengths():
    """
    These parameters were measured using the marker positions on an empty trial.
    For the length of the springs attached to the frame, the length was measured on an unloaded sping with a measuring tape.
    """
    # de bas en haut :
    dL = (
        np.array(
            [
                510.71703748,
                261.87522103,
                293.42186099,
                298.42486747,
                300.67352585,
                298.88879749,
                299.6946861,
                300.4158083,
                304.52115312,
                297.72780618,
                300.53723415,
                298.27144226,
                298.42486747,
                293.42186099,
                261.87522103,
                510.71703748,
            ]
        )
        * 0.001
    )  # ecart entre les lignes, de bas en haut
    dL = np.array([np.mean([dL[i], dL[-(i + 1)]]) for i in range(16)])

    dLmilieu = np.array([151.21983556, 153.50844775]) * 0.001
    dLmilieu = np.array([np.mean([dLmilieu[i], dLmilieu[-(i + 1)]]) for i in range(2)])
    # de droite a gauche :
    dl = (
        np.array(
            [
                494.38703513,
                208.96708367,
                265.1669672,
                254.56358938,
                267.84760997,
                268.60351809,
                253.26974254,
                267.02823864,
                208.07894712,
                501.49013437,
            ]
        )
        * 0.001
    )
    dl = np.array([np.mean([dl[i], dl[-(i + 1)]]) for i in range(10)])

    dlmilieu = np.array([126.53897435, 127.45173517]) * 0.001
    dlmilieu = np.array([np.mean([dlmilieu[i], dlmilieu[-(i + 1)]]) for i in range(2)])

    l_droite = np.sum(dl[:5])
    l_gauche = np.sum(dl[5:])

    L_haut = np.sum(dL[:8])
    L_bas = np.sum(dL[8:])

    ################################
    # LONGUEURS AU REPOS
    l_repos = np.zeros(
        Nb_ressorts_cadre
    )  # on fera des append plus tard, l_repos prend bien en compte tous les ressorts non obliques

    # entre la toile et le cadre :
    # ecart entre les marqueurs - taille du ressort en pretension + taille ressort hors trampo
    l_bord_horz = dl[0] - 0.388 + 0.264
    l_bord_vert = dL[0] - 0.388 + 0.264
    l_repos[0:n], l_repos[n + m : 2 * n + m] = l_bord_horz, l_bord_horz
    l_repos[n : n + m], l_repos[2 * n + m : 2 * n + 2 * m] = l_bord_vert, l_bord_vert

    l_bord_coin = np.mean([l_bord_vert, l_bord_horz])  # pas sure !!!
    l_repos[0], l_repos[n - 1], l_repos[n + m], l_repos[2 * n + m - 1] = (
        l_bord_coin,
        l_bord_coin,
        l_bord_coin,
        l_bord_coin,
    )
    l_repos[n], l_repos[n + m - 1], l_repos[2 * n + m], l_repos[2 * (n + m) - 1] = (
        l_bord_coin,
        l_bord_coin,
        l_bord_coin,
        l_bord_coin,
    )

    # dans la toile : on dit que les spring_lengths au repos sont les memes que en pretension
    # ressorts horizontaux internes a la toile :
    l_horz = np.array([dl[j] * np.ones(n) for j in range(1, m)])
    l_horz = np.reshape(l_horz, Nb_ressorts_horz)
    l_repos = np.append(l_repos, l_horz)

    # ressorts verticaux internes a la toile :
    l_vert = np.array([dL[j] * np.ones(m) for j in range(1, n)])
    l_vert = np.reshape(l_vert, Nb_ressorts_vert)
    l_repos = np.append(l_repos, l_vert)

    # ressorts obliques internes a la toile :
    l_repos_croix = []
    for j in range(m - 1):  # on fait colonne par colonne
        l_repos_croixj = np.zeros(n - 1)
        l_repos_croixj[0 : n - 1] = (l_vert[0 : m * (n - 1) : m] ** 2 + l_horz[j * n] ** 2) ** 0.5
        l_repos_croix = np.append(l_repos_croix, l_repos_croixj)
    # dans chaque maille il y a deux ressorts obliques :
    l_repos_croix_double = np.zeros((int(Nb_ressorts_croix / 2), 2))
    for i in range(int(Nb_ressorts_croix / 2)):
        l_repos_croix_double[i] = [l_repos_croix[i], l_repos_croix[i]]
    l_repos_croix = np.reshape(l_repos_croix_double, Nb_ressorts_croix)

    dict_fixed_params = {
        "dL": dL,
        "dLmilieu": dLmilieu,
        "dl": dl,
        "dlmilieu": dlmilieu,
        "l_droite": l_droite,
        "l_gauche": l_gauche,
        "L_haut": L_haut,
        "L_bas": L_bas,
        "l_repos": l_repos,
        "l_repos_croix": l_repos_croix,
    }

    return dict_fixed_params


def Param():
    # Optimal K parameters from the static optimization in N/m

    k1 = 1.21175669e05
    k2 = 3.20423906e03
    k3 = 4.11963416e03
    k4 = 2.48125477e03
    k5 = 7.56820743e03
    k6 = 4.95811865e05
    k7 = 1.30776275e-03
    k8 = 3.23131678e05
    k1ob = 7.48735556e02
    k2ob = 1.08944449e-04
    k3ob = 3.89409909e03
    k4ob = 1.04226031e-04

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    k_bord = np.zeros(Nb_ressorts_cadre)

    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m : 2 * n + m] = k2, k2

    # cotes horizontaux :
    k_bord[n : n + m], k_bord[2 * n + m : 2 * n + 2 * m] = k4, k4

    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    # ressorts horizontaux dans la toile
    k_horizontaux = k6 * np.ones(n * (m - 1))
    k_horizontaux[0 : n * m - 1 : n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1 : n * (m - 1) : n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    k_verticaux = k8 * np.ones(m * (n - 1))
    k_verticaux[0 : m * (n - 1) : m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1 : n * m - 1 : m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche

    k = np.append(k_horizontaux, k_verticaux)
    k = np.append(k_bord, k)

    # ressorts obliques dans la toile
    k_oblique = np.ones(Nb_ressorts_croix)
    k_oblique[0], k_oblique[1] = k1ob, k1ob  # en bas a droite
    k_oblique[2 * (n - 1) - 1], k_oblique[2 * (n - 1) - 2] = k1ob, k1ob  # en haut a droite
    k_oblique[Nb_ressorts_croix - 1], k_oblique[Nb_ressorts_croix - 2] = k1ob, k1ob  # en haut a gauche
    k_oblique[2 * (n - 1) * (m - 2)], k_oblique[2 * (n - 1) * (m - 2) + 1] = k1ob, k1ob  # en bas a gauche

    # côtés verticaux :
    k_oblique[2 : 2 * (n - 1) - 2] = k2ob  # côté droit
    k_oblique[2 * (n - 1) * (m - 2) + 2 : Nb_ressorts_croix - 2] = k2ob  # côté gauche

    # côtés horizontaux :
    k_oblique[28:169:28], k_oblique[29:170:28] = k3ob, k3ob  # en bas
    k_oblique[55:196:28], k_oblique[54:195:28] = k3ob, k3ob  # en haut

    # milieu :
    for j in range(1, 7):
        k_oblique[2 + 2 * j * (n - 1) : 26 + 2 * j * (n - 1)] = k4ob

    ##################################################################################################################
    # COEFFICIENTS D'AMORTISSEMENT a changer
    C = 2 * np.ones(n * m)

    ##################################################################################################################
    # MASSES (pris en compte la masse ajoutee par lathlete) :
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

    M = mmilieu * np.ones((n * m))  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n : n * (m - 1) : n] = mpetitbord  # masses du cote bas
    M[2 * n - 1 : n * m - 1 : n] = mpetitbord  # masses du cote haut
    M[1 : n - 1] = mgrandbord  # masse du cote droit
    M[n * (m - 1) + 1 : n * m - 1] = mgrandbord  # masse du cote gauche

    return k, k_oblique, M, C


def Points_ancrage_repos(dict_fixed_params):
    dL = dict_fixed_params["dL"]
    dl = dict_fixed_params["dl"]
    l_droite = dict_fixed_params["l_droite"]
    l_gauche = dict_fixed_params["l_gauche"]
    L_haut = dict_fixed_params["L_haut"]
    L_bas = dict_fixed_params["L_bas"]

    Pt_repos = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pt_repos[i + j * n, :] = np.array([-np.sum(dl[: j + 1]), np.sum(dL[: i + 1]), 0])

    # The point 0 is the origin, thus we remove his position from the points position
    Pt_repos_new = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pt_repos_new[i + j * n, :] = Pt_repos[i + j * n, :] - Pt_repos[67, :]

    # ancrage :
    Pt_ancrage = np.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pt_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = np.array([0, L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = np.array([0, L_haut, 0]) + np.array([np.sum(dl[1 + j - n : 5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = np.array([0, L_haut, 0]) - np.array([np.sum(dl[5 : j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[k, 1:2] = -Pt_repos_new[k - n - m, 1:2]
        Pt_ancrage[k, 0] = -l_gauche
    # cote bas :
    Pt_ancrage[2 * n + m + 4, :] = np.array([0, -L_bas, 0])

    Pt_ancrage[2 * n + m, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:9]), 0, 0])
    Pt_ancrage[2 * n + m + 1, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:8]), 0, 0])
    Pt_ancrage[2 * n + m + 2, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:7]), 0, 0])
    Pt_ancrage[2 * n + m + 3, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:6]), 0, 0])

    Pt_ancrage[2 * n + m + 5, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[4:5]), 0, 0])
    Pt_ancrage[2 * n + m + 6, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[3:5]), 0, 0])
    Pt_ancrage[2 * n + m + 7, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[2:5]), 0, 0])
    Pt_ancrage[2 * n + m + 8, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[1:5]), 0, 0])

    Pt_ancrage, Pt_repos_new = rotation_points(Pt_ancrage, Pt_repos_new)

    return Pt_ancrage, Pt_repos_new


def Spring_bouts_repos(Pt_repos, Pt_ancrage, time, Nb_increments):
    # Definition des ressorts (position, taille)
    Spring_bout_1 = np.zeros((Nb_increments, Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[time, i, :] = Pt_ancrage[time, i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[time, Nb_ressorts_cadre + i, :] = Pt_repos[time, i, :]

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[time, Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pt_repos[time, i + n * j, :]
            k += 1

    Spring_bout_2 = np.zeros((Nb_increments, Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n):  # points droite du bord de la toile
        Spring_bout_2[time, i, :] = Pt_repos[time, i, :]

    k = 0
    for i in range(n - 1, m * n, n):  # points hauts du bord de la toile
        Spring_bout_2[time, n + k, :] = Pt_repos[time, i, :]
        k += 1

    k = 0
    for i in range(m * n - 1, n * (m - 1) - 1, -1):  # points gauche du bord de la toile
        Spring_bout_2[time, n + m + k, :] = Pt_repos[time, i, :]
        k += 1

    k = 0
    for i in range(n * (m - 1), -1, -n):  # points bas du bord de la toile
        Spring_bout_2[time, 2 * n + m + k, :] = Pt_repos[time, i, :]
        k += 1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k = 0
    for i in range(n, n * m):
        Spring_bout_2[time, Nb_ressorts_cadre + k, :] = Pt_repos[time, i, :]
        k += 1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[time, Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pt_repos[time, i + n * j, :]
            k += 1

    return (Spring_bout_1, Spring_bout_2)


def Spring_bouts_cross_repos(Pt_repos):
    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = np.zeros((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pt_repos[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pt_repos[i, :]
            k += 1

    Spring_bout_croix_2 = np.zeros((Nb_ressorts_croix, 3))
    # Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    # pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    # cetait un peu complique mais ca marche, faut pas le changer
    j = 1
    k = 0
    while j < m:
        for i in range(j * n, (j + 1) * n - 2, 2):
            Spring_bout_croix_2[k, :] = Pt_repos[i + 1, :]
            Spring_bout_croix_2[k + 1, :] = Pt_repos[i, :]
            Spring_bout_croix_2[k + 2, :] = Pt_repos[i + 2, :]
            Spring_bout_croix_2[k + 3, :] = Pt_repos[i + 1, :]
            k += 4
        j += 1

    return Spring_bout_croix_1, Spring_bout_croix_2


def Spring_bouts(Pt, Pt_ancrage):
    # Definition des ressorts (position, taille)
    Spring_bout_1 = np.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[i, :] = Pt_ancrage[i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[Nb_ressorts_cadre + i, :] = Pt[i, :]

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pt[i + n * j, :]
            k += 1
    ####################################################################################################################
    Spring_bout_2 = np.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n):  # points droite du bord de la toile
        Spring_bout_2[i, :] = Pt[i, :]

    k = 0
    for i in range(n - 1, m * n, n):  # points hauts du bord de la toile
        Spring_bout_2[n + k, :] = Pt[i, :]
        k += 1

    k = 0
    for i in range(m * n - 1, n * (m - 1) - 1, -1):  # points gauche du bord de la toile
        Spring_bout_2[n + m + k, :] = Pt[i, :]
        k += 1

    k = 0
    for i in range(n * (m - 1), -1, -n):  # points bas du bord de la toile
        Spring_bout_2[2 * n + m + k, :] = Pt[i, :]
        k += 1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k = 0
    for i in range(n, n * m):
        Spring_bout_2[Nb_ressorts_cadre + k, :] = Pt[i, :]
        k += 1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pt[i + n * j, :]
            k += 1

    return Spring_bout_1, Spring_bout_2


def Spring_bouts_cross(Pt):
    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = np.zeros((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pt[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pt[i, :]
            k += 1

    Spring_bout_croix_2 = np.zeros((Nb_ressorts_croix, 3))
    # Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    # pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    # cetait un peu complique mais ca marche, faut pas le changer
    j = 1
    k = 0
    while j < m:
        for i in range(j * n, (j + 1) * n - 2, 2):
            Spring_bout_croix_2[k, :] = Pt[i + 1, :]
            Spring_bout_croix_2[k + 1, :] = Pt[i, :]
            Spring_bout_croix_2[k + 2, :] = Pt[i + 2, :]
            Spring_bout_croix_2[k + 3, :] = Pt[i + 1, :]
            k += 4
        j += 1

    return Spring_bout_croix_1, Spring_bout_croix_2


def rotation_points(Pt_ancrage, Pt_repos):
    """
    Apply a rotation matrix to the points to get the same orientation as the real markers from the data collection.
    """

    mat_base_collecte = np.array(
        [
            [0.99964304, -0.02650231, 0.00338079],
            [0.02650787, 0.99964731, -0.00160831],
            [-0.00333697, 0.00169736, 0.99999299],
        ]
    )
    # calcul inverse
    # mat_base_inv_np = np.linalg.inv(mat_base_collecte)
    mat_base_inv_np = mat_base_collecte

    Pt_ancrage_new = np.zeros((Nb_ressorts_cadre, 3))
    for index in range(Nb_ressorts_cadre):
        Pt_ancrage_new[index, :] = np.matmul(
            Pt_ancrage[index, :], mat_base_inv_np
        )  # multplication de matrices en casadi

    Pt_repos_new = np.zeros((n * m, 3))
    for index in range(n * m):
        Pt_repos_new[index, :] = np.matmul(Pt_repos[index, :], mat_base_inv_np)

    return Pt_ancrage_new, Pt_repos_new

def static_forces_calc(
    Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params
):  # force en chaque ressort
    k, k_oblique, M, C = Param()
    l_repos = dict_fixed_params["l_repos"]
    l_repos_croix = dict_fixed_params["l_repos_croix"]

    F_spring = np.zeros((Nb_ressorts, 3))
    Vect_unit_dir_F = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts):
        Vect_unit_dir_F[i, :] = (Spring_bout_2[i, :] - Spring_bout_1[i, :]) / np.linalg.norm(
            Spring_bout_2[i, :] - Spring_bout_1[i, :]
        )

    for ispring in range(Nb_ressorts):
        F_spring[ispring, :] = (
            Vect_unit_dir_F[ispring, :]
            * k[ispring]
            * (np.linalg.norm(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring])
        )

    # F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    Vect_unit_dir_F_croix = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts_croix):
        Vect_unit_dir_F_croix[i, :] = (Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :]) / np.linalg.norm(
            Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :]
        )
    for ispring in range(Nb_ressorts_croix):
        F_spring_croix[ispring, :] = (
            Vect_unit_dir_F_croix[ispring, :]
            * k_oblique[ispring]
            * (
                np.linalg.norm(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :])
                - l_repos_croix[ispring]
            )
        )
    F_masses = np.zeros((n * m, 3))
    F_masses[:, 2] = -M * 9.81

    return M, F_spring, F_spring_croix, F_masses


def static_force_in_each_point(F_spring, F_spring_croix, F_masses):
    """
    Computes the static resulting force in each point (elastic force + weight).
    """
    F_spring_points = np.zeros((n * m, 3))

    # - points des coin de la toile : VERIFIE CEST OK
    F_spring_points[0, :] = (
        F_spring[0, :]
        + F_spring[Nb_ressorts_cadre - 1, :]
        - F_spring[Nb_ressorts_cadre, :]
        - F_spring[Nb_ressorts_cadre + Nb_ressorts_horz, :]
        - F_spring_croix[0, :]
    )  # en bas a droite : premier ressort du cadre + dernier ressort du cadre + premiers ressorts horz, vert et croix
    F_spring_points[n - 1, :] = (
        F_spring[n - 1, :]
        + F_spring[n, :]
        - F_spring[Nb_ressorts_cadre + n - 1, :]
        + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + Nb_ressorts_vert - m, :]
        - F_spring_croix[2 * (n - 1) - 1, :]
    )  # en haut a droite
    F_spring_points[(m - 1) * n, :] = (
        F_spring[2 * n + m - 1, :]
        + F_spring[2 * n + m, :]
        + F_spring[Nb_ressorts_cadre + (m - 2) * n, :]
        - F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m - 1, :]
        + F_spring_croix[Nb_ressorts_croix - 2 * (n - 1) + 1, :]
    )  # en bas a gauche
    F_spring_points[m * n - 1, :] = (
        F_spring[n + m - 1, :]
        + F_spring[n + m, :]
        + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz - 1, :]
        + F_spring[Nb_ressorts - 1, :]
        + F_spring_croix[Nb_ressorts_croix - 2, :]
    )  # en haut a gauche

    # - points du bord de la toile> Pour lordre des termes de la somme, on part du ressort cadre puis sens trigo
    # - cote droit VERIFIE CEST OK
    for i in range(1, n - 1):
        F_spring_points[i, :] = (
            F_spring[i, :]
            - F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i, :]
            - F_spring_croix[2 * (i - 1) + 1, :]
            - F_spring[Nb_ressorts_cadre + i, :]
            - F_spring_croix[2 * (i - 1) + 2, :]
            + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1), :]
        )
        # - cote gauche VERIFIE CEST OK
    j = 0
    for i in range((m - 1) * n + 1, m * n - 1):
        F_spring_points[i, :] = (
            F_spring[Nb_ressorts_cadre - m - (2 + j), :]
            + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (j + 1) * m - 1, :]
            + F_spring_croix[Nb_ressorts_croix - 2 * n + 1 + 2 * (j + 2), :]
            + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz - n + j + 1, :]
            + F_spring_croix[Nb_ressorts_croix - 2 * n + 2 * (j + 1), :]
            - F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (j + 2) * m - 1, :]
        )
        j += 1

        # - cote haut VERIFIE CEST OK
    j = 0
    for i in range(2 * n - 1, (m - 1) * n, n):
        F_spring_points[i, :] = (
            F_spring[n + 1 + j, :]
            - F_spring[Nb_ressorts_cadre + i, :]
            - F_spring_croix[(j + 2) * (n - 1) * 2 - 1, :]
            + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (Nb_ressorts_vert + 1) - (m - j), :]
            + F_spring_croix[(j + 1) * (n - 1) * 2 - 2, :]
            + F_spring[Nb_ressorts_cadre + i - n, :]
        )
        j += 1
        # - cote bas VERIFIE CEST OK
    j = 0
    for i in range(n, (m - 2) * n + 1, n):
        F_spring_points[i, :] = (
            F_spring[Nb_ressorts_cadre - (2 + j), :]
            + F_spring[Nb_ressorts_cadre + n * j, :]
            + F_spring_croix[1 + 2 * (n - 1) * j, :]
            - F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + j + 1, :]
            - F_spring_croix[2 * (n - 1) * (j + 1), :]
            - F_spring[Nb_ressorts_cadre + n * (j + 1), :]
        )
        j += 1

    # Points du centre de la toile (tous les points qui ne sont pas en contact avec le cadre)
    # on fait une colonne puis on passe a la colonne de gauche etc
    # dans lordre de la somme : ressort horizontal de droite puis sens trigo
    for j in range(1, m - 1):
        for i in range(1, n - 1):
            F_spring_points[j * n + i, :] = (
                F_spring[Nb_ressorts_cadre + (j - 1) * n + i, :]
                + F_spring_croix[2 * j * (n - 1) - 2 * n + 3 + 2 * i, :]
                - F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i + j, :]
                - F_spring_croix[j * 2 * (n - 1) + i * 2, :]
                - F_spring[Nb_ressorts_cadre + j * n + i, :]
                - F_spring_croix[j * 2 * (n - 1) + i * 2 - 1, :]
                + F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1) + j, :]
                + F_spring_croix[j * 2 * (n - 1) - 2 * n + 2 * i, :]
            )

    F_point = F_masses - F_spring_points

    return F_point


def Resultat_PF_collecte(participant, empty_trial_name, trial_name, jump_frame_index_interval):
    def open_c3d(participant, trial_name):
        dossiers = ["statique", "participant_01", "participant_02"]
        file_path = "../../Data/DataCollection/c3d_files/" + dossiers[participant]
        c3d_file = c3d(file_path + "/" + trial_name + ".c3d")
        return c3d_file

    def matrices():
        """
        Calibration matrices to apply to the force plates.
        """
        # M1,M2,M4 sont les matrices obtenues apres la calibration sur la plateforme 3
        M4_new = [
            [5.4526, 0.1216, 0.0937, -0.0001, -0.0002, 0.0001],
            [0.4785, 5.7700, 0.0178, 0.0001, 0.0001, 0.0001],
            [-0.1496, -0.1084, 24.6172, 0.0000, -0.0000, 0.0002],
            [12.1726, -504.1483, -24.9599, 3.0468, 0.0222, 0.0042],
            [475.4033, 10.6904, -4.2437, -0.0008, 3.0510, 0.0066],
            [-6.1370, 4.3463, -4.6699, -0.0050, 0.0038, 1.4944],
        ]

        M1_new = [
            [2.4752, 0.1407, 0.0170, -0.0000, -0.0001, 0.0001],
            [0.3011, 2.6737, -0.0307, 0.0000, 0.0001, 0.0000],
            [0.5321, 0.3136, 11.5012, -0.0000, -0.0002, 0.0011],
            [20.7501, -466.7832, -8.4437, 1.2666, -0.0026, 0.0359],
            [459.7415, 9.3886, -4.1276, -0.0049, 1.2787, -0.0057],
            [265.6717, 303.7544, -45.4375, -0.0505, -0.1338, 0.8252],
        ]

        M2_new = [
            [2.9967, -0.0382, 0.0294, -0.0000, 0.0000, -0.0000],
            [-0.1039, 3.0104, -0.0324, -0.0000, -0.0000, 0.0000],
            [-0.0847, -0.0177, 11.4614, -0.0000, -0.0000, -0.0000],
            [13.6128, 260.5267, 17.9746, 1.2413, 0.0029, 0.0158],
            [-245.7452, -7.5971, 11.5052, 0.0255, 1.2505, -0.0119],
            [-10.3828, -0.9917, 15.3484, -0.0063, -0.0030, 0.5928],
        ]

        M3_new = [
            [2.8992, 0, 0, 0, 0, 0],
            [0, 2.9086, 0, 0, 0, 0],
            [0, 0, 11.4256, 0, 0, 0],
            [0, 0, 0, 1.2571, 0, 0],
            [0, 0, 0, 0, 1.2571, 0],
            [0, 0, 0, 0, 0, 0.5791],
        ]

        # Experimental zero given by Nexus
        zeros1 = np.array([1.0751899, 2.4828501, -0.1168980, 6.8177500, -3.0313399, -0.9456340])
        zeros2 = np.array([0.0, -2.0, -2.0, 0.0, 0.0, 0.0])
        zeros3 = np.array([0.0307411, -5.0, -4.0, -0.0093422, -0.0079338, 0.0058189])
        zeros4 = np.array([-0.1032560, -3.0, -3.0, 0.2141770, 0.5169040, -0.3714130])

        return M1_new, M2_new, M3_new, M4_new, zeros1, zeros2, zeros3, zeros4

    def matrices_rotation():
        """
        Rotation matrices to apply to the force plates due to imperfect alignement of the force plates.
        """
        theta31 = 0.53 * np.pi / 180
        rot31 = np.array([[np.cos(theta31), -np.sin(theta31)], [np.sin(theta31), np.cos(theta31)]])

        theta34 = 0.27 * np.pi / 180
        rot34 = np.array([[np.cos(theta34), -np.sin(theta34)], [np.sin(theta34), np.cos(theta34)]])

        theta32 = 0.94 * np.pi / 180
        rot32 = np.array([[np.cos(theta32), -np.sin(theta32)], [np.sin(theta32), np.cos(theta32)]])

        return rot31, rot34, rot32

    def plateformes_separees_rawpins(c3d):
        # on garde seulement les Raw pins
        force_labels = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
        ind = []
        for i in range(len(force_labels)):
            if "Raw" in force_labels[i]:
                ind = np.append(ind, i)
        # ind_stop=int(ind[0])
        indices = np.array([int(ind[i]) for i in range(len(ind))])
        ana = c3d["data"]["analogs"][0, indices, :]
        platform1 = ana[0:6, :]  # pins 10 a 15
        platform2 = ana[6:12, :]  # pins 19 a 24
        platform3 = ana[12:18, :]  # pins 28 a 33
        platform4 = ana[18:24, :]  # pins 1 a 6

        platform = np.array([platform1, platform2, platform3, platform4])
        return platform

    def soustracting_the_zero_from_force_paltes(platform):  # soustrait juste la valeur du debut aux raw values
        longueur = np.size(platform[0, 0])
        zero_variable = np.zeros((4, 6))
        for i in range(6):
            for j in range(4):
                zero_variable[j, i] = np.mean(platform[j, i, 0:100])
                platform[j, i, :] = platform[j, i, :] - zero_variable[j, i] * np.ones(longueur)
        return platform

    def force_plates_rearangement(
        platform, jump_frame_index_interval, participant
    ):  # prend les plateformes separees, passe les N en Nmm, calibre, multiplie par mat rotation, met dans la bonne orientation
        M1, M2, M3, M4, zeros1, zeros2, zeros3, zeros4 = matrices()
        rot31, rot34, rot32 = matrices_rotation()

        # N--> Nmm
        platform[:, 3:6] = platform[:, 3:6] * 1000

        # calibration
        platform[0] = np.matmul(M1, platform[0]) * 100
        platform[1] = np.matmul(M2, platform[1]) * 200
        platform[2] = np.matmul(M3, platform[2]) * 100
        platform[3] = np.matmul(M4, platform[3]) * 25

        # matrices de rotation ; nouvelle position des PF
        platform[0, 0:2] = np.matmul(rot31, platform[0, 0:2])
        platform[1, 0:2] = np.matmul(rot32, platform[1, 0:2])
        platform[3, 0:2] = np.matmul(rot34, platform[3, 0:2])

        # bonne orientation ; nouvelle position des PF (pas sure si avant ou apres)
        platform[0, 1] = -platform[0, 1]
        platform[1, 0] = -platform[1, 0]
        platform[2, 0] = -platform[2, 0]
        platform[3, 1] = -platform[3, 1]

        # prendre un point sur 4 pour avoir la même fréquence que les caméras
        platform_new = np.zeros((4, 6, int(np.shape(platform)[2] / 4)))
        for i in range(np.shape(platform)[2]):
            if i % 4 == 0:
                platform_new[:, :, i // 4] = platform[:, :, i]

        platform_new = platform_new[:, :, jump_frame_index_interval[0] : jump_frame_index_interval[1]]

        return platform_new

    def soustracting_zero_from_force_plates(c3d_experimental, c3d_vide, jump_frame_index_interval, participant):  # pour les forces calculees par Vicon
        platform_statique = plateformes_separees_rawpins(c3d_experimental)
        platform_vide = plateformes_separees_rawpins(c3d_vide)
        platform = np.copy(platform_statique)

        # on soustrait les valeurs du fichier a vide
        for j in range(6):
            for i in range(4):
                platform[i, j, :] = platform_statique[i, j, :] - np.mean(platform_vide[i, j, :])
        platform = force_plates_rearangement(platform, jump_frame_index_interval, participant)
        return platform

    def named_markers(c3d_experimental):
        labels = c3d_experimental["parameters"]["POINT"]["LABELS"]["value"]

        indices_supp = []
        for i in range(len(labels)):
            if "*" in labels[i]:
                indices_supp = np.append(indices_supp, i)

        if len(indices_supp) == 0:
            ind_stop = int(len(labels))
        if len(indices_supp) != 0:
            ind_stop = int(indices_supp[0])

        labels = c3d_experimental["parameters"]["POINT"]["LABELS"]["value"][0:ind_stop]
        named_positions = c3d_experimental["data"]["points"][0:3, 0:ind_stop, :]

        if "t67" not in labels:
            raise RuntimeError("deal with the case where the middle marker is not visible")
        ind_milieu = labels.index("t67")
        moyenne_milieu = np.mean(named_positions[:, ind_milieu, :100], axis=1)

        return labels, moyenne_milieu, named_positions

    def dynamics_position(named_positions, moyenne_milieu, jump_frame_index_interval):
        # on soustrait la moyenne de la position du milieu sur les 100 premiers points
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_milieu[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = -named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        position_moyenne = np.zeros((3, np.shape(named_positions_bonsens)[1]))
        for ind_print in range(np.shape(named_positions_bonsens)[1]):
            position_moyenne[:, ind_print] = np.array(
                [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)]
            )

        # passage de mm en m :
        named_positions_bonsens *= 0.001

        positions_new = named_positions_bonsens[:, :, jump_frame_index_interval[0] : jump_frame_index_interval[1]]

        return positions_new

    def find_lowest_marker(points, labels, jump_frame_index_interval):
        """
        Finds the label of the lowest marker on the jump frame interval.
        points: list of the experimental marker position
        labels: labels of the markers
        jump_frame_index_interval: frame interval of the jump
        """
        position_min = 1
        labels_min = None
        not_middle_labels_index = [i for i, str in enumerate(labels) if "M" not in str]
        labels_without_middle = []
        for label in labels:
            if "M" not in label:
                labels_without_middle += [label]
        for frame in range(0, jump_frame_index_interval[1] - jump_frame_index_interval[0]):
            min_on_current_frame = np.nanmin(points[2, not_middle_labels_index, frame])
            min_idx_on_curent_frame = np.nanargmin(points[2, not_middle_labels_index, frame])
            if min_on_current_frame < position_min:
                position_min = min_on_current_frame
                labels_min = labels_without_middle[min_idx_on_curent_frame]
        return position_min, labels_min

    c3d_vide = open_c3d(0, empty_trial_name)
    c3d_experimental = open_c3d(participant, trial_name)
    platform = soustracting_zero_from_force_plates(c3d_experimental, c3d_vide, jump_frame_index_interval, participant)
    labels, moyenne_milieu, named_positions = named_markers(c3d_experimental)
    labels_vide, moyenne_milieu_vide, named_positions_vide = named_markers(c3d_vide)
    Pt_collecte = dynamics_position(named_positions, moyenne_milieu_vide, jump_frame_index_interval)

    longueur = np.size(platform[0, 0])
    F_totale_collecte = np.zeros((longueur, 3))
    for i in range(3):
        for x in range(longueur):
            F_totale_collecte[x, i] = platform[0, i, x] + platform[1, i, x] + platform[2, i, x] + platform[3, i, x]

    # position_instant = Pt_collecte[:, :, int(7050)]
    argmin_marqueur, label_min = find_lowest_marker(
        Pt_collecte, labels, jump_frame_index_interval
    )  # coordonnée du marqueur le plus bas dans labels
    ind_marqueur_min = int(label_min[1:])  # coordonnées de ce marqueur adaptées à la simulation
    print("Point le plus bas sur l'intervalle " + str(jump_frame_index_interval) + " : " + str(label_min))

    # retourner des tableaux casadi
    F_collecte_cas = np.zeros(np.shape(F_totale_collecte))
    F_collecte_cas[:, :] = F_totale_collecte[:, :]

    Pt_collecte_tab = [0 for i in range(np.shape(Pt_collecte)[2])]
    for time in range(np.shape(Pt_collecte)[2]):
        # passage en casadi
        Pt_time = np.zeros((3, np.shape(Pt_collecte)[1]))
        Pt_time[:, :] = Pt_collecte[:, :, time]  # attention pas dans le même ordre que Pt_simu
        # séparation en Nb_increments tableaux
        Pt_collecte_tab[time] = Pt_time

    return F_collecte_cas, Pt_collecte_tab, labels, ind_marqueur_min


def Point_ancrage(Point_collecte, labels):
    """
    Reorder the marker coordinates to match the position they should occupy in the model (only the "C" markers are considered).
    """
    point_ancrage = np.zeros((len(Point_collecte), 2 * (n + m), 3))
    point_ancrage[:, :, :] = np.nan
    label_ancrage = []
    for frame in range(len(Point_collecte)):
        for idx, lab in enumerate(labels):
            if "C" in lab:
                point_ancrage[frame, int(lab[1:]), :] = Point_collecte[frame][:, idx]
            if lab not in label_ancrage:
                label_ancrage.append(lab)
    return point_ancrage, label_ancrage


def Point_toile_init(Point_collecte, labels):
    """
    Reorder the marker coordinates to match the position they should occupy in the model (only the "t" markers are considered).
    """
    point_toile = np.zeros((len(Point_collecte), 3, m * n))
    point_toile[:, :, :] = np.nan
    label_toile = []
    for frame in range(len(Point_collecte)):
        for idx, lab in enumerate(labels):
            if "t" in lab:
                point_toile[frame, :, int(lab[1:])] = Point_collecte[frame][:, idx]
            if lab not in label_toile:
                label_toile.append(lab)
    return point_toile, label_toile

def surface_interpolation_collecte(Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, dict_fixed_params, with_plot=False):
    """
    Interpolate to fill the missing markers.
    """

    n = 15
    m = 9

    dL = dict_fixed_params["dL"]
    dl = dict_fixed_params["dl"]

    Pt_interpolated = np.zeros((len(Pt_collecte), m * n, 3))
    Pt_interpolated[:, :, :] = np.nan
    Pt_ancrage_interpolated = np.zeros((len(Pt_collecte), 2 * (m + n), 3))
    Pt_ancrage_interpolated[:, :, :] = np.nan
    Pt_needs_interpolation = np.ones((len(Pt_collecte), m * n, 3))
    Pt_ancrage_need_interpolation = np.ones((len(Pt_collecte), 2 * (m + n), 3))
    mean_position_ancrage = np.zeros((len(Pt_collecte), 3))
    for frame in range(len(Pt_collecte)):
        if with_plot:
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_box_aspect([1.1, 1.8, 1])

        # Fill markers data that we have
        for ind in range(m * n):
            if np.isnan(Pt_collecte[frame][0, ind]) == False:
                Pt_interpolated[frame, ind, :] = Pt_collecte[frame][:, ind]
                Pt_needs_interpolation[frame, ind, :] = 0
                if with_plot:
                    ax.plot(
                        Pt_interpolated[frame, ind, 0],
                        Pt_interpolated[frame, ind, 1],
                        Pt_interpolated[frame, ind, 2],
                        ".b",
                    )
        for ind in range(2*(m+n)):
            if np.isnan(Pt_ancrage[frame][ind, 0]) == False:
                Pt_ancrage_interpolated[frame, ind, :] = Pt_ancrage[frame][ind, :]
                Pt_ancrage_need_interpolation[frame, ind, :] = 0
                if with_plot:
                    ax.plot(
                        Pt_ancrage_interpolated[frame, ind, 0],
                        Pt_ancrage_interpolated[frame, ind, 1],
                        Pt_ancrage_interpolated[frame, ind, 2],
                        "ob",
                    )

        known_indices_ancrage = np.where(Pt_ancrage_need_interpolation[frame, :, 0] == 0)[0]
        mean_position_ancrage[frame, :] = np.mean(Pt_ancrage_interpolated[frame, known_indices_ancrage, :])

        known_indices = np.where(Pt_needs_interpolation[frame, :, 0] == 0)[0]
        missing_indices = np.where(Pt_needs_interpolation[frame, :, 0] == 1)[0]
        missing_ancrage_indices = np.where(Pt_ancrage_need_interpolation[frame, :, 0] == 1)[0]


        # Split the points by column + interpolation  based on splines
        Pt_column_interpolated = np.zeros((m * n, 3))
        for i in range(m):
            Pt_column_i = np.zeros((n + 2, 3))
            Pt_column_repos_i = np.zeros((n + 2, 3))
            Pt_column_i[:, :] = np.nan
            Pt_column_repos_i[:, :] = np.nan
            Pt_column_i[0, :] = Pt_ancrage_repos[2 * (n + m) - 1 - i, :]
            Pt_column_repos_i[0, :] = Pt_ancrage_repos[2 * (n + m) - 1 - i, :]
            Pt_column_i[1:n + 1, :] = Pt_interpolated[frame, n * i: n * (i + 1), :]
            Pt_column_repos_i[1:n + 1, :] = Pt_repos[n * i: n * (i + 1), :]
            Pt_column_i[-1, :] = Pt_ancrage_repos[n + i, :]
            Pt_column_repos_i[-1, :] = Pt_ancrage_repos[n + i, :]
            nan_idx = np.where(np.isnan(Pt_column_i[:, 0]))[0]
            non_nan_idx = np.where(np.isnan(Pt_column_i[:, 0]) != True)[0]
            if np.sum(nan_idx) > 0:
                if len(Pt_column_i[non_nan_idx, 0]) <= 5:
                    if len(Pt_column_i[non_nan_idx, 0]) < 1:
                        degree = 1
                    else:
                        degree = len(Pt_column_i[non_nan_idx, 0]) - 1
                else:
                    degree = 5

                t_non_normalized = []
                for ii in range(1, n+1):
                    t_non_normalized += [np.sum(dL[:ii])]
                t_new = t_non_normalized / np.sum(dL)
                t_new = t_new[nan_idx-1]

                if t_new.shape != (0,):
                    tck, u = scipy.interpolate.splprep([Pt_column_i[non_nan_idx, 0], Pt_column_i[non_nan_idx, 1], Pt_column_i[non_nan_idx, 2]], k=degree)
                    new_points = scipy.interpolate.splev(t_new, tck)
                    curves_to_plot = scipy.interpolate.splev(np.linspace(0, 1, 100), tck)
                    if with_plot:
                        ax.plot(curves_to_plot[0], curves_to_plot[1], curves_to_plot[2], "g")
                        ax.plot(new_points[0], new_points[1], new_points[2], ".g")
                    idx_added = 0
                    for idx_to_add in range(n+2):
                        if idx_to_add in nan_idx and idx_to_add != 0 and idx_to_add != n+1:
                            Pt_column_interpolated[n * i + idx_to_add-1, 0] = new_points[0][idx_added]
                            Pt_column_interpolated[n * i + idx_to_add-1, 1] = new_points[1][idx_added]
                            Pt_column_interpolated[n * i + idx_to_add-1, 2] = new_points[2][idx_added]
                            idx_added += 1

        # Split the points by row + interpolation  based on splines
        Pt_row_interpolated = np.zeros((m * n, 3))
        for i in range(n):
            Pt_row_i = np.zeros((m + 2, 3))
            Pt_row_repos_i = np.zeros((m + 2, 3))
            Pt_row_i[:, :] = np.nan
            Pt_row_repos_i[:, :] = np.nan
            Pt_row_i[0, :] = Pt_ancrage_repos[i, :]
            Pt_row_repos_i[0, :] = Pt_ancrage_repos[i, :]
            Pt_row_i[1:m + 1, :] = Pt_interpolated[frame, i::n, :]
            Pt_row_repos_i[1:m + 1, :] = Pt_repos[i::n, :]
            Pt_row_i[-1, :] = Pt_ancrage_repos[2*n + m - 1 - i, :]
            Pt_row_repos_i[-1, :] = Pt_ancrage_repos[2*n + m - 1 - i, :]
            nan_idx = np.where(np.isnan(Pt_row_i[:, 0]))[0]
            non_nan_idx = np.where(np.isnan(Pt_row_i[:, 0]) != True)[0]
            if np.sum(nan_idx) > 0:
                if len(Pt_row_i[non_nan_idx, 0]) <= 5:
                    if len(Pt_row_i[non_nan_idx, 0]) < 1:
                        degree = 1
                    else:
                        degree = len(Pt_row_i[non_nan_idx, 0]) - 1
                else:
                    degree = 5

                t_non_normalized = []
                for ii in range(1, n+1):
                    t_non_normalized += [np.sum(dl[:ii])]
                t_new = t_non_normalized / np.sum(dl)
                t_new = t_new[nan_idx-1]

                if t_new.shape != (0,):
                    tck, u = scipy.interpolate.splprep(
                        [Pt_row_i[non_nan_idx, 0], Pt_row_i[non_nan_idx, 1], Pt_row_i[non_nan_idx, 2]],
                        k=degree)
                    new_points = scipy.interpolate.splev(t_new, tck)
                    curves_to_plot = scipy.interpolate.splev(np.linspace(0, 1, 100), tck)
                    if with_plot:
                        ax.plot(curves_to_plot[0], curves_to_plot[1], curves_to_plot[2], "m")
                        ax.plot(new_points[0], new_points[1], new_points[2], ".m")
                    idx_added = 0
                    for idx_to_add in range(m + 2):
                        if idx_to_add in nan_idx and idx_to_add != 0 and idx_to_add != m + 1:
                            Pt_row_interpolated[n * (idx_to_add - 1) + i, 0] = new_points[0][idx_added]
                            Pt_row_interpolated[n * (idx_to_add - 1) + i, 1] = new_points[1][idx_added]
                            Pt_row_interpolated[n * (idx_to_add - 1) + i, 2] = new_points[2][idx_added]
                            idx_added += 1


        grid_from_row_and_columns = np.mean(np.concatenate((Pt_row_interpolated.reshape(m*n, 3, 1), Pt_column_interpolated.reshape(m*n, 3, 1)), axis=2), axis=2)

        # Fit a polynomial surface to the points that we have
        degree = 10
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(
            np.vstack((Pt_interpolated[frame, known_indices, :2], Pt_ancrage_repos[:, :2])),
            np.vstack(
                (
                    np.reshape(Pt_interpolated[frame, known_indices, 2], (-1, 1)),
                    np.reshape(Pt_ancrage_repos[:, 2], (-1, 1)),
                )
            ),
        )

        # Approximate the position of the missing markers + plot
        Z_new = model.predict(grid_from_row_and_columns[missing_indices, :2])
        Z_new[np.where(Z_new > 0.01)] = 0.01

        if with_plot:
            ax.scatter(
                grid_from_row_and_columns[missing_indices, 0],
                grid_from_row_and_columns[missing_indices, 1],
                Z_new,
                marker="x",
                color="r",
                label="Predicted points",
            )
            ax.scatter(
                Pt_ancrage_repos[missing_ancrage_indices, 0],
                Pt_ancrage_repos[missing_ancrage_indices, 1],
                Pt_ancrage_repos[missing_ancrage_indices, 2],
                marker="o",
                color="r",
            )

        xx, yy = np.meshgrid(
            np.linspace(np.min(Pt_ancrage_repos[:, 0]), np.max(Pt_ancrage_repos[:, 0]), 50),
            np.linspace(np.min(Pt_ancrage_repos[:, 1]), np.max(Pt_ancrage_repos[:, 1]), 50),
        )
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        if with_plot:
            ax.plot_surface(xx, yy, zz, alpha=0.5, color="k", label="Fitted Surface")

            ax.legend()
            # ax.view_init(0, 90)
            ax.set_zlim(-1, 1)
            if frame % 10 == 0:
                plt.savefig("../Plots/interpolaion_frame_" + str(frame) + ".png")
            plt.show()

        Pt_interpolated[frame, missing_indices, :2] = grid_from_row_and_columns[missing_indices, :2]
        Pt_interpolated[frame, missing_indices, 2] = Z_new[:, 0]

    mean_mean_position_ancrage = np.mean(mean_position_ancrage, axis=0)
    mean_position_ancrage = mean_position_ancrage - mean_mean_position_ancrage
    mean_position_ancrage = mean_position_ancrage.reshape(len(Pt_collecte), 3)
    for frame in range(len(Pt_collecte)):
        for ind in range(2 * (m + n)):
            if np.isnan(Pt_ancrage_interpolated[frame, ind, 0]):
                Pt_ancrage_interpolated[frame, ind, :2] = Pt_ancrage_repos[ind, :2]
                Pt_ancrage_interpolated[frame, ind, 2] = Pt_ancrage_repos[ind, 2] + mean_position_ancrage[frame][2]

    return Pt_interpolated, Pt_ancrage_interpolated


def spring_bouts_collecte(Pt_interpolated, Pt_ancrage_repos):
    """
    Returns the coordinates of the spring ends from the markers coordinates (linking the right markers together).
    """
    Spring_bouts1, Spring_bouts2 = Spring_bouts(Pt_interpolated.T, Pt_ancrage_repos)
    Spring_bouts_cross1, Spring_bouts_cross2 = Spring_bouts_cross(Pt_interpolated.T)
    return Spring_bouts1, Spring_bouts2, Spring_bouts_cross1, Spring_bouts_cross2


def Affichage_points_collecte_t(Pt_toile, Pt_ancrage, Ressort, nb_frame, ind_masse):
    """

    :param Pt_toile: points de la toile collectés, avec interpolation ou non des points inexistants en Nan
    :param Pt_ancrage: points du cadre collectés, avec interpolation ou non des points inexistants en Nan
    :param ressort: Booleen, if true on affiche les ressorts

    """
    bout1, bout2, boutc1, boutc2 = spring_bouts_collecte(Pt_toile, Pt_ancrage)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(0, 0, -1.2, "ow")  # mettre a une echelle lisible et reelle

    # afficher les points d'ancrage et les points de la toile avec des couleurs differentes
    ax.plot(Pt_ancrage[:, 0], Pt_ancrage[:, 1], Pt_ancrage[:, 2], "ok", label="Point du cadre ")
    ax.plot(Pt_toile[0, :], Pt_toile[1, :], Pt_toile[2, :], "ob", label="Point de la toile")
    ax.plot(
        Pt_toile[0, ind_masse],
        Pt_toile[1, ind_masse],
        Pt_toile[2, ind_masse],
        "og",
        label="Point le plus bas sur l'intervalle dynamics",
    )

    if Ressort == True:
        # affichage des ressort
        for j in range(Nb_ressorts):
            a = []
            a = np.append(a, bout1[j, 0])
            a = np.append(a, bout2[j, 0])

            b = []
            b = np.append(b, bout1[j, 1])
            b = np.append(b, bout2[j, 1])

            c = []
            c = np.append(c, bout1[j, 2])
            c = np.append(c, bout2[j, 2])

            ax.plot3D(a, b, c, "-r", linewidth=1)

        for j in range(Nb_ressorts_croix):
            # pas tres elegant mais cest le seul moyen pour que ca fonctionne
            a = []
            a = np.append(a, boutc1[j, 0])
            a = np.append(a, boutc2[j, 0])

            b = []
            b = np.append(b, boutc1[j, 1])
            b = np.append(b, boutc2[j, 1])

            c = []
            c = np.append(c, boutc1[j, 2])
            c = np.append(c, boutc2[j, 2])

            ax.plot3D(a, b, c, "-g", linewidth=1)

    return ax


def velocity_from_finite_difference(Pt_interpoles, idx_before, idx_after):
    """
    Approximates the velocity of the markers using the finite difference method.
    """
    position_imoins1 = Pt_interpoles[idx_before]
    position_iplus1 = Pt_interpoles[idx_after]
    distance_xyz = position_iplus1 - position_imoins1
    approx_velocity = distance_xyz / (2 * 0.002)
    return approx_velocity


def integration_error(Pt_intergres, Pt_interpoles):
    """
    Computes the error between the point positions computed from integration and the actual markers
    """

    position_iplus1 = Pt_interpoles
    point_theorique = position_iplus1

    err_rel = np.abs(point_theorique - Pt_intergres) / point_theorique
    err_abs = np.abs(point_theorique - Pt_intergres)

    print("STD relative error : " + str(np.nanstd(np.linalg.norm(err_rel, axis=1))))
    print("STD absolute error : " + str(np.nanstd(np.linalg.norm(err_abs, axis=1))))

    return err_rel, err_abs


def update(time, Pt_integrated, Pt_markers, integrated_point, markers_point):
    for i_point in range(len(integrated_point)):
        integrated_point[i_point][0].set_data(
            np.array([Pt_integrated[time, i_point, 0]]), np.array([Pt_integrated[time, i_point, 1]])
        )
        integrated_point[i_point][0].set_3d_properties(np.array([Pt_integrated[time, i_point, 2]]))

    for i_point in range(len(markers_point)):
        markers_point[i_point][0].set_data(
            np.array([Pt_markers[time][0, i_point]]), np.array([Pt_markers[time][1, i_point]])
        )
        markers_point[i_point][0].set_3d_properties(np.array([Pt_markers[time][2, i_point]]))

    return


def multiple_shooting_euler_integration(nb_frame, Pt_interpoles, Pt_ancrage, labels):
    """
    Computes the position of the points by integrating (from the forces and initial conditions)
    -------
    -position des points integres
    -forces calculées a chaque frame en chaque point
    -erreurs de position relative entre points integres et points collectes
    -erreurs de position absolues entre points integres et points collectes
    """

    # --- Initzalization ---#
    Pt_tot = np.zeros((nb_frame, n * m, 3))
    F_all_point = np.zeros((nb_frame, n * m, 3))
    Velocity_tot = np.zeros((nb_frame, n * m, 3))
    relative_error, absolute_error = [], []

    # Velocity of the second frame by finite difference
    current_velocity = velocity_from_finite_difference(Pt_interpoles, idx_before=0, idx_after=2)
    Velocity_tot[0, :, :] = current_velocity.T

    Pt_tot[0, :, :] = Pt_interpoles[0].T

    for frame in range(nb_frame - 1):
        # --- At the current frame ---#
        bt1, bt2, btc1, btc2 = spring_bouts_collecte(Pt_interpoles[frame], Pt_ancrage)

        # Computation the forces based on the state of the model
        M, F_spring, F_spring_croix, F_masses = static_forces_calc(bt1, bt2, btc1, btc2, dict_fixed_params)
        F_point = static_force_in_each_point(F_spring, F_spring_croix, F_masses)

        F_all_point[frame, :, :] = F_point

        accel_current = np.zeros((n * m, 3))
        velocity_next = np.zeros((n * m, 3))
        Pt_integ = np.zeros((n * m, 3))
        for i in range(0, n * m):
            accel_current[i, :] = F_point[i, :] / M[i]
            velocity_next[i, :] = dt * accel_current[i, :] + current_velocity[i, :]
            Pt_integ[i, :] = dt * Velocity_tot[frame, i, :] + Pt_interpoles[frame][i, :]

        # Make sure the intial velocity approximation does not have too much of an effect on the integration
        # by regularizing with the finite difference velocity
        # if frame == 1:
        #     velocity_next = (initial_velocity + velocity_next) / 2

        Pt_tot[frame + 1, :, :] = Pt_integ
        Velocity_tot[frame + 1, :, :] = velocity_next

        # --- Errors ---#
        relative_error.append(integration_error(Pt_integ, Pt_interpoles[frame + 1])[0])
        absolute_error.append(integration_error(Pt_integ, Pt_interpoles[frame + 1])[1])

    return Pt_tot, Velocity_tot, erreur_relative, erreur_absolue, F_all_point


def single_shooting_euler_integration(nb_frame, Pt_interpoles, Pt_ancrage_interpoles, labels):
    """
    Computes the position of the points by integrating (from the forces and initial conditions)
    -------
    -position des points integres
    -forces calculées a chaque frame en chaque point
    -erreurs de position relative entre points integres et points collectes
    -erreurs de position absolues entre points integres et points collectes
    """

    # --- Initzalization ---#
    Pt_tot = np.zeros((nb_frame, n * m, 3))
    F_all_point = np.zeros((nb_frame, n * m, 3))
    Velocity_tot = np.zeros((nb_frame, n * m, 3))
    relative_error, absolute_error = [], []

    # Velocity of the second frame by finite difference
    initial_velocity = velocity_from_finite_difference(Pt_interpoles[0], Pt_interpoles[2], labels)
    Velocity_tot[0, :, :] = initial_velocity
    Pt_tot[0, :, :] = Pt_interpoles[0]

    for frame in range(nb_frame - 1):
        # --- At the current frame ---#
        bt1, bt2, btc1, btc2 = spring_bouts_collecte(Pt_tot[frame], Pt_ancrage_interpoles)

        # Computation the forces based on the state of the model
        M, F_spring, F_spring_croix, F_masses = static_forces_calc(bt1, bt2, btc1, btc2, dict_fixed_params)
        F_point = static_force_in_each_point(F_spring, F_spring_croix, F_masses)

        F_all_point[frame, :, :] = F_point

        accel_current = np.zeros((n * m, 3))
        velocity_next = np.zeros((n * m, 3))
        Pt_integ = np.zeros((n * m, 3))
        for i in range(0, n * m):
            accel_current[i, :] = F_point[i, :] / M[i]
            velocity_next[i, :] = dt * accel_current[i, :] + Velocity_tot[frame, i, :]
            Pt_integ[i, :] = dt * velocity_next[i, :] + Pt_tot.T[frame, i, :]

        # Make sure the intial velocity approximation does not have too much of an effect on the integration
        # by regularizing with the finite difference velocity
        if frame == 1:
            velocity_next = (initial_velocity + velocity_next) / 2

        Pt_tot[frame + 1, :, :] = Pt_integ
        Velocity_tot[frame + 1, :, :] = velocity_next

        # --- Errors ---#
        relative_error.append(integration_error(Pt_integ, Pt_interpoles[frame + 1])[0])
        absolute_error.append(integration_error(Pt_integ, Pt_interpoles[frame + 1])[1])

    return Pt_tot, erreur_relative, erreur_absolue, F_all_point, v_all


def multiple_shooting_integration(nb_frame, Pt_interpoles, Pt_ancrage_interpoles, dict_fixed_params):
    """
    Computes the position of the points by integrating (from the forces and initial conditions).
    """

    def dyn_fun(t, y, Pt_ancrage_interpoles):
        """
        x = [p, v]
        dx = [v, dv]
        """
        p = y[: m * n * 3].reshape(m * n, 3)
        v = y[m * n * 3 :].reshape(m * n, 3)

        bt1, bt2, btc1, btc2 = spring_bouts_collecte(p.T, Pt_ancrage_interpoles)
        M, F_spring, F_spring_croix, F_masses = static_forces_calc(bt1, bt2, btc1, btc2, dict_fixed_params)
        F_point = static_force_in_each_point(F_spring, F_spring_croix, F_masses)
        accel_current = np.zeros((n * m, 3))
        for i in range(0, n * m):
            accel_current[i, :] = F_point[i, :] / M[i]

        return np.hstack((v.flatten(), accel_current.flatten()))

    # --- Initzalization ---#
    Pt_tot = np.zeros((nb_frame, n * m, 3))
    F_all_point = np.zeros((nb_frame, n * m, 3))
    Velocity_tot = np.zeros((nb_frame, n * m, 3))
    relative_error, absolute_error = [], []

    # Velocity of the second frame by finite difference
    current_velocity = velocity_from_finite_difference(Pt_interpoles, idx_before=0, idx_after=2)
    Velocity_tot[0, :, :] = current_velocity
    Pt_tot[0, :, :] = Pt_interpoles[0]

    for frame in range(nb_frame - 1):
        # --- At the current frame ---#
        if frame == 0:
            current_velocity = (
                velocity_from_finite_difference(Pt_interpoles, idx_before=frame, idx_after=frame + 1) * dt
            )
        else:
            current_velocity = velocity_from_finite_difference(Pt_interpoles, idx_before=frame - 1, idx_after=frame + 1)
        Velocity_tot[frame, :, :] = current_velocity

        y0 = np.hstack((Pt_interpoles[frame, :, :].T.flatten(), Velocity_tot[frame, :, :].flatten()))
        sol = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_fun(t, y, Pt_ancrage_interpoles[frame]), t_span=[0, dt], y0=y0, method="DOP853")
        position_diff = sol.y[: m * n * 3, -1].reshape(m * n, 3)
        velocity_diff = sol.y[m * n * 3 :, -1].reshape(m * n, 3)
        velocity_next = current_velocity + velocity_diff
        Pt_integ = Pt_interpoles[frame] + position_diff

        Pt_tot[frame + 1, :, :] = Pt_integ
        Velocity_tot[frame + 1, :, :] = velocity_next

        # --- Errors ---#
        rel, abs = integration_error(Pt_integ, Pt_interpoles[frame + 1])
        relative_error.append(rel)
        absolute_error.append(abs)

    return Pt_tot, Velocity_tot, erreur_relative, erreur_absolue, F_all_point


def Animation(Pt_integres, Pt_collecte_tab, jump_frame_index_interval):
    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-2.2, top=2.2)
    ax.axes.set_zlim3d(bottom=-2.5, top=0.5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    colors_colormap = sns.color_palette(palette="viridis", n_colors=n * m)
    colors = [[] for i in range(n * m)]
    for i in range(n * m):
        col_0 = colors_colormap[i][0]
        col_1 = colors_colormap[i][1]
        col_2 = colors_colormap[i][2]
        colors[i] = (col_0, col_1, col_2)

    ax.set_box_aspect([1.1, 1.8, 1])
    integrated_point = [ax.plot(0, 0, 0, ".", mfc="none", color=colors[i]) for i in range(n * m)]
    marker_point = [ax.plot(0, 0, 0, ".", markersize=5, color=colors[i]) for i in range(Pt_collecte_tab[0].shape[1])]

    nb_frame = jump_frame_index_interval[1] - jump_frame_index_interval[0] - 1
    animate = animation.FuncAnimation(
        fig, update, frames=nb_frame, fargs=(Pt_integres, Pt_collecte_tab, integrated_point, marker_point), blit=False
    )
    output_file_name = "simulation.mp4"

    animate.save(output_file_name, fps=20)
    plt.show()

def plot_results(Pt_ancrage_repos, Pt_repos, Pt_ancrage_collecte, Pt_toile_collecte, ind_masse):
    """
    Plot the position of the markers vs model points
    """

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(0, 0, -1.2, "ow")

    ax.plot(
        Pt_ancrage_repos[:, 0],
        Pt_ancrage_repos[:, 1],
        Pt_ancrage_repos[:, 2],
        ".k",
        mfc="none",
        alpha=0.5,
        label="Model Frame",
    )
    ax.plot(Pt_repos.T[0, :], Pt_repos.T[1, :], Pt_repos.T[2, :], ".b", mfc="none", alpha=0.5, label="Model Trampline")

    ax.plot(
        Pt_ancrage_collecte[:, 0],
        Pt_ancrage_collecte[:, 1],
        Pt_ancrage_collecte[:, 2],
        "ok",
        label="Experimental Frame",
    )
    ax.plot(
        Pt_toile_collecte[0, :], Pt_toile_collecte[1, :], Pt_toile_collecte[2, :], ".b", label="Experimental Trampoline"
    )
    ax.plot(
        Pt_toile_collecte[0, ind_masse],
        Pt_toile_collecte[1, ind_masse],
        Pt_toile_collecte[2, ind_masse],
        ".g",
        label="Lowest point on the interval",
    )

    ax.set_title("Data collection markers vs model points")
    ax.legend()
    plt.show()


###############################################################################
###############################################################################
###############################################################################


def main():
    # SELECTION OF THE RESULTS FROM THE DATA COLLECTION
    participant = 1
    static_trial_name = "labeled_statique_leftfront_D7"
    trial_name = "labeled_p1_sauthaut_01"
    empty_trial_name = "labeled_statique_centrefront_vide"
    jump_frame_index_interval = [
        7000,
        7170,
    ]  # This range repends on the trial. To find it, one should use the code plateforme_verification_toutesversions.py.
    dt = 1 / 500  # Hz

    nb_frame = jump_frame_index_interval[1] - jump_frame_index_interval[0] - 1  # The first frame is excluded
    dict_fixed_params = spring_lengths()

    # Récupération de tous les points des frames de l'intervalle dynamics
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(
        participant, empty_trial_name, trial_name, jump_frame_index_interval
    )

    # Récupération des parametres du problemes
    k, k_oblique, M, C = Param()
    Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)

    Pt_ancrage_collecte, labels_ancrage = Point_ancrage(Pt_collecte_tab, labels)
    Pt_toile_collecte, label_toile = Point_toile_init(Pt_collecte_tab, labels)

    Pt_interpoles, Pt_ancrage_interpolated = surface_interpolation_collecte(
        Pt_toile_collecte, Pt_ancrage_collecte, Pt_repos, Pt_ancrage_repos, dict_fixed_params, with_plot=True
    )

    # Pt_integres, erreur_relative, erreur_absolue, static_force_in_each_points, v_all = multiple_shooting_euler_integration(nb_frame, Pt_interpoles, labels, Masse_centre)
    Pt_integres, erreur_relative, erreur_absolue, static_force_in_each_points, v_all = multiple_shooting_integration(
        nb_frame, Pt_interpoles, Pt_ancrage_interpolated, dict_fixed_params
    )

    plot_results(Pt_ancrage_repos, Pt_repos, Pt_ancrage_collecte, Pt_toile_collecte, ind_masse)

    # Animation of the markers vs integrated marker position
    Animation(Pt_integres, Pt_collecte_tab, jump_frame_index_interval)

    # Plot the model and springs at the initial instant
    Affichage_points_collecte_t(Pt_repos.T, Pt_ancrage_repos, True, nb_frame, ind_masse)
    plt.show()

    all_F_totale_collecte, all_Pt_collecte_tab, all_labels, all_ind_masse = Resultat_PF_collecte(
        participant, empty_trial_name, trial_name, [0, 7763]
    )
    all_Pt_ancrage_collecte, label_ancrage = Point_ancrage(all_Pt_collecte_tab, all_labels)

    # Plot the exeprimental forces vs model forces
    time = np.linspace(0, len(all_Pt_ancrage_collecte) * dt, len(all_Pt_ancrage_collecte) + 1)
    z = [np.array([0]) for i_marker in range(all_Pt_ancrage_collecte[0].shape[0])]
    z_filtered = [np.array([0]) for i_marker in range(all_Pt_ancrage_collecte[0].shape[0])]
    az = [np.array([0]) for i_marker in range(all_Pt_ancrage_collecte[0].shape[0])]
    times_without_nans = [np.array([0]) for i_marker in range(all_Pt_ancrage_collecte[0].shape[0])]
    for i_marker in range(all_Pt_ancrage_collecte[0].shape[0]):
        first_non_nan_hit = False
        for i_frame in range(len(all_Pt_ancrage_collecte)):
            if np.isnan(all_Pt_ancrage_collecte[i_frame][i_marker, 2]):
                if not first_non_nan_hit:
                    continue
                else:
                    z[i_marker] = np.vstack((z[i_marker], z[i_marker][-1]))
            else:
                z[i_marker] = np.vstack((z[i_marker], all_Pt_ancrage_collecte[i_frame][i_marker, 2]))
                first_non_nan_hit = True
                times_without_nans[i_marker] = np.vstack((times_without_nans[i_marker], time[i_frame]))
        z[i_marker] = z[i_marker][1:]
        times_without_nans[i_marker] = times_without_nans[i_marker][1:]

        a, b = scipy.signal.butter(4, 0.015)
        z_filtered[i_marker] = scipy.signal.filtfilt(a, b, z[i_marker], method="gust")
        az[i_marker] = (z_filtered[i_marker][2:] + z_filtered[i_marker][:-2] - 2 * z_filtered[i_marker][1:-1]) / (
            dt * dt
        )

    fig, ax = plt.subplots(2, 1)
    for i_marker in range(len(z)):
        if z[i_marker].shape != (0,):
            ax[0].plot(time[:-1], z[i_marker])
            ax[0].plot(time[:-1], z_filtered[i_marker], "-r")
            ax[1].plot(time[1:-2], az[i_marker])
    fig.suptitle("Frame markers")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Z [m]")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel(r"Acceleration [$m/s^-2$]")
    plt.savefig("../Plots/frame_markers.png")
    plt.show()

    # TODO: Plot the forces from the model vs force plates
    masse_cadre = 270


if __name__ == "__main__":
    main()
