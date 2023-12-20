
import numpy as np
import casadi as cas


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


# FONCTIONS AVEC LES PARAMÈTRES FIXES :
def Param_fixe():
    # ESPACES ENTRE LES MARQUEURS :

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

    ####################################################################################################################

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

    # dans la toile : on dit que les longueurs au repos sont les memes que en pretension
    # ressorts horizontaux internes a la toile :
    l_horz = np.array([dl[j] * np.ones(n) for j in range(1, m)])
    l_horz = np.reshape(l_horz, Nb_ressorts_horz)
    l_repos = np.append(l_repos, l_horz)

    # ressorts verticaux internes a la toile :
    l_vert = np.array([dL[j] * np.ones(m) for j in range(1, n)])
    l_vert = np.reshape(l_vert, Nb_ressorts_vert)
    l_repos = np.append(l_repos, l_vert)

    # # ressorts obliques internes a la toile :
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

    return dict_fixed_params  # np

def Param_variable(k_type, WITH_K_OBLIQUE):
    """
    Répartir les raideurs des ressorts à partir des différents types de ressorts
    :param k_type: cas.MX(12): les 12 types de ressorts qu'on retrouve dans la toile (8 structurels, 4 cisaillement)
    :return: k: cas.MX(Nb_ressorts): ensemble des raideurs des ressorts non obliques, dont ressorts du cadre
    :return: k_oblique: cas.MX(Nb_ressorts_croix): ensemble des raideurs des ressorts obliques
    """

    if type(k_type) == cas.MX:
        zero_fcn = cas.MX.zeros
        ones_fcn = cas.MX.ones
    elif type(k_type) == cas.DM:
        zero_fcn = cas.DM.zeros
        ones_fcn = cas.DM.ones
    elif type(k_type) == np.ndarray:
        zero_fcn = np.zeros
        ones_fcn = np.ones

    # RAIDEURS A CHANGER
    k1 = k_type[0]  # un type de coin (ressort horizontal)
    k2 = k_type[1]  # ressorts horizontaux du bord (bord vertical)
    k3 = k_type[2]  # un type de coin (ressort vertical)
    k4 = k_type[3]  # ressorts verticaux du bord (bord horizontal)
    k5 = k_type[4]  # ressorts horizontaux du bord horizontal de la toile
    k6 = k_type[5]  # ressorts horizontaux
    k7 = k_type[6]  # ressorts verticaux du bord vertical de la toile
    k8 = k_type[7]  # ressorts verticaux
    if WITH_K_OBLIQUE:
        k_oblique_1 = k_type[8]  # 4 ressorts des coins
        k_oblique_2 = k_type[9]  # ressorts des bords verticaux
        k_oblique_3 = k_type[10]  # ressorts des bords horizontaux
        k_oblique_4 = k_type[11]  # ressorts quelconques

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    k_bord = zero_fcn((Nb_ressorts_cadre, 1))
    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m: 2 * n + m] = k2, k2
    # cotes horizontaux :
    k_bord[n: n + m], k_bord[2 * n + m: 2 * n + 2 * m] = k4, k4
    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    # ressorts horizontaux dans la toile
    k_horizontaux = k6 * ones_fcn((n * (m - 1), 1))
    k_horizontaux[0: n * (m - 1): n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1: n * (m - 1): n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    k_verticaux = k8 * ones_fcn((m * (n - 1), 1))
    k_verticaux[0: m * (n - 1): m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1: n * m - m: m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche

    if type(k_type) == np.ndarray:
        k = np.vstack((k_horizontaux, k_verticaux))
        k = np.vstack((k_bord, k))
    else:
        k = cas.vertcat(k_horizontaux, k_verticaux)
        k = cas.vertcat(k_bord, k)

    ######################################################################################################################

    # RESSORTS OBLIQUES
    # milieux :
    if WITH_K_OBLIQUE:
        k_oblique = zero_fcn((Nb_ressorts_croix, 1))

        # coins :
        k_oblique[0], k_oblique[1] = k_oblique_1, k_oblique_1  # en bas a droite
        k_oblique[2 * (n - 1) - 1], k_oblique[2 * (n - 1) - 2] = k_oblique_1, k_oblique_1  # en haut a droite
        k_oblique[Nb_ressorts_croix - 1], k_oblique[
            Nb_ressorts_croix - 2] = k_oblique_1, k_oblique_1  # en haut a gauche
        k_oblique[2 * (n - 1) * (m - 2)], k_oblique[
            2 * (n - 1) * (m - 2) + 1] = k_oblique_1, k_oblique_1  # en bas a gauche

        # côtés verticaux :
        k_oblique[2: 2 * (n - 1) - 2] = k_oblique_2  # côté droit
        k_oblique[2 * (n - 1) * (m - 2) + 2: Nb_ressorts_croix - 2] = k_oblique_2  # côté gauche

        # côtés horizontaux :
        k_oblique[28:169:28], k_oblique[29:170:28] = k_oblique_3, k_oblique_3  # en bas
        k_oblique[55:196:28], k_oblique[54:195:28] = k_oblique_3, k_oblique_3  # en haut

        # milieu :
        for j in range(1, 7):
            k_oblique[2 + 2 * j * (n - 1): 26 + 2 * j * (n - 1)] = k_oblique_4
    else:
        k_oblique = None

    return k, k_oblique


def Param_variable_masse(ind_masse, Ma):

    if type(Ma) == cas.MX:
        ones_fcn = cas.MX.ones
    elif type(Ma) == cas.DM:
        ones_fcn = cas.DM.ones
    elif type(Ma) == np.ndarray:
        ones_fcn = np.ones

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

    M = mmilieu * ones_fcn(n * m)  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n : n * (m - 1) : n] = mpetitbord  # masses du cote bas
    M[2 * n - 1 : n * m - 1 : n] = mpetitbord  # masses du cote haut
    M[1 : n - 1] = mgrandbord  # masse du cote droit
    M[n * (m - 1) + 1 : n * m - 1] = mgrandbord  # masse du cote gauche

    M[ind_masse] += Ma[0]
    M[ind_masse + 1] += Ma[1]
    M[ind_masse - 1] += Ma[2]
    M[ind_masse + 15] += Ma[3]
    M[ind_masse - 15] += Ma[4]

    return M


def Calcul_Pt_F(X, Pt_ancrage, dict_fixed_params, K, ind_masse, Ma, WITH_K_OBLIQUE, NO_COMPRESSION=False):

    if type(X) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(X) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(X) == np.ndarray:
        zero_fcn = np.zeros

    k, k_croix = Param_variable(K, WITH_K_OBLIQUE)
    M = Param_variable_masse(ind_masse, Ma)
    Pt = list2tab(X)

    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    F_spring, F_spring_croix, F_masses = Force_calc(
        Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, k, k_croix, M, dict_fixed_params, NO_COMPRESSION
    )

    if not WITH_K_OBLIQUE:
        F_spring_croix = zero_fcn(Spring_bout_croix_1.shape)
    F_point = Force_point(F_spring, F_spring_croix, F_masses)

    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_box_aspect([1.1, 1.8, 1])
    # ax.plot(0, 0, -1.2, "ow")
    #
    # ax.plot(
    #     Spring_bout_1[:, 0],
    #     Spring_bout_1[:, 1],
    #     Spring_bout_1[:, 2],
    #     "ob",
    #     mfc="none",
    #     alpha=0.5,
    #     markersize=3,
    #     label="1",
    # )
    #
    # ax.plot(
    #     Spring_bout_2[:, 0],
    #     Spring_bout_2[:, 1],
    #     Spring_bout_2[:, 2],
    #     ".r",
    #     markersize=3,
    #     label="2",
    # )
    # for i in range(Spring_bout_1.shape[0]):
    #     plt.plot(np.vstack((Spring_bout_1[i, 0], Spring_bout_2[i, 0])),
    #              np.vstack((Spring_bout_1[i, 1], Spring_bout_2[i, 1])),
    #              np.vstack((Spring_bout_1[i, 2], Spring_bout_2[i, 2])),
    #              "-k")
    # ax.legend()
    # plt.show()
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_box_aspect([1.1, 1.8, 1])
    # ax.plot(0, 0, -1.2, "ow")
    #
    # ax.plot(
    #     Spring_bout_croix_1[:, 0],
    #     Spring_bout_croix_1[:, 1],
    #     Spring_bout_croix_1[:, 2],
    #     "ob",
    #     mfc="none",
    #     alpha=0.5,
    #     markersize=3,
    #     label="1",
    # )
    #
    # ax.plot(
    #     Spring_bout_croix_2[:, 0],
    #     Spring_bout_croix_2[:, 1],
    #     Spring_bout_croix_2[:, 2],
    #     ".r",
    #     markersize=3,
    #     label="2",
    # )
    # for i in range(Spring_bout_croix_1.shape[0]):
    #     plt.plot(np.vstack((Spring_bout_croix_1[i, 0], Spring_bout_croix_2[i, 0])),
    #              np.vstack((Spring_bout_croix_1[i, 1], Spring_bout_croix_2[i, 1])),
    #              np.vstack((Spring_bout_croix_1[i, 2], Spring_bout_croix_2[i, 2])),
    #              "-k")
    # ax.legend()
    # plt.show()

    F_totale = zero_fcn((3, 1))
    for ind in range(F_point.shape[0]):
        for i in range(3):
            F_totale[i] += F_point[ind, i]

    return F_totale, F_point



def list2tab(list):
    """
    Transformer un MX de taille 405x1 en MX de taille 135x3
    :param list: MX(n*m*3,1)
    :return: tab: MX(135,3)
    """
    if type(list) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(list) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(list) == np.ndarray:
        zero_fcn = np.zeros

    tab = zero_fcn((135, 3))
    for ind in range(135):
        for i in range(3):
            tab[ind, i] = list[i + 3 * ind]
    return tab


def tab2list(tab):
    if type(tab) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(tab) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(tab) == np.ndarray:
        zero_fcn = np.zeros

    list = zero_fcn((135 * 3))
    for i in range(135):
        for j in range(3):
            list[j + 3 * i] = tab[i, j]
    return list



def rotation_points(Pt_ancrage, Pos_repos):
    """
    Appliquer la rotation pour avoir la même orientation que les points de la collecte
    :param Pos_repos: cas.DM(n*m,3): coordonnées (2D) des points de la toile
    :param Pt_ancrage: cas.DM(2*n+2*m,3): coordonnées des points du cadre
    :return: Pos_repos, Pt_ancrage
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
        )

    Pos_repos_new = np.zeros((n * m, 3))
    for index in range(n * m):
        Pos_repos_new[index, :] = np.matmul(Pos_repos[index, :], mat_base_inv_np)

    return Pt_ancrage_new, Pos_repos_new


def Points_ancrage_repos(dict_fixed_params):
    """
    :param dict_fixed_params: dictionnaire contenant les paramètres fixés
    :return: Pos_repos: cas.DM(n*m,3): coordonnées (2D) des points de la toile
    :return: Pt_ancrage: cas.DM(2*n+2*m,3): coordonnées des points du cadre
    """
    dL = dict_fixed_params["dL"]
    dl = dict_fixed_params["dl"]
    l_droite = dict_fixed_params["l_droite"]
    l_gauche = dict_fixed_params["l_gauche"]
    L_haut = dict_fixed_params["L_haut"]
    L_bas = dict_fixed_params["L_bas"]

    # repos :
    Pos_repos = np.zeros((n * m, 3))

    # on dit que le point numero 0 est a l'origine
    for j in range(m):
        for i in range(n):
            # Pos_repos[i + j * n] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])
            Pos_repos[i + j * n, :] = np.array([-np.sum(dl[: j + 1]), np.sum(dL[: i + 1]), 0])

    Pos_repos_new = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pos_repos_new[i + j * n, :] = Pos_repos[i + j * n, :] - Pos_repos[67, :]
    # Pos_repos_new = np.copy(Pos_repos)

    # ancrage :
    Pt_ancrage = np.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pos_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = np.array([0, L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = np.array([0, L_haut, 0]) + np.array([np.sum(dl[1 + j - n : 5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = np.array([0, L_haut, 0]) - np.array([np.sum(dl[5 : j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[k, 1:2] = -Pos_repos_new[k - n - m, 1:2]
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

    Pt_ancrage, Pos_repos_new = rotation_points(Pos_repos_new, Pt_ancrage)

    Pt_ancrage_cas = cas.DM(Pt_ancrage)
    Pos_repos_cas = cas.DM(Pos_repos_new)
    return Pt_ancrage_cas, Pos_repos_cas  # np


def Spring_bouts(Pt, Pt_ancrage):
    """
    Returns the coordinates of the two ends of each spring horizontal and vertical (no diagonal).
    The first end is the
    Pt: points on the trampoline bed (n*m,3)
    Pt_ancrage: points on the frame (2*n+2*m,3)
    """
    if type(Pt) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(Pt) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(Pt) == np.ndarray:
        zero_fcn = np.zeros

    # Definition des ressorts (position, taille)
    Spring_bout_1 = zero_fcn((Nb_ressorts, 3))

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
    Spring_bout_2 = zero_fcn((Nb_ressorts, 3))

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

    return (Spring_bout_1, Spring_bout_2)


def Spring_bouts_croix(Pt):  # sx
    """

    :param Pt: cas.MX(n*m,3): coordonnées des n*m points de la toile

    :return: Spring_bout_croix_1: cas.MX((Nb_ressorts_croix, 3)): bout 1 de chaque ressort oblique
    :return: Spring_bout_croix_2: cas.MX((Nb_ressorts_croix, 3)): bout 2 de chaque ressort oblique
    """

    if type(Pt) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(Pt) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(Pt) == np.ndarray:
        zero_fcn = np.zeros

    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = zero_fcn((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pt[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pt[i, :]
            k += 1

    Spring_bout_croix_2 = zero_fcn((Nb_ressorts_croix, 3))
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


def Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, k, k_oblique, M, dict_fixed_params, NO_COMPRESSION=False
):  # force dans chaque ressort
    """

    :param Spring_bout_1: cas.MX((Nb_ressorts, 3)): bout 1 de chaque ressort non oblique
    :param Spring_bout_2: cas.MX((Nb_ressorts, 3)): bout 2 de chaque ressort non oblique
    :param Spring_bout_croix_1: cas.MX((Nb_ressorts_croix, 3)): bout 1 de chaque ressort oblique
    :param Spring_bout_croix_2: cas.MX((Nb_ressorts_croix, 3)): bout 2 de chaque ressort oblique
    :param k: cas.MX(Nb_ressorts): raideurs de tous les ressorts non obliques
    :param k_oblique: cas.MX(Nb_ressorts_croix): raideurs de tous les ressorts obliques
    :param dict_fixed_params: dictionnaire contenant les paramètres fixes

    :return: F_spring: cas.MX(Nb_ressorts, 3): force élastique de chaque ressort non oblique (dont ressorts du cadre)
    :return: F_spring_croix: cas.MX(Nb_ressorts_croix, 3): force élastique de chaque ressort oblique
    :return: F_masses: cas.MX(n*m,3): force de gravité appliquée à chaque point
    """

    if type(Spring_bout_1) == cas.MX:
        zero_fcn = cas.MX.zeros
        norm_fcn = cas.norm_fro
    elif type(Spring_bout_1) == cas.DM:
        zero_fcn = cas.DM.zeros
        norm_fcn = cas.norm_fro
    elif type(Spring_bout_1) == np.ndarray:
        zero_fcn = np.zeros
        norm_fcn = np.linalg.norm

    l_repos = dict_fixed_params["l_repos"]
    l_repos_croix = dict_fixed_params["l_repos_croix"]

    F_spring = zero_fcn((Nb_ressorts, 3))
    Vect_unit_dir_F = zero_fcn((Nb_ressorts, 3))
    for ispring in range(Nb_ressorts):
        vect = (Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) / norm_fcn(
            Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]
        )
        for i in range(3):
            Vect_unit_dir_F[ispring, i] = vect[i]
        elongation = norm_fcn(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring]
        vect = Vect_unit_dir_F[ispring, :] * k[ispring] * elongation
        for i in range(3):
            if NO_COMPRESSION:
                if type(Spring_bout_1) == np.ndarray:
                    if elongation > 0:
                        F_spring[ispring, i] = vect[i]
                    else:
                        F_spring[ispring, i] = 0
                else:
                    F_spring[ispring, :] = cas.if_else(elongation > 0,  # Condition
                                                   Vect_unit_dir_F[ispring, :] * k[ispring] * elongation,  # if
                                                   cas.MX.zeros(1, 3)  # else
                                                   )
            else:
                F_spring[ispring, i] = vect[i]

    if k_oblique is not None:
        F_spring_croix = zero_fcn((Nb_ressorts_croix, 3))
        Vect_unit_dir_F_croix = zero_fcn((Nb_ressorts, 3))
        for ispring in range(Nb_ressorts_croix):
            vect = (Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) / norm_fcn(
                Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]
            )
            for i in range(3):
                Vect_unit_dir_F_croix[ispring, i] = vect[i]
            elongation_croix = norm_fcn(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) - l_repos_croix[ispring]
            vect = Vect_unit_dir_F_croix[ispring, :] * k_oblique[ispring] * elongation_croix
            for i in range(3):
                if NO_COMPRESSION:
                    if type(Spring_bout_1) == np.ndarray:
                        if elongation_croix > 0:
                            F_spring_croix[ispring, i] = vect[i]
                        else:
                            F_spring_croix[ispring, i] = 0
                    else:
                        F_spring_croix[ispring, :] = cas.if_else(elongation_croix > 0,  # Condition
                                                       Vect_unit_dir_F_croix[ispring, :] * k_oblique[ispring] * elongation_croix,  # if
                                                       cas.MX.zeros(1, 3)  # else
                                                       )
                else:
                    F_spring_croix[ispring, i] = vect[i]
    else:
        F_spring_croix = None

    F_masses = zero_fcn((n * m, 3))
    F_masses[:, 2] = -M * 9.81

    return F_spring, F_spring_croix, F_masses


def Force_point(F_spring, F_spring_croix, F_masses):  # --> resultante des forces en chaque point a un instant donne
    """

    :param F_spring: cas.MX(Nb_ressorts, 3): force élastique de chaque ressort non oblique (dont ressorts du cadre)
    :param F_spring_croix: cas.MX(Nb_ressorts_croix, 3): force élastique de chaque ressort oblique
    :param F_masses: cas.MX(n*m,3): force de gravité appliquée à chaque point

    :return: F_point: cas.MX(n*m,3): résultantes des forces en chaque point
    """

    if type(F_spring) == cas.MX:
        zero_fcn = cas.MX.zeros
    elif type(F_spring) == cas.DM:
        zero_fcn = cas.DM.zeros
    elif type(F_spring) == np.ndarray:
        zero_fcn = np.zeros

    # forces elastiques
    F_spring_points = zero_fcn((n * m, 3))

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


