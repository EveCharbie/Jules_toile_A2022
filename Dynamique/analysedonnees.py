import numpy as np
import matplotlib.pyplot as plt
import pickle
from ezc3d import c3d
from scipy import signal
import casadi as cas

n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


#FONCTIONS AVEC LES PARAMÈTRES FIXES :

def Param_fixe(Masse_centre):
    #ESPACES EN TRE LES MARQUEURS :

    # de bas en haut :
    dL = np.array([510.71703748, 261.87522103, 293.42186099, 298.42486747, 300.67352585,
                   298.88879749, 299.6946861, 300.4158083, 304.52115312, 297.72780618,
                   300.53723415, 298.27144226, 298.42486747, 293.42186099, 261.87522103,
                   510.71703748]) * 0.001  # ecart entre les lignes, de bas en haut
    dL = np.array([np.mean([dL[i], dL[-(i + 1)]]) for i in range(16)])

    dLmilieu = np.array([151.21983556, 153.50844775]) * 0.001
    dLmilieu = np.array([np.mean([dLmilieu[i], dLmilieu[-(i + 1)]]) for i in range(2)])
    # de droite a gauche :
    dl = np.array(
        [494.38703513, 208.96708367, 265.1669672, 254.56358938, 267.84760997, 268.60351809, 253.26974254,
         267.02823864,
         208.07894712, 501.49013437]) * 0.001
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
        Nb_ressorts_cadre)  # on fera des append plus tard, l_repos prend bien en compte tous les ressorts non obliques

    # entre la toile et le cadre :
    # ecart entre les marqueurs - taille du ressort en pretension + taille ressort hors trampo
    l_bord_horz = dl[0] - 0.388 + 0.264
    l_bord_vert = dL[0] - 0.388 + 0.264
    l_repos[0:n], l_repos[n + m:2 * n + m] = l_bord_horz, l_bord_horz
    l_repos[n:n + m], l_repos[2 * n + m:2 * n + 2 * m] = l_bord_vert, l_bord_vert

    l_bord_coin = np.mean([l_bord_vert, l_bord_horz])  # pas sure !!!
    l_repos[0], l_repos[n - 1], l_repos[n + m], l_repos[
        2 * n + m - 1] = l_bord_coin, l_bord_coin, l_bord_coin, l_bord_coin
    l_repos[n], l_repos[n + m - 1], l_repos[2 * n + m], l_repos[
        2 * (n + m) - 1] = l_bord_coin, l_bord_coin, l_bord_coin, l_bord_coin

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
        l_repos_croixj[0:n - 1] = (l_vert[0:m * (n - 1):m] ** 2 + l_horz[j * n] ** 2) ** 0.5
        l_repos_croix = np.append(l_repos_croix, l_repos_croixj)
    # dans chaque maille il y a deux ressorts obliques :
    l_repos_croix_double = np.zeros((int(Nb_ressorts_croix / 2), 2))
    for i in range(int(Nb_ressorts_croix / 2)):
        l_repos_croix_double[i] = [l_repos_croix[i], l_repos_croix[i]]
    l_repos_croix = np.reshape(l_repos_croix_double, Nb_ressorts_croix)


    ##################################################################################################################
    # MASSES (pris en compte la masse ajoutee par lathlete) :
    Mtrampo = 5.00
    mressort_bord = 0.322
    mressort_coin = 0.553
    mattache = 0.025  # attache metallique entre trampo et ressort

    mmilieu = Mtrampo / ((n - 1) * (m - 1))  # masse d'un point se trouvant au milieu de la toile
    mpetitbord = 0.5 * (Mtrampo / ((n - 1) * (m - 1))) + 2 * (
            (mressort_bord / 2) + mattache)  # masse d un point situé sur le petit bord
    mgrandbord = 0.5 * (Mtrampo / ((n - 1) * (m - 1))) + (33 / 13) * (
            (mressort_bord / 2) + mattache)  # masse d un point situé sur le grand bord
    mcoin = 0.25 * (Mtrampo / ((n - 1) * (m - 1))) + mressort_coin + 4 * (
            (mressort_bord / 2) + mattache)  # masse d un point situé dans un coin

    M = mmilieu * np.ones((n * m))  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n:n * (m - 1):n] = mpetitbord  # masses du cote bas
    M[2 * n - 1:n * m - 1:n] = mpetitbord  # masses du cote haut
    M[1:n - 1] = mgrandbord  # masse du cote droit
    M[n * (m - 1) + 1:n * m - 1] = mgrandbord  # masse du cote gauche

    # k multistart

    k1 = 1.21175669e+05
    k2 = 3.20423906e+03
    k3 = 4.11963416e+03
    k4 = 2.48125477e+03
    k5 = 7.56820743e+03
    k6 = 4.95811865e+05
    k7 = 1.30776275e-03
    k8 = 3.23131678e+05
    k1ob = 7.48735556e+02
    k2ob = 1.08944449e-04
    k3ob = 3.89409909e+03
    k4ob = 1.04226031e-04

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    k_bord = np.zeros(Nb_ressorts_cadre)

    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m:2 * n + m] = k2, k2

    # cotes horizontaux :
    k_bord[n:n + m], k_bord[2 * n + m:2 * n + 2 * m] = k4, k4

    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    # ressorts horizontaux dans la toile
    k_horizontaux = k6 * np.ones(n * (m - 1))
    k_horizontaux[0:n * m - 1:n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1:n * (m - 1):n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    k_verticaux = k8 * np.ones(m * (n - 1))
    k_verticaux[0:m * (n - 1):m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1:n * m - 1:m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche

    k = np.append(k_horizontaux, k_verticaux)
    k = np.append(k_bord, k)

    # ressorts obliques dans la toile
    k_oblique = np.ones(Nb_ressorts_croix)
    k_oblique[0], k_oblique[1] = k1ob, k1ob  # en bas a droite
    k_oblique[2 * (n - 1) - 1], k_oblique[2 * (n - 1) - 2] = k1ob, k1ob  # en haut a droite
    k_oblique[Nb_ressorts_croix - 1], k_oblique[Nb_ressorts_croix - 2] = k1ob, k1ob  # en haut a gauche
    k_oblique[2 * (n - 1) * (m - 2)], k_oblique[2 * (n - 1) * (m - 2) + 1] = k1ob, k1ob  # en bas a gauche

    # côtés verticaux :
    k_oblique[2: 2 * (n - 1) - 2] = k2ob  # côté droit
    k_oblique[2 * (n - 1) * (m - 2) + 2: Nb_ressorts_croix - 2] = k2ob  # côté gauche

    # côtés horizontaux :
    k_oblique[28:169:28], k_oblique[29:170:28] = k3ob, k3ob  # en bas
    k_oblique[55:196:28], k_oblique[54:195:28] = k3ob, k3ob  # en haut

    # milieu :
    for j in range(1, 7):
        k_oblique[2 + 2 * j * (n - 1): 26 + 2 * j * (n - 1)] = k4ob

    dict_fixed_params = {'dL': dL,
                         'dLmilieu': dLmilieu,
                         'dl': dl,
                         'dlmilieu': dlmilieu,
                         'l_droite': l_droite,
                         'l_gauche': l_gauche,
                         'L_haut': L_haut,
                         'L_bas': L_bas,
                         'l_repos': l_repos,
                         'l_repos_croix': l_repos_croix,
                         'Masse_centre': Masse_centre,
                         'M': M,
                         'k': k,
                         'k_ob': k_oblique}

    return dict_fixed_params

def Points_ancrage_fix(dict_fixed_params):

    dL = dict_fixed_params['dL']
    dl = dict_fixed_params['dl']
    l_droite = dict_fixed_params['l_droite']
    l_gauche = dict_fixed_params['l_gauche']
    L_haut = dict_fixed_params['L_haut']
    L_bas = dict_fixed_params['L_bas']

    # repos :
    Pos_repos = np.zeros((n * m, 3))

    # on dit que le point numero 0 est a l'origine
    for j in range(m):
        for i in range(n):
            # Pos_repos[i + j * n] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])
            Pos_repos[i + j * n,:] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])

    Pos_repos_new = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pos_repos_new[i + j * n,:] = Pos_repos[i + j * n,:] - Pos_repos[67,:]


    # ancrage :
    Pt_ancrage = np.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pos_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = np.array([0,L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = np.array([0,L_haut, 0]) + np.array([np.sum(dl[1 + j - n:5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = np.array([0,L_haut, 0]) - np.array([np.sum(dl[5: j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[k, 1:2] = - Pos_repos_new[k - n - m, 1:2]
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

    Pos_repos_new, Pt_ancrage = rotation_points(Pos_repos_new,Pt_ancrage)

    # fig = plt.figure(0)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1.1, 1.8, 1])
    # ax.plot(Pos_repos_new[:,0], Pos_repos_new[:,1], Pos_repos_new[:,2], 'ob')
    # ax.plot(Pt_ancrage[:, 0], Pt_ancrage[:, 1], Pt_ancrage[:, 2], 'ok')
    # ax.plot(0, 0, -1.2, 'ow')

    # bout1, bout2, boutc1, boutc2 = Bouts_ressorts_repos(Pt_ancrage,Pos_repos_new)
    #
    # for j in range(Nb_ressorts):
    #     a = []
    #     a = np.append(a, bout1[j, 0])
    #     a = np.append(a, bout2[j, 0])
    #
    #     b = []
    #     b = np.append(b, bout1[j, 1])
    #     b = np.append(b, bout2[j, 1])
    #
    #     c = []
    #     c = np.append(c, bout1[j, 2])
    #     c = np.append(c, bout2[j, 2])
    #
    #     ax.plot3D(a, b, c, '-r', linewidth=1)
    #
    # for j in range(Nb_ressorts_croix):
    #     # pas tres elegant mais cest le seul moyen pour que ca fonctionne
    #     a = []
    #     a = np.append(a, boutc1[j, 0])
    #     a = np.append(a, boutc2[j, 0])
    #
    #     b = []
    #     b = np.append(b, boutc1[j, 1])
    #     b = np.append(b, boutc2[j, 1])
    #
    #     c = []
    #     c = np.append(c, boutc1[j, 2])
    #     c = np.append(c, boutc2[j, 2])
    #
    #     ax.plot3D(a, b, c, '-g', linewidth=1)
    #
    #
    # plt.show()

    return Pt_ancrage

def Points_ancrage_repos(dict_fixed_params):

    dL = dict_fixed_params['dL']
    dl = dict_fixed_params['dl']
    l_droite = dict_fixed_params['l_droite']
    l_gauche = dict_fixed_params['l_gauche']
    L_haut = dict_fixed_params['L_haut']
    L_bas = dict_fixed_params['L_bas']

    # repos :
    Pos_repos = np.zeros((n * m, 3))

    # on dit que le point numero 0 est a l'origine
    for j in range(m):
        for i in range(n):
            # Pos_repos[i + j * n] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])
            Pos_repos[i + j * n,:] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])

    Pos_repos_new = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pos_repos_new[i + j * n,:] = Pos_repos[i + j * n,:] - Pos_repos[67,:]


    # ancrage :
    Pt_ancrage = np.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pos_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = np.array([0,L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = np.array([0,L_haut, 0]) + np.array([np.sum(dl[1 + j - n:5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = np.array([0,L_haut, 0]) - np.array([np.sum(dl[5: j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[k, 1:2] = - Pos_repos_new[k - n - m, 1:2]
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

    Pos_repos_new, Pt_ancrage = rotation_points(Pos_repos_new,Pt_ancrage)

    return Pt_ancrage, Pos_repos_new

def Spring_bouts_repos(Pos_repos, Pt_ancrage):
    # Definition des ressorts (position, taille)
    Spring_bout_1 = np.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[i, :] = Pt_ancrage[i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[Nb_ressorts_cadre + i, :] = Pos_repos[i, :]

    # RESSORTS VERTICAUX : il y en a m*(n
    k = 0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pos_repos[i + n * j, :]
            k += 1
    ####################################################################################################################
    Spring_bout_2 = np.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n):  # points droite du bord de la toile
        Spring_bout_2[i, :] = Pos_repos[i, :]

    k = 0
    for i in range(n - 1, m * n, n):  # points hauts du bord de la toile
        Spring_bout_2[n + k, :] = Pos_repos[i, :]
        k += 1

    k = 0
    for i in range(m * n - 1, n * (m - 1) - 1, -1):  # points gauche du bord de la toile
        Spring_bout_2[n + m + k, :] = Pos_repos[i, :]
        k += 1

    k = 0
    for i in range(n * (m - 1), -1, -n):  # points bas du bord de la toile
        Spring_bout_2[2 * n + m + k, :] = Pos_repos[i, :]
        k += 1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k = 0
    for i in range(n, n * m):
        Spring_bout_2[Nb_ressorts_cadre + k, :] = Pos_repos[i, :]
        k += 1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pos_repos[i + n * j, :]
            k += 1

    return Spring_bout_1, Spring_bout_2

def Spring_bouts_croix_repos(Pos_repos):
    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = np.zeros((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pos_repos[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pos_repos[i, :]
            k += 1

    Spring_bout_croix_2 = np.zeros((Nb_ressorts_croix, 3))
    # Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    # pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    # cetait un peu complique mais ca marche, faut pas le changer
    j = 1
    k = 0
    while j < m:
        for i in range(j * n, (j + 1) * n - 2, 2):
            Spring_bout_croix_2[k, :] = Pos_repos[i + 1, :]
            Spring_bout_croix_2[k + 1, :] = Pos_repos[i, :]
            Spring_bout_croix_2[k + 2, :] = Pos_repos[i + 2, :]
            Spring_bout_croix_2[k + 3, :] = Pos_repos[i + 1, :]
            k += 4
        j += 1

    return Spring_bout_croix_1, Spring_bout_croix_2


###################################################
# --- FONCTIONS AVEC LES PARAMÈTRES VARIABLES --- #
###################################################

def Param_variable(C_symetrie):
    #COEFFICIENTS D'AMORTISSEMENT : la toile est séparéee en 4 quarts dont les C sont les mêmes par symétrie avec le centre
    # C_symetrie = 0.3*np.ones(5*8)
    C=np.zeros(n*m)

    # coin en bas a droite de la toile : c'est le coin qui définit tous les autres
    C[0:8] = C_symetrie[0:8].T
    C[15:23] = C_symetrie[8:16].T
    C[30:38] = C_symetrie[16:24].T
    C[45:53] = C_symetrie[24:32].T
    C[60:68] = C_symetrie[32:40].T

    # coin en bas a gauche de la toile :
    C[75:83] = C_symetrie[24:32].T
    C[90:98] = C_symetrie[16:24].T
    C[105:113] = C_symetrie[8:16].T
    C[120:128] = C_symetrie[0:8].T

    #coin en haut a droite de la toile :
    C[14:7:-1] = C_symetrie[0:7].T
    C[29:22:-1] = C_symetrie[8:15].T
    C[44:37:-1] = C_symetrie[16:23].T
    C[59:52:-1] = C_symetrie[24:31].T
    C[74:67:-1] = C_symetrie[32:39].T

    # coin en haut a gauche de la toile :
    C[89:82:-1] = C_symetrie[24:31].T
    C[104:97:-1] = C_symetrie[16:23].T
    C[119:112:-1] = C_symetrie[8:15].T
    C[134:127:-1] = C_symetrie[0:7].T

    return C  #  sx

def Param_variable_force(F):
    F_tab = np.zeros((5,3))
    F_tab[0, :] = F[:3].T
    F_tab[1, :] = F[3:6].T
    F_tab[1, :] = F[6:9].T
    F_tab[1, :] = F[9:12].T
    F_tab[1, :] = F[12:15].T
    return F_tab

def Spring_bouts(Pt, Pt_ancrage): #sx
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

def Spring_bouts_croix(Pt): #sx
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

def Bouts_ressorts_repos(Pt_ancrage, Pos_repos):
    Spring_bout_1, Spring_bout_2 = Spring_bouts_repos(Pos_repos, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix_repos(Pos_repos)
    return Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2

def Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params):  # force dans chaque ressort

    M = dict_fixed_params['M']
    l_repos = dict_fixed_params['l_repos']
    l_repos_croix = dict_fixed_params['l_repos_croix']
    k = dict_fixed_params['k']
    k_oblique = dict_fixed_params['k_ob']

    F_spring = np.zeros((Nb_ressorts, 3))
    Vect_unit_dir_F = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts):
        Vect_unit_dir_F[i, :] = (Spring_bout_2[i, :] - Spring_bout_1[i, :]) / np.linalg.norm(
            Spring_bout_2[i, :] - Spring_bout_1[i, :])

    for ispring in range(Nb_ressorts):
        F_spring[ispring, :] = Vect_unit_dir_F[ispring, :] * k[ispring] * (
                np.linalg.norm(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring])

    # F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    Vect_unit_dir_F_croix = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts_croix):
        Vect_unit_dir_F_croix[i, :] = (Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :]) / np.linalg.norm(
            Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :])

    for ispring in range(Nb_ressorts_croix):
        F_spring_croix[ispring, :] = Vect_unit_dir_F_croix[ispring, :] * k_oblique[ispring] * (
                np.linalg.norm(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) - l_repos_croix[
            ispring])

    F_masses = np.zeros((n * m, 3))
    F_masses[:, 2] = - M * 9.81

    return F_spring, F_spring_croix, F_masses

def Force_amortissement(Xdot, C):

    C = Param_variable(C)
    F_amortissement = np.zeros((n*m, 3))
    for point in range (n*m):
            F_amortissement[point, 2] = - C[point] * Xdot[point, 2]**2   # turbulent = amortissement quadratique ????

    return F_amortissement

def Force_point(F_spring, F_spring_croix, F_masses, F_amortissement):  # --> resultante des forces en chaque point a un instant donne

    # forces elastiques
    F_spring_points = np.zeros((n * m, 3))

    # - points des coin de la toile : VERIFIE CEST OK
    F_spring_points[0, :] = F_spring[0, :] + \
                            F_spring[Nb_ressorts_cadre - 1, :] - \
                            F_spring[Nb_ressorts_cadre, :] - \
                            F_spring[Nb_ressorts_cadre + Nb_ressorts_horz, :] - \
                            F_spring_croix[0,:]  # en bas a droite : premier ressort du cadre + dernier ressort du cadre + premiers ressorts horz, vert et croix
    F_spring_points[n - 1, :] = F_spring[n - 1, :] + \
                                F_spring[n, :] - \
                                F_spring[Nb_ressorts_cadre + n - 1, :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + Nb_ressorts_vert - m, :] - \
                                F_spring_croix[2 * (n - 1) - 1, :]  # en haut a droite
    F_spring_points[(m - 1) * n, :] = F_spring[2 * n + m - 1, :] + \
                                      F_spring[2 * n + m, :] + \
                                      F_spring[Nb_ressorts_cadre + (m - 2) * n, :] - \
                                      F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m - 1, :] + \
                                      F_spring_croix[Nb_ressorts_croix - 2 * (n - 1) + 1, :]  # en bas a gauche
    F_spring_points[m * n - 1, :] = F_spring[n + m - 1, :] + \
                                    F_spring[n + m, :] + \
                                    F_spring[Nb_ressorts_cadre + Nb_ressorts_horz - 1, :] + \
                                    F_spring[Nb_ressorts - 1, :] + \
                                    F_spring_croix[Nb_ressorts_croix - 2, :]  # en haut a gauche


    # - points du bord de la toile> Pour lordre des termes de la somme, on part du ressort cadre puis sens trigo
    # - cote droit VERIFIE CEST OK
    for i in range(1, n - 1):
        F_spring_points[i, :] = F_spring[i, :] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i, :] - \
                                F_spring_croix[2 * (i - 1) + 1, :] - \
                                F_spring[Nb_ressorts_cadre + i, :] - \
                                F_spring_croix[2 * (i - 1) + 2, :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1), :]

        # - cote gauche VERIFIE CEST OK
    j = 0
    for i in range((m - 1) * n + 1, m * n - 1):
        F_spring_points[i, :] = F_spring[Nb_ressorts_cadre - m - (2 + j), :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (j + 1) * m - 1, :] + \
                                F_spring_croix[Nb_ressorts_croix - 2 * n + 1 + 2 * (j + 2), :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz - n + j + 1, :] + \
                                F_spring_croix[Nb_ressorts_croix - 2 * n + 2 * (j + 1), :] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (j + 2) * m - 1, :]
        j += 1

        # - cote haut VERIFIE CEST OK
    j = 0
    for i in range(2 * n - 1, (m - 1) * n, n):
        F_spring_points[i, :] = F_spring[n + 1 + j, :] - \
                                F_spring[Nb_ressorts_cadre + i, :] - \
                                F_spring_croix[(j + 2) * (n - 1) * 2 - 1, :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (Nb_ressorts_vert + 1) - (m - j),
                                :] + \
                                F_spring_croix[(j + 1) * (n - 1) * 2 - 2, :] + \
                                F_spring[Nb_ressorts_cadre + i - n, :]
        j += 1


        # - cote bas VERIFIE CEST OK
    j = 0
    for i in range(n, (m - 2) * n + 1, n):
        F_spring_points[i, :] = F_spring[Nb_ressorts_cadre - (2 + j), :] + \
                                F_spring[Nb_ressorts_cadre + n * j, :] + \
                                F_spring_croix[1 + 2 * (n - 1) * j, :] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + j + 1, :] - \
                                F_spring_croix[2 * (n - 1) * (j + 1), :] - \
                                F_spring[Nb_ressorts_cadre + n * (j + 1), :]
        j += 1


    # Points du centre de la toile (tous les points qui ne sont pas en contact avec le cadre)
    # on fait une colonne puis on passe a la colonne de gauche etc
    # dans lordre de la somme : ressort horizontal de droite puis sens trigo
    for j in range(1, m - 1):
        for i in range(1, n - 1):
            F_spring_points[j * n + i, :] = F_spring[Nb_ressorts_cadre + (j - 1) * n + i, :] + \
                                            F_spring_croix[2 * j * (n - 1) - 2 * n + 3 + 2 * i, :] - \
                                            F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i + j, :] - \
                                            F_spring_croix[j * 2 * (n - 1) + i * 2, :] - \
                                            F_spring[Nb_ressorts_cadre + j * n + i, :] - \
                                            F_spring_croix[j * 2 * (n - 1) + i * 2 - 1, :] + \
                                            F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1) + j, :] + \
                                            F_spring_croix[j * 2 * (n - 1) - 2 * n + 2 * i, :]

    F_point = F_masses - F_spring_points - F_amortissement #SIGNE ?????

    return F_point

def Force_totale_par_points(X, Xdot, C, F, Masse_centre, ind_masse):

    Pt = X
    F = Param_variable_force(F)
    dict_fixed_params = Param_fixe(Masse_centre)
    Pt_ancrage = Points_ancrage_fix(dict_fixed_params)
    Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2 = Bouts_ressorts_repos(Pt_ancrage, Pt)

    F_spring, F_spring_croix, F_masses = Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params)
    F_amortissement = Force_amortissement(Xdot, C)

    F_point = Force_point(F_spring, F_spring_croix, F_masses,  F_amortissement)

    # ajout des forces de l'athlete
    F_point[ind_masse, :] = F[0, :]
    F_point[ind_masse + 1, :] = F[1, :]
    F_point[ind_masse - 1, :] = F[2, :]
    F_point[ind_masse + 15, :] = F[3, :]
    F_point[ind_masse - 15, :] = F[4, :]

    return F_point

def Force_totale_points(X, Xdot, C, F, Masse_centre, ind_masse):

    Pt = X
    F = Param_variable_force(F)
    dict_fixed_params = Param_fixe(Masse_centre)
    Pt_ancrage = Points_ancrage_fix(dict_fixed_params)
    Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2 = Bouts_ressorts_repos(Pt_ancrage, Pt)

    F_spring, F_spring_croix, F_masses = Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params)
    F_amortissement = Force_amortissement(Xdot, C)

    F_point = Force_point(F_spring, F_spring_croix, F_masses,  F_amortissement)

    # ajout des forces de l'athlete
    F_point[ind_masse, :] = F[0, :]
    F_point[ind_masse + 1, :] = F[1, :]
    F_point[ind_masse - 1, :] = F[2, :]
    F_point[ind_masse + 15, :] = F[3, :]
    F_point[ind_masse - 15, :] = F[4, :]

    F_tot =np.zeros((1, 3))
    for point in range (n*m):
            F_tot[0, 0] += F_point[point, 0]
            F_tot[0, 1] += F_point[point, 1]
            F_tot[0, 2] += F_point[point, 2]

    return F_tot

def rotation_points(Pos_repos,Pt_ancrage):
    mat_base_collecte = np.array([[ 0.99964304, -0.02650231,  0.00338079],
               [ 0.02650787,  0.99964731, -0.00160831],
               [-0.00333697,  0.00169736,  0.99999299]])
    #calcul inverse
    mat_base_inv_np = np.linalg.inv(mat_base_collecte)

    Pt_ancrage_new = np.zeros((Nb_ressorts_cadre,3))
    for index in range (Nb_ressorts_cadre) :
        Pt_ancrage_new[index,:] = np.matmul(Pt_ancrage[index,:], mat_base_inv_np)

    Pos_repos_new = np.zeros((n*m, 3))
    for index in range(n*m):
        Pos_repos_new[index, :] = np.matmul(Pos_repos[index, :], mat_base_inv_np)

    return Pos_repos_new,Pt_ancrage_new


def tab2list (tab) :
    list = np.zeros(135 * 3)
    for i in range(135):
        for j in range(3):
            list[j + 3 * i] = tab[i, j]
    return list

def list2tab (list) :
    tab = np.zeros(135, 3)
    for ind in range(135):
        for i in range(3):
            tab[ind, i] = list[i + 3 * ind]
    return tab

def interpolation_collecte(Pt_collecte, labels) :
    """
    Interpoler lespoints manquants de la collecte pour les utiliser dans l'initial guess
    :param Pt_collecte: DM(3,135)
    :param labels: list(nombre de labels)
    :return: Pt_interpole: DM(3,135) (même dimension que Pos_repos)
    """
    #liste avec les bons points aux bons endroits, et le reste vaut 0
    Pt_interpole = np.zeros((3,135))
    Pt_interpole[:] = np.nan # on met des Nan partout
    for ind in range (135) :
        if 't' + str(ind) in labels and np.isnan(Pt_collecte[0, labels.index('t' + str(ind))])==False :
            Pt_interpole[:,ind] = Pt_collecte[:, labels.index('t' + str(ind))]

    return Pt_interpole

def interpolation_collecte_lineaire(Pt_collecte, Pt_ancrage, labels) :
    """
    Interpoler lespoints manquants de la collecte pour les utiliser dans l'initial guess
    :param Pt_collecte: DM(3,135)
    :param labels: list(nombre de labels)
    :return: Pt_interpole: DM(3,135) (même dimension que Pos_repos)
    """
    #liste avec les bons points aux bons endroits, et le reste vaut 0

    Pt_interpole = np.zeros((3,135))
    for ind in range (135) :
        if 't' + str(ind) in labels and np.isnan(Pt_collecte[0, labels.index('t' + str(ind))])==False :
            Pt_interpole[:,ind] = Pt_collecte[:, labels.index('t' + str(ind))]

    #séparation des colonnes
    Pt_colonnes = []
    for i in range (9) :
        Pt_colonnei = np.zeros((3,17))
        Pt_colonnei[:,0] = Pt_ancrage[2*(n+m) - 1 - i, :]
        Pt_colonnei[:,1:16] = Pt_interpole[:, 15*i:15*(i+1)]
        Pt_colonnei[:,-1] = Pt_ancrage[n + i, :]
        Pt_colonnes+= [Pt_colonnei]

    #interpolation des points de chaque colonne
    Pt_inter_liste=[]
    for colonne in range (9) :
        for ind in range (17) :
            if Pt_colonnes[colonne][0,ind] == 0 :
                gauche = Pt_colonnes[colonne][:,ind-1]
                j=1
                while Pt_colonnes[colonne][0,ind+j] == 0 :
                    j+=1
                droite= Pt_colonnes[colonne][:,ind+j]
                Pt_colonnes[colonne][:, ind] = gauche + (droite-gauche)/(j+1)
        Pt_colonne_ind = Pt_colonnes[colonne][:, 1:16]
        Pt_inter_liste += [Pt_colonnes[colonne][:, 1:16]]

    #on recolle les colonnes interpolées
    Pt_inter = []
    for i in range (9) :
        Pt_inter = cas.horzcat(Pt_inter, Pt_inter_liste[i])

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1.1, 1.8, 1])
    # ax.plot(np.array((Pt_inter))[0, :], np.array((Pt_inter))[1, :], np.array((Pt_inter))[2, :], '.b')
    # plt.show()

    Pt_inter = np.array((Pt_inter))

    return Pt_inter

def Vit_initial(Ptavant, Ptapres, labels):
    """
    :param Pt intervalle dynamique
    :param labels
    :return:
    vitesse initiale a l'instant -1, de la frame 0 de l'intervalle dynamique
    """

    position_imoins1 = interpolation_collecte(Ptavant, labels)
    position_iplus1 = interpolation_collecte(Ptapres, labels)
    distance_xyz = np.abs(position_imoins1 - position_iplus1)
    vitesse_initiale = distance_xyz / (2 * 0.002)

    return vitesse_initiale

def Bouts_ressort_collecte(Pt_interpolés, Pt_ancrage_collecte, label_ancrage, Masse_centre):
    """
    :param Pt_interpolés: point collecte
    :param nb_frame: nombre de frame dans l'intervalle dynamique

    :return: bouts des ressorts, coordonnées
    """

    dict_fixed_params = Param_fixe(Masse_centre)
    Pt_ancrage_collecte_interpolés = Interpolation_ancrage(Pt_ancrage_collecte, label_ancrage)  # renvoie les pt dancrages mais avec des nan si il y a rien (48*3)
    # Pt_ancrage, Pos_repos = Points_ancrage_repos(dict_fixed_params)
    Spring_bouts1, Spring_bouts2 = Spring_bouts(Pt_interpolés, Pt_ancrage_collecte_interpolés)
    Spring_bouts_croix1, Spring_bouts_croix2 = Spring_bouts_croix(Pt_interpolés)

    return Spring_bouts1, Spring_bouts2, Spring_bouts_croix1, Spring_bouts_croix2

def Integration(X, Xdot, F, C, Masse_centre, ind_masse):

    dict_fixed_params = Param_fixe(Masse_centre)
    Pt_ancrage = Points_ancrage_fix(dict_fixed_params)
    M = dict_fixed_params['M']
    dt = 1/500
    Pt = list2tab(X)
    Vitesse = list2tab(Xdot)
    F = Param_variable_force(F) #bonnes dimensions

    # initialisation
    Pt_integ = np.zeros((n * m, 3))
    vitesse_calc = np.zeros((n * m, 3))
    accel_calc = np.zeros((n * m, 3))

    Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2 = Bouts_ressorts_repos(Pt_ancrage, Pt)
    F_spring, F_spring_croix, F_masses = Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, dict_fixed_params)
    F_amortissement = Force_amortissement(Vitesse, C)
    F_point = Force_point(F_spring, F_spring_croix, F_masses, F_amortissement)
    # ajout des forces de l'athlete
    F_point[ind_masse, :] = F[0, :]
    F_point[ind_masse + 1, :] = F[1, :]
    F_point[ind_masse - 1, :] = F[2, :]
    F_point[ind_masse + 15, :] = F[3, :]
    F_point[ind_masse - 15, :] = F[4, :]

    for i in range(0, n * m):
        # acceleration
        accel_calc[i, :] = F_point[i, :] / M[i]
        # vitesse
        vitesse_calc[i, :] = dt * accel_calc[i, :] + Vitesse[i, :]
        # position
        Pt_integ[i, :] = dt * vitesse_calc[i, :] + Pt[i, :]

    return Pt_integ, vitesse_calc

def obj_force(Force_collecte, force_accel_cadre):
    masse_trampo = 270

    obj_force = np.zeros((1, 3))
    obj_force[0, 0] = Force_collecte[0] - force_accel_cadre[0]
    obj_force[0, 1] = Force_collecte[1] - force_accel_cadre[1]
    obj_force[0, 2] = Force_collecte[2] + masse_trampo * 9.81 - force_accel_cadre[2]

    return obj_force

def Acceleration_cadre(Pt, total_frame):
    masse_cadre = 260
    dt = 1 / 500

    axe = 11
    time = np.linspace(0, 10, total_frame)
    time2 = np.linspace(0, 10, total_frame-2)

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
        accy.append(((yfil[pos + 1] + yfil[pos - 1] - 2 * yfil[pos]) / (dt ** 2)))
        accx.append(((xfil[pos + 1] + xfil[pos - 1] - 2 * xfil[pos]) / (dt ** 2)))
        fz.append(((zfil[pos + 1] + zfil[pos - 1] - 2 * zfil[pos]) / (dt ** 2)) * (masse_cadre))
        fy.append(((yfil[pos + 1] + yfil[pos - 1] - 2 * yfil[pos]) / (dt ** 2)) * (masse_cadre))
        fx.append(((xfil[pos + 1] + xfil[pos - 1] - 2 * xfil[pos]) / (dt ** 2)) * (masse_cadre))


    fig, ax = plt.subplots(2, 3)
    fig.suptitle('Position sur Z du point d\'ancrage')

    ax[0, 0].plot(time[:len(Pt)], x, label='Données collectées')
    ax[0, 0].plot(time[:len(Pt)], xfil, '-r', label='Données filtrées')
    ax[0, 0].set_xlabel('Temps (s)')
    ax[0, 0].set_ylabel('X (m)')
    # ax[1,0].plot(time2, accx, '-g')
    ax[1, 0].plot(time2[:len(fx)], fx, color='lime', marker='o')
    ax[1, 0].set_xlabel('Temps (s)')
    ax[1, 0].set_ylabel('Force cadre X (N)')

    ax[0, 1].plot(time[:len(Pt)], y)
    ax[0, 1].plot(time[:len(Pt)], yfil, '-r')
    ax[0, 1].set_xlabel('Temps (s)')
    ax[0, 1].set_ylabel('Y (m)')
    # ax[1,1].plot(time2, accy, '-g')
    ax[1, 1].plot(time2[:len(fy)], fy, color='lime', marker='o')
    ax[1, 1].set_xlabel('Temps (s)')
    ax[1, 1].set_ylabel('Force cadre Y (N)')

    ax[0, 2].plot(time[:len(Pt)], z)
    ax[0, 2].plot(time[:len(Pt)], zfil, '-r')
    ax[0, 2].set_xlabel('Temps (s)')
    ax[0, 2].set_ylabel('Z (m)')
    # ax[1,2].plot(time2, accz, '-g', label = 'Accélération')
    ax[1, 2].plot(time2[:len(fz)], fz, color='lime', marker='o', label='Force de l\'accélération du cadre')
    ax[1, 2].set_xlabel('Temps (s)')
    ax[1, 2].set_ylabel('Force cadre Z (N)')

    fig.legend(shadow=True)
    # plt.show()

    return np.array(( fx, fy, fz))

def Point_ancrage(Point_collecte, labels):
    """
    :param Point_collecte: ensemble des points de collecte de l'intervalle dynamique
    :param labels: labels des points
    :return:
    ensemble des coordonnées des points d'ancrage a chaque frame (point du cadre avec label C) sous forme de tableaux
    """
    point_ancrage = []
    label_ancrage = []
    for frame in range (len(Point_collecte)):
        pt_ancrage_frame = []
        for idx, lab in enumerate(labels):
            if 'C' in lab:
                pt_ancrage_frame.append(Point_collecte[frame][:,idx])

                if frame == 0:
                    label_ancrage.append(lab)
        point_ancrage.append(pt_ancrage_frame)

    #transformation des elements en array
    liste_point_ancrage = []
    for list_point in point_ancrage:
        tab_point = np.array((list_point[0]))
        for ligne in range(1, len(list_point)):
            tab_point = np.vstack((tab_point, list_point[ligne]))
        liste_point_ancrage.append(tab_point)

    # for point_anc in liste_point_ancrage:
    #     point_anc = rotation_points(point_anc)

    return liste_point_ancrage, label_ancrage

def Interpolation_ancrage(liste_point_ancrage, label_ancrage):
    Pt_anc_interp = np.zeros((48,3))
    Pt_anc_interp[:,:] = np.nan
    for ind in range(48):
        if 'C' + str(ind) in label_ancrage:
            Pt_anc_interp[ind, :] = liste_point_ancrage[label_ancrage.index('C' + str(ind)) , :]

    return Pt_anc_interp

def Resultat_PF_collecte(participant,statique_name, vide_name, trial_name, intervalle_dyna) :
    def open_c3d(participant, trial_name):
        dossiers = ['c3d/statique', 'c3d/participant_01', 'c3d/participant_02', 'c3d/', 'c3d/test_plateformes']
        file_path = '/home/lim/Documents/Thea/UDEM_S2M_Thea_WIP/collecte/' + dossiers[participant]
        c3d_file = c3d(file_path + '/' + trial_name + '.c3d')
        return c3d_file

    def matrices():
        # M1,M2,M4 sont les matrices obtenues apres la calibration sur la plateforme 3
        M4_new = [[5.4526, 0.1216, 0.0937, -0.0001, -0.0002, 0.0001],
                  [0.4785, 5.7700, 0.0178, 0.0001, 0.0001, 0.0001],
                  [-0.1496, -0.1084, 24.6172, 0.0000, -0.0000, 0.0002],
                  [12.1726, -504.1483, -24.9599, 3.0468, 0.0222, 0.0042],
                  [475.4033, 10.6904, -4.2437, -0.0008, 3.0510, 0.0066],
                  [-6.1370, 4.3463, -4.6699, -0.0050, 0.0038, 1.4944]]

        M1_new = [[2.4752, 0.1407, 0.0170, -0.0000, -0.0001, 0.0001],
                  [0.3011, 2.6737, -0.0307, 0.0000, 0.0001, 0.0000],
                  [0.5321, 0.3136, 11.5012, -0.0000, -0.0002, 0.0011],
                  [20.7501, -466.7832, -8.4437, 1.2666, -0.0026, 0.0359],
                  [459.7415, 9.3886, -4.1276, -0.0049, 1.2787, -0.0057],
                  [265.6717, 303.7544, -45.4375, -0.0505, -0.1338, 0.8252]]

        M2_new = [[2.9967, -0.0382, 0.0294, -0.0000, 0.0000, -0.0000],
                  [-0.1039, 3.0104, -0.0324, -0.0000, -0.0000, 0.0000],
                  [-0.0847, -0.0177, 11.4614, -0.0000, -0.0000, -0.0000],
                  [13.6128, 260.5267, 17.9746, 1.2413, 0.0029, 0.0158],
                  [-245.7452, -7.5971, 11.5052, 0.0255, 1.2505, -0.0119],
                  [-10.3828, -0.9917, 15.3484, -0.0063, -0.0030, 0.5928]]

        M3_new = [[2.8992, 0, 0, 0, 0, 0],
                  [0, 2.9086, 0, 0, 0, 0],
                  [0, 0, 11.4256, 0, 0, 0],
                  [0, 0, 0, 1.2571, 0, 0],
                  [0, 0, 0, 0, 1.2571, 0],
                  [0, 0, 0, 0, 0, 0.5791]]

        # zeros donnes par Nexus
        zeros1 = np.array([1.0751899, 2.4828501, -0.1168980, 6.8177500, -3.0313399, -0.9456340])
        zeros2 = np.array([0., -2., -2., 0., 0., 0.])
        zeros3 = np.array([0.0307411, -5., -4., -0.0093422, -0.0079338, 0.0058189])
        zeros4 = np.array([-0.1032560, -3., -3., 0.2141770, 0.5169040, -0.3714130])

        return M1_new, M2_new, M3_new, M4_new, zeros1, zeros2, zeros3, zeros4

    def matrices_rotation():
        theta31 = 0.53 * np.pi / 180
        rot31 = np.array([[np.cos(theta31), -np.sin(theta31)],
                          [np.sin(theta31), np.cos(theta31)]])

        theta34 = 0.27 * np.pi / 180
        rot34 = np.array([[np.cos(theta34), -np.sin(theta34)],
                          [np.sin(theta34), np.cos(theta34)]])

        theta32 = 0.94 * np.pi / 180
        rot32 = np.array([[np.cos(theta32), -np.sin(theta32)],
                          [np.sin(theta32), np.cos(theta32)]])

        return rot31,rot34,rot32

    def plateformes_separees_rawpins(c3d):
        # on garde seulement les Raw pins
        force_labels = c3d['parameters']['ANALOG']['LABELS']['value']
        ind = []
        for i in range(len(force_labels)):
            if 'Raw' in force_labels[i]:
                ind = np.append(ind, i)
        # ind_stop=int(ind[0])
        indices = np.array([int(ind[i]) for i in range(len(ind))])
        ana = c3d['data']['analogs'][0, indices, :]
        platform1 = ana[0:6, :]  # pins 10 a 15
        platform2 = ana[6:12, :]  # pins 19 a 24
        platform3 = ana[12:18, :]  # pins 28 a 33
        platform4 = ana[18:24, :]  # pins 1 a 6

        platform = np.array([platform1, platform2, platform3, platform4])
        return platform

    def soustraction_zero(platform):  # soustrait juste la valeur du debut aux raw values
        longueur = np.size(platform[0, 0])
        zero_variable = np.zeros((4, 6))
        for i in range(6):
            for j in range(4):
                zero_variable[j, i] = np.mean(platform[j, i, 0:100])
                platform[j, i, :] = platform[j, i, :] - zero_variable[j, i] * np.ones(longueur)
        return platform

    def plateforme_calcul(platform,intervalle_dyna,participant):  # prend les plateformes separees, passe les N en Nmm, calibre, multiplie par mat rotation, met dans la bonne orientation

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

        if participant !=0 :
            platform_new = platform_new[:,:,intervalle_dyna[0]:intervalle_dyna[1]]

        return platform_new

    def soustraction_vide(c3d_statique, c3d_vide):  # pour les forces calculees par Vicon
        platform_statique = plateformes_separees_rawpins(c3d_statique)
        platform_vide = plateformes_separees_rawpins(c3d_vide)
        platform = np.copy(platform_statique)

        # on soustrait les valeurs du fichier a vide
        for j in range(6):
            for i in range(4):
                platform[i, j, :] = platform_statique[i, j, :] - np.mean(platform_vide[i, j, :])
        platform = plateforme_calcul(platform,0,0)
        return platform

    def dynamique(c3d_experimental,intervalle_dyna,participant):
        platform = plateformes_separees_rawpins(c3d_experimental)
        platform = soustraction_zero(platform)
        platform = plateforme_calcul(platform,intervalle_dyna,participant)
        return platform

    def Named_markers(c3d_experimental):
        labels = c3d_experimental['parameters']['POINT']['LABELS']['value']

        indices_supp = []
        for i in range(len(labels)):
            if '*' in labels[i]:
                indices_supp = np.append(indices_supp, i)
        ind_stop = int(indices_supp[0])

        labels = c3d_experimental['parameters']['POINT']['LABELS']['value'][0:ind_stop]
        named_positions = c3d_experimental['data']['points'][0:3, 0:ind_stop, :]

        ind_milieu = labels.index('t67')
        moyenne_milieu = np.array([np.mean(named_positions[i, ind_milieu, :100]) for i in range(3)])
        return labels,moyenne_milieu, named_positions

    def position_statique(named_positions, moyenne,named_positions_vide,moyenne_vide):
        # on soustrait la moyenne de la position du milieu
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_vide[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        position_moyenne = np.zeros((3, np.shape(named_positions_bonsens)[1]))
        for ind_print in range(np.shape(named_positions_bonsens)[1]):
            position_moyenne[:, ind_print] = np.array(
                [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)])

        #passage de mm en m :
        named_positions_bonsens *= 0.001

        return named_positions_bonsens

    def position_dynamique(named_positions, moyenne_milieu, intervalle_dyna):
        # on soustrait la moyenne de la position du milieu sur les 100 premiers points
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_milieu[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        position_moyenne = np.zeros((3, np.shape(named_positions_bonsens)[1]))
        for ind_print in range(np.shape(named_positions_bonsens)[1]):
            position_moyenne[:, ind_print] = np.array(
                [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)])

        #passage de mm en m :
        named_positions_bonsens*= 0.001

        positions_new = named_positions_bonsens[:,:,intervalle_dyna[0] : intervalle_dyna[1]]

        return positions_new

    def point_le_plus_bas(points, labels,intervalle_dyna):
        # on cherche le z min obtenu sur notre intervalle

        #compter le nombre de nan par marqueur au cours de l'intervalle :
        isnan_marqueur = np.zeros(len(labels))
        for i in range (len(labels)) :
            # for time in range (intervalle_dyna[0], intervalle_dyna[1]) :
            for time in range(intervalle_dyna[1] - intervalle_dyna[0]):
                if np.isnan(points[2, i,time])==True :
                    isnan_marqueur[i] += 1

        #trier les marqueurs en t qui sont des nan sur tout l'intervalle
        labels_notnan = []
        for i in range (len(labels)) :
            #s'il y a autant de nan que de points dans l'intervalle, alors on n'en veut pas :
            if isnan_marqueur[i] != intervalle_dyna[1] - intervalle_dyna[0] :
                labels_notnan += [labels[i]]

        indice_notnan= []
        for i in range (len (labels_notnan)) :
            indice_notnan += [labels.index(labels_notnan[i])]

        labels_modified=[labels[i] for i in indice_notnan]
        points_modified = points[:,indice_notnan,:]

        #on peut enfin calculer le minimum :
        # on cherche le z min de chaque marqueur (on en profite pour supprimer les nan)
        # minimum_marqueur = [np.nanmin(points_modified[2, i,intervalle_dyna[0] : intervalle_dyna[1]]) for i in range(len(indice_notnan))]
        minimum_marqueur = [np.nanmin(points_modified[2, i, :]) for i in range(len(indice_notnan))]

        #indice du marqueur ayant le plus petit z sur tout l'intervalle
        argmin_marqueur = np.argmin(minimum_marqueur)
        label_min = labels_modified[argmin_marqueur]

        return argmin_marqueur,label_min

    if participant == 0 :
        c3d_vide = open_c3d(0, vide_name)
        c3d_statique = open_c3d(0, statique_name)
        platform = soustraction_vide(c3d_statique,c3d_vide)  # plateforme statique a laquelle on a soustrait la valeur de la plateforme a vide
        labels,moyenne_milieu, named_positions = Named_markers(c3d_statique)
        labels_vide,moyenne_milieu_vide, named_positions_vide = Named_markers(c3d_vide)
        Pt_collecte = position_statique(named_positions,moyenne_milieu, named_positions_vide, moyenne_milieu_vide)

    else :
        c3d_experimental = open_c3d(participant, trial_name)
        platform = dynamique(c3d_experimental,intervalle_dyna,participant)
        labels,moyenne_milieu, named_positions = Named_markers(c3d_experimental)
        Pt_collecte = position_dynamique(named_positions, moyenne_milieu,intervalle_dyna)

    longueur = np.size(platform[0, 0])
    F_totale_collecte = np.zeros((longueur,3))
    for i in range (3) :
        for x in range (longueur) :
            F_totale_collecte[x,i] = platform[0,i,x] + platform[1,i,x] + platform[2,i,x] + platform[3,i,x]

    # position_instant = Pt_collecte[:, :, int(7050)]
    argmin_marqueur,label_min = point_le_plus_bas(Pt_collecte, labels,intervalle_dyna) #coordonnée du marqueur le plus bas dans labels
    ind_marqueur_min = int(label_min[1:])#coordonnées de ce marqueur adaptées à la simulation
    print('Point le plus bas sur l\'intervalle ' + str(intervalle_dyna) + ' : ' + str(label_min))

    #retourner des tableaux casadi
    F_collecte_cas = np.zeros(np.shape(F_totale_collecte))
    F_collecte_cas[:,:] = F_totale_collecte[:,:]

    Pt_collecte_tab=[0 for i in range (np.shape(Pt_collecte)[2])]
    for time in range (np.shape(Pt_collecte)[2]) :
        #passage en casadi
        Pt_time = np.zeros((3,np.shape(Pt_collecte)[1]))
        Pt_time[:,:] = Pt_collecte[:,:,time] #attention pas dans le même ordre que Pt_simu
        #séparation en Nb_increments tableaux
        Pt_collecte_tab[time] = Pt_time

    return F_collecte_cas,Pt_collecte_tab,labels,ind_marqueur_min


######################
######################
######################
participant = 2
statique_name = 'labeled_statique_leftfront_D7'
vide_name = 'labeled_statique_centrefront_vide'
trial_name  = 'labeled_p2_troisquartback_01'
Masse_centre = 64.5

total_frame = 7763
intervalle_dyna = [5170, 5173]
nb_frame = intervalle_dyna[1] - intervalle_dyna[0]

dict_fixed_params = Param_fixe(Masse_centre)
Pt_ancrage, Pos_repos = Points_ancrage_repos(dict_fixed_params)

F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(participant, statique_name, vide_name, trial_name, intervalle_dyna)

## - Donnees collectées
Pt_ancrage_collecte, label_ancrage = Point_ancrage(Pt_collecte_tab, labels)

Pt_collecte_tab[1] = interpolation_collecte_lineaire(Pt_collecte_tab[1], Pt_ancrage, labels)
Pt_collecte_tab[2] = interpolation_collecte_lineaire(Pt_collecte_tab[2], Pt_ancrage, labels)


force_acceleration_cadre = Acceleration_cadre(Pt_ancrage_collecte, total_frame).T

F_obj1 = obj_force(F_totale_collecte[1], force_acceleration_cadre[0])
F_obj1 = obj_force(F_totale_collecte[2], force_acceleration_cadre[0])



path = '/home/lim/Documents/Jules/Jules_toile_A2022/Dynamique/Result/Optim_C.plk'
with open(path, 'rb') as file:
    sol = pickle.load(file)
    w0 = pickle.load(file)
    ubw = pickle.load(file)
    lbw = pickle.load(file)

solution = np.array((sol['x']))
objectif = sol['f']

C = solution[:40]
c_tot = Param_variable(C).reshape((9,15))
plt.matshow(c_tot)
plt.title('Reparition des C')
plt.show()
C_init = w0[:40]

position1 = solution[40:445].reshape((135,3))
vitesse1 = solution[445:850].reshape((135,3))
force1 = solution[850:865].reshape((5,3))

position2 = solution[865:1270].reshape((135,3))
vitesse2 = solution[1270:1675].reshape((135,3))
force2 = solution[1675:1690].reshape((5,3))

pos1init = np.array(w0[40:445]).reshape((135,3))
pos2init = np.array(w0[865:1270]).reshape((135,3))


# -- positions -- #
fig = plt.figure()
## frame 1
ax = plt.subplot(1,2,1, projection='3d')
ax.set_box_aspect([1.1, 1.8, 1])
ax.plot(0,0,-1.2,'xw')
# point simulés
ax.plot(position1[:,0] ,position1[:,1] ,position1[:,2], '+r')
# point collectés
ax.plot(Pt_collecte_tab[1][0,:] ,Pt_collecte_tab[1][1,:] ,Pt_collecte_tab[1][2,:], '.b')


## frame 2
ax = plt.subplot(1,2,2, projection='3d')
ax.set_box_aspect([1.1, 1.8, 1])
ax.plot(0,0,-1.2,'xw')
# point collectés
ax.plot(Pt_collecte_tab[2][0,:] ,Pt_collecte_tab[2][1,:] ,Pt_collecte_tab[2][2,:], '.b')
# point simulés
ax.plot(position2[:,0] ,position2[:,1] ,position2[:,2], '+r')

plt.show()

# -- vitesses
# plt.matshow(vitesse1, aspect='auto')
# plt.matshow(vitesse2, aspect='auto')
# plt.show()


# afficher forces en chaque point
F_ready1 = solution[850:865]
F_ready2 = solution[1675:1690]
F_instant_t = Force_totale_par_points(position1, vitesse1, C, F_ready1, Masse_centre, ind_masse)
F_instant_t1 = Force_totale_par_points(position2, vitesse2, C, F_ready2, Masse_centre, ind_masse)

F_tot1 = Force_totale_points(position1, vitesse1, C, F_ready1, Masse_centre, ind_masse)


plt.matshow(F_instant_t, aspect='auto')
plt.matshow(F_instant_t1, aspect='auto')
plt.show()


test = 'test'