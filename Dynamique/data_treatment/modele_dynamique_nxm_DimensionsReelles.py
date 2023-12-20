"""
This code presents the error made by using a static model to simulate a dynamic trampoline deformation.
The static K coefficients were found using static optimization.
The error is the difference from the integration of the position of the points vs the marker position measured during the data collection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns
import scipy

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
