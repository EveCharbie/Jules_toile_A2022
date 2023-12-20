
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from ezc3d import c3d
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import sys

sys.path.append("../Statique/")
from utils_static import rotation_points

n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)


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


def linear_interpolation_collecte(Pt_collecte, Pt_ancrage, labels):
    """
    Interpoler lespoints manquants de la collecte pour les utiliser dans l'initial guess
    :param Pt_collecte: DM(3,n*m)
    :param labels: list(nombre de labels)
    :return: Pt_interpole: DM(3,135) (même dimension que Pos_repos)
    """
    # liste avec les bons points aux bons endroits, et le reste vaut 0
    Pt_interpolated = np.zeros((3, 135))
    for ind in range(135):
        if "t" + str(ind) in labels and np.isnan(Pt_collecte[0, labels.index("t" + str(ind))]) == False:
            Pt_interpolated[:, ind] = Pt_collecte[:, labels.index("t" + str(ind))]

    # séparation des colonnes
    Pt_colonnes = []
    for i in range(9):
        Pt_colonnei = np.zeros((3, 17))
        Pt_colonnei[:, 0] = Pt_ancrage[2 * (n + m) - 1 - i, :]
        Pt_colonnei[:, 1:16] = Pt_interpolated[:, 15 * i : 15 * (i + 1)]
        Pt_colonnei[:, -1] = Pt_ancrage[n + i, :]
        Pt_colonnes += [Pt_colonnei]

    # interpolation des points de chaque colonne
    Pt_inter_liste = []
    for colonne in range(9):
        for ind in range(17):
            if Pt_colonnes[colonne][0, ind] == 0:
                gauche = Pt_colonnes[colonne][:, ind - 1]
                j = 1
                while Pt_colonnes[colonne][0, ind + j] == 0:
                    j += 1
                droite = Pt_colonnes[colonne][:, ind + j]
                Pt_colonnes[colonne][:, ind] = gauche + (droite - gauche) / (j + 1)
        Pt_colonne_ind = Pt_colonnes[colonne][:, 1:16]
        Pt_inter_liste += [Pt_colonnes[colonne][:, 1:16]]

    # on recolle les colonnes interpolées
    Pt_inter = []
    for i in range(9):
        Pt_inter = cas.horzcat(Pt_inter, Pt_inter_liste[i])

    return Pt_inter


def surface_interpolation_collecte(Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, dict_fixed_params, PLOT_FLAG=False):
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
        if PLOT_FLAG:
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_box_aspect([1.1, 1.8, 1])

        # Fill markers data that we have
        for ind in range(m * n):
            if np.isnan(Pt_collecte[frame][0, ind]) == False:
                Pt_interpolated[frame, ind, :] = Pt_collecte[frame][:, ind]
                Pt_needs_interpolation[frame, ind, :] = 0
                if PLOT_FLAG:
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
                if PLOT_FLAG:
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
                    if PLOT_FLAG:
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
                    if PLOT_FLAG:
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

        if PLOT_FLAG:
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

        if PLOT_FLAG:
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


def get_list_results_dynamic(participant, empty_trial_name, trial_name, jump_frame_index_interval):
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(
        participant, empty_trial_name, trial_name, jump_frame_index_interval
    )
    Pt_ancrage, labels_ancrage = Point_ancrage(Pt_collecte_tab, labels)
    Pt_collecte, label_toile = Point_toile_init(Pt_collecte_tab, labels)

    return F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage

