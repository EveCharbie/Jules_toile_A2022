"""
Optimization of the position and velocity of the trampoline bed model points.
The elasticity coefficients are taken from the static optimization (K are constants).
No damping coefficients are used.
The goal is to minimize the distance between the model points and the markers and the difference between the force

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
import time
import pickle

import sys
sys.path.append("../")
from enums import InitialGuessType

sys.path.append("../Statique/")
from Optim_35_essais_kM_regul_koblique import Param_fixe
from Verif_optim_position_k_fixe import Optimisation
from modele_dynamique_nxm_DimensionsReelles import Resultat_PF_collecte, Point_ancrage, Point_toile_init

def get_list_results_dynamic(participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval):
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )
    Pt_ancrage, labels_ancrage = Point_ancrage(Pt_collecte_tab, labels)
    Pt_collecte, label_toile = Point_toile_init(Pt_collecte_tab, labels)

    return F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage

##########################################################################################################################
def main():

    # SELECTION OF THE RESULTS FROM THE DATA COLLECTION
    participant = 1
    # participant_1: 64.5 kg
    # participant_2: 87.2 kg
    weight = 64.5
    static_trial_name = "labeled_statique_leftfront_D7"
    trial_name = "labeled_p1_sauthaut_01"
    empty_trial_name = "labeled_statique_centrefront_vide"
    jump_frame_index_interval = [
        7000,
        7020,
        # 7170,
    ]  # This range repends on the trial. To find it, one should use the code plateforme_verification_toutesversions.py.
    dt = 1 / 500  # Hz

    initial_guess = InitialGuessType.SURFACE_INTERPOLATION
    optimize_static_mass = False

    dict_fixed_params = Param_fixe()
    F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage = get_list_results_dynamic(participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval)

    ########################################################################################################################

    start_main = time.time()

    for idx, frame in enumerate(jump_frame_index_interval):
        Solution, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, f = Optimisation(
            F_totale_collecte[idx, :], Pt_collecte[idx, :, :], labels, ind_masse, Pt_ancrage[idx, :, :], weight, trial_name, initial_guess,
            optimize_static_mass, dict_fixed_params
        )

        # recuperation et affichage
        k = np.array(Solution[:12])
        M = []
        Pt = []
        F_totale = []
        F_point = []

        for i in range(len(essais)):
            M.append(np.array(Solution[12 + n*m*3 * i + 5 * i : 17 + n*m*3 * i + 5 * i]))
            Pt.append(np.reshape(Solution[17 + n*m*3 * i + 5 * i : 422 + n*m*3 * i + 5 * i], (n*m, 3)))

            F_totale.append(
                Calcul_Pt_F(
                    Solution[17 + n*m*3 * i + 5 * i : 422 + n*m*3 * i + 5 * i],
                    Pt_ancrage[i],
                    dict_fixed_params,
                    k,
                    ind_masse[i],
                    Solution[12 + n*m*3 * i + 5 * i : 17 + n*m*3 * i + 5 * i],
                )[0]
            )
            F_point.append(
                Calcul_Pt_F(
                    Solution[17 + n*m*3 * i + 5 * i : 422 + n*m*3 * i + 5 * i],
                    Pt_ancrage[i],
                    dict_fixed_params,
                    k,
                    ind_masse[i],
                    Solution[12 + n*m*3 * i + 5 * i : 17 + n*m*3 * i + 5 * i],
                )[1]
            )

            F_totale[i] = cas.evalf(F_totale[i])
            F_point[i] = cas.evalf(F_point[i])
            F_point[i] = np.array(F_point[i])

            Pt_collecte[i] = np.array(Pt_collecte[i])  # permet de mettre pt collecte sous la bonne forme pour l'utiliser apres
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
