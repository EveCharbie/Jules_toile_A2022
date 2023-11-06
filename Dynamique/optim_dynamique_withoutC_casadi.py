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

    n = 15
    m = 9

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
    F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage = get_list_results_dynamic(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )

    ########################################################################################################################

    for idx, frame in enumerate(jump_frame_index_interval):
        Solution, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, f, status = Optimisation(
            F_totale_collecte[idx, :],
            Pt_collecte[idx, :, :],
            labels,
            ind_masse,
            Pt_ancrage[idx, :, :],
            weight,
            trial_name,
            initial_guess,
            optimize_static_mass,
            dict_fixed_params,
        )

        M = np.array(Solution[0:5])
        Pt = np.reshape(Solution[5:], (n * m, 3))

        # ENREGISTREMENT PICKLE#
        if status == 0:
            ends_with = "CVG"
        else:
            ends_with = "DVG"
        path = f"results_multiple_static_optim_in_a_row/frame{frame}_{ends_with}.pkl"
        with open(path, "wb") as file:
            pickle.dump(Solution, file)
            pickle.dump(M, file)
            pickle.dump(Pt, file)
            pickle.dump(labels, file)
            pickle.dump(Pt_collecte, file)
            pickle.dump(Pt_ancrage, file)
            pickle.dump(ind_masse, file)
            pickle.dump(f, file)
            pickle.dump(dict_fixed_params, file)
            pickle.dump(trial_name, file)
            pickle.dump(frame, file)


if __name__ == "__main__":
    main()
