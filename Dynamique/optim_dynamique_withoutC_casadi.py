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
from Optim_35_essais_kM_regul_koblique import Param_fixe, Calcul_Pt_F, list2tab
from modele_dynamique_nxm_DimensionsReelles import Resultat_PF_collecte, Point_ancrage, Point_toile_init, Points_ancrage_repos
from Optim_multi_essais_kM_regul_koblique import Pt_bounds, m_bounds
from Verif_optim_position_k_fixe import Param_variable

def get_list_results_dynamic(participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval):
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )
    Pt_ancrage, labels_ancrage = Point_ancrage(Pt_collecte_tab, labels)
    Pt_collecte, label_toile = Point_toile_init(Pt_collecte_tab, labels)

    return F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage


def F_bounds():
    w0_F = np.zeros((5*3))
    lbw_F = np.ones((5*3)) * -1000
    ubw_F = np.ones((5*3)) * 1000
    return w0_F, lbw_F, ubw_F



def a_minimiser(X, K, Ma, F_athl, Pt_collecte, Pt_ancrage, Pt_interpolated, dict_fixed_params, labels, ind_masse):

    n = 15
    m = 9

    _, F_point = Calcul_Pt_F(X, Pt_ancrage, dict_fixed_params, K, ind_masse, Ma)
    F_point[ind_masse, :] = F_athl[0:3]
    F_point[ind_masse+1, :] = F_athl[3:6]
    F_point[ind_masse-1, :] = F_athl[6:9]
    F_point[ind_masse+15, :] = F_athl[9:12]
    F_point[ind_masse-15, :] = F_athl[12:15]

    Pt = list2tab(X)

    Difference = cas.MX.zeros(1)
    for i in range(3):
        for ind in range(n * m):

            if "t" + str(ind) in labels:
                ind_collecte = labels.index("t" + str(ind))  # ATTENTION gérer les nans
                if np.isnan(Pt_collecte[i, ind_collecte]):  # gérer les nans
                    Difference += (
                        0.01 * (Pt[ind, i] - Pt_interpolated[i, ind]) ** 2
                    )  # on donne un poids moins important aux données interpolées
                elif ind in [ind_masse, ind_masse-1, ind_masse+1, ind_masse-15, ind_masse+15]:
                    Difference += 500 * (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2
                else:
                    Difference += (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2
            else:
                Difference += 0.01 * (Pt[ind, i] - Pt_interpolated[i, ind]) ** 2

            Difference += (F_point[ind, i]) ** 2

    obj = cas.Function("f", [X, Ma, F_athl], [1e-6 * Difference]).expand()

    return obj

def Optimisation(
    F_totale_collecte,
    Pt_collecte,
    labels,
    ind_masse,
    Pt_ancrage,
    Masse_centre,
    trial_name,
    initial_guess,
    optimize_static_mass,
    dict_fixed_params,
):
    # PARAM FIXES
    n = 15
    m = 9
    Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)

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
    Ma = cas.MX.sym("Ma", 5)
    X = cas.MX.sym("X", n * m * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...)
    F_athl = cas.MX.sym("F_athl", 5 * 3) # Force applied by the athlete on the 5 points they touch

    # Ma
    w0_m, lbw_m, ubw_m = m_bounds(Masse_centre)
    w0 += w0_m
    lbw += lbw_m
    ubw += ubw_m
    w += [Ma]

    # X
    w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess, Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, labels)
    lbw += lbw_Pt
    ubw += ubw_Pt
    w0 += w0_Pt
    w += [X]

    # Ma
    w0_F, lbw_F, ubw_F = F_bounds()
    w0 += list(w0_F)
    lbw += list(lbw_F)
    ubw += list(ubw_F)
    w += [F_athl]

    # fonction contrainte :
    g += [Ma[0] + Ma[1] + Ma[2] + Ma[3] + Ma[4] - Masse_centre]
    lbg += [0]
    ubg += [0]

    # en statique on ne fait pas de boucle sur le temps :
    K, _, _ = Param_variable(Ma, ind_masse)
    J = a_minimiser(
        X,
        K,
        Ma,
        F_athl,
        Pt_collecte,
        Pt_ancrage_interpolated,
        Pt_interpolated,
        dict_fixed_params,
        labels,
        ind_masse,
    )
    obj = J(X, Ma, F_athl)

    # Create an NLP solver
    prob = {"f": obj, "x": cas.vertcat(*w), "g": cas.vertcat(*g)}
    opts = {"ipopt": {"max_iter": 10000, "linear_solver": "ma57"}}  #  "nlp_scaling_method": "none"
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(
        x0=cas.vertcat(*w0), lbx=cas.vertcat(*lbw), ubx=cas.vertcat(*ubw), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg)
    )
    w_opt = sol["x"].full().flatten()
    status = solver.stats()["return_status"]

    return w_opt, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, sol.get("f"), status


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
        7100,
        7120,
        # 7170,
    ]  # This range repends on the trial. To find it, one should use the code plateforme_verification_toutesversions.py.
    dt = 1 / 500  # Hz

    initial_guess = InitialGuessType.SURFACE_INTERPOLATION
    optimize_static_mass = False

    dict_fixed_params = Param_fixe()
    Fs_totale_collecte, Pts_collecte, labels, ind_masse, Pts_ancrage = get_list_results_dynamic(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )

    ########################################################################################################################

    for idx, frame in enumerate(jump_frame_index_interval):
        Solution, Pt_collecte, F_totale_collecte, ind_masse, labels, Pt_ancrage, dict_fixed_params, f, status = Optimisation(
            Fs_totale_collecte[idx, :],
            Pts_collecte[idx, :, :],
            labels,
            ind_masse,
            Pts_ancrage[idx, :, :],
            weight,
            trial_name,
            initial_guess,
            optimize_static_mass,
            dict_fixed_params,
        )

        M = np.array(Solution[0:5])
        Pt = np.reshape(Solution[5:-15], (n * m, 3))
        F_athl = np.reshape(Solution[-15:], (5, 3))

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
            pickle.dump(F_athl, file)
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
