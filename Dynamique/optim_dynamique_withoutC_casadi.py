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
import os

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
from Optim_35_essais_kM_regul_koblique import Param_fixe, Calcul_Pt_F, list2tab, Spring_bouts, Spring_bouts_croix
from modele_dynamique_nxm_DimensionsReelles import (
    Resultat_PF_collecte,
    Point_ancrage,
    Point_toile_init,
    Points_ancrage_repos,
    surface_interpolation_collecte,
    spring_bouts_collecte,
    static_forces_calc,
    static_force_in_each_point,
    multiple_shooting_integration,
)
from Optim_multi_essais_kM_regul_koblique import m_bounds
from Verif_optim_position_k_fixe import Param_variable

def get_list_results_dynamic(participant, empty_trial_name, trial_name, jump_frame_index_interval):
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(
        participant, empty_trial_name, trial_name, jump_frame_index_interval
    )
    Pt_ancrage, labels_ancrage = Point_ancrage(Pt_collecte_tab, labels)
    Pt_collecte, label_toile = Point_toile_init(Pt_collecte_tab, labels)

    return F_totale_collecte, Pt_collecte, labels, ind_masse, Pt_ancrage


def F_bounds(initial_guess, trial_name):
    if initial_guess == InitialGuessType.GACO:
        with open(f"{trial_name}_gaco.pkl", "rb") as file:
            data = pickle.load(file)
            F_athl = data["F_athl"]
            w0_F = F_athl.flatten()
    else:
        w0_F = np.zeros((5*3))
    lbw_F = np.ones((5*3)) * -10000
    ubw_F = np.ones((5*3)) * 10000
    return w0_F, lbw_F, ubw_F

def Pt_bounds(initial_guess, Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, dict_fixed_params, trial_name):
    """
    Returns the bounds on the position of the points of the model based on the interpolation of the missing points.
    """

    n = 15
    m = 9

    if initial_guess == InitialGuessType.SURFACE_INTERPOLATION:
        Pt_interpolated, Pt_ancrage_interpolated = surface_interpolation_collecte(
            [Pt_collecte], [Pt_ancrage], Pt_repos, Pt_ancrage_repos, dict_fixed_params, False
        )
        Pt_interpolated = Pt_interpolated[0,:,:].T
        Pt_ancrage_interpolated = Pt_ancrage_interpolated[0, :, :]
    elif initial_guess == InitialGuessType.GACO:
        with open(f"{trial_name}_gaco.pkl", "rb") as file:
            data = pickle.load(file)
            Pt_interpolated = data["Pt"].T
            Pt_ancrage_interpolated = data["Pt_ancrage_interpolated"]
    else:
        raise RuntimeError(f"The interpolation type of the initial guess {initial_guess} is not accepted for this problem.")

    # bounds and initial guess
    lbw_Pt = []
    ubw_Pt = []
    w0_Pt = []

    for k in range(n * m):
        if np.isnan(Pt_collecte[0, k]):
            lbw_Pt += [Pt_interpolated[:, k] - 0.5]
            ubw_Pt += [Pt_interpolated[:, k] + 0.5]
            w0_Pt += [Pt_interpolated[:, k]]
        else:
            lbw_Pt += [Pt_collecte[:, k] - 0.05]
            ubw_Pt += [Pt_collecte[:, k] + 0.05]
            w0_Pt += [Pt_collecte[:, k]]

    return w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated, Pt_ancrage_interpolated

def cost_function(X, K, Ma, F_athl, Pt_collecte, Pt_ancrage, Pt_interpolated, dict_fixed_params, labels, ind_masse):

    _, F_point = Calcul_Pt_F(X, Pt_ancrage, dict_fixed_params, K, ind_masse, Ma)
    Pt = list2tab(X)
    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    spring_elongation = cas.norm_fro(Spring_bout_2 - Spring_bout_1) - dict_fixed_params["l_repos"]
    spring_elongation_croix = cas.norm_fro(Spring_bout_croix_2 - Spring_bout_croix_1) - dict_fixed_params["l_repos_croix"]

    F_point[ind_masse, :] += F_athl[0:3].T
    F_point[ind_masse+1, :] += F_athl[3:6].T
    F_point[ind_masse-1, :] += F_athl[6:9].T
    F_point[ind_masse+15, :] += F_athl[9:12].T
    F_point[ind_masse-15, :] += F_athl[12:15].T

    Pt = list2tab(X)

    Difference = cas.MX.zeros(1)
    for i in range(3):
        for ind in range(n * m):

            if "t" + str(ind) in labels:
                ind_collecte = labels.index("t" + str(ind))  # ATTENTION gérer les nans
                if not np.isnan(Pt_collecte[i, ind_collecte]):  # gérer les nans
                    Difference += 500 * (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2

            if ind in [ind_masse, ind_masse-1, ind_masse+1, ind_masse-15, ind_masse+15]:
                if i == 2:
                    Difference += 0.001 * (F_point[ind, i]) ** 2
                else:
                    Difference += (F_point[ind, i]) ** 2
            else:
                Difference += (F_point[ind, i]) ** 2

    for ind in range(spring_elongation.shape[0]):
        Difference += 500 * spring_elongation[ind] ** 2
    for ind in range(spring_elongation_croix.shape[0]):
        Difference += 0.01 * spring_elongation_croix[ind] ** 2

    obj = cas.Function("f", [X, Ma, F_athl], [Difference]).expand()

    return obj

def constraints(X, Ma, Pt_ancrage, dict_fixed_params, Masse_centre):
    Pt = list2tab(X)
    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    spring_elongation = cas.norm_fro(Spring_bout_2 - Spring_bout_1) - dict_fixed_params["l_repos"]
    spring_elongation_croix = cas.norm_fro(Spring_bout_croix_2 - Spring_bout_croix_1) - dict_fixed_params["l_repos_croix"]

    g = []
    lbg = []
    ubg = []

    g += [Ma[0] + Ma[1] + Ma[2] + Ma[3] + Ma[4] - Masse_centre]
    lbg += [0]
    ubg += [0]

    g += [spring_elongation] + [spring_elongation_croix]
    lbg += [0] * (spring_elongation.shape[0] + spring_elongation_croix.shape[0])
    ubg += [np.inf] * (spring_elongation.shape[0] + spring_elongation_croix.shape[0])

    g = cas.vertcat(*g)

    return g, lbg, ubg

def Optimisation(
    F_totale_collecte,
    Pt_collecte,
    labels,
    ind_masse,
    Pt_ancrage,
    Masse_centre,
    initial_guess,
    dict_fixed_params,
    trial_name,
):
    # PARAM FIXES
    Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)

    # OPTIMISATION :
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []

    # NLP VALUES
    Ma = cas.MX.sym("Ma", 5)
    X = cas.MX.sym("X", n * m * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...)
    F_athl = cas.MX.sym("F_athl", 5 * 3) # Force applied by the athlete on the 5 points they touch

    # PARAM VARIABLE
    K, _, _ = Param_variable(Ma, ind_masse)

    # Ma
    w0_m, lbw_m, ubw_m = m_bounds(Masse_centre, initial_guess, trial_name)
    w0 += w0_m
    lbw += lbw_m
    ubw += ubw_m
    w += [Ma]

    # X
    w0_Pt, lbw_Pt, ubw_Pt, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess, Pt_collecte, Pt_ancrage, Pt_repos, Pt_ancrage_repos, dict_fixed_params, trial_name)
    lbw += lbw_Pt
    ubw += ubw_Pt
    w0 += w0_Pt
    w += [X]

    # Ma
    w0_F, lbw_F, ubw_F = F_bounds(initial_guess, trial_name)
    w0 += list(w0_F)
    lbw += list(lbw_F)
    ubw += list(ubw_F)
    w += [F_athl]

    # fonction contrainte :
    g, lbg, ubg = constraints(X, Ma, Pt_ancrage_interpolated, dict_fixed_params, Masse_centre)

    # en statique on ne fait pas de boucle sur le temps :
    J = cost_function(
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
    prob = {"f": obj, "x": cas.vertcat(*w), "g": g}
    opts = {"ipopt": {"max_iter": 1000, "linear_solver": "ma57", "nlp_scaling_method": "none"}}
    solver = cas.nlpsol("solver", "ipopt", prob, opts)

    # Solve the NLP
    sol = solver(
        x0=cas.vertcat(*w0), lbx=cas.vertcat(*lbw), ubx=cas.vertcat(*ubw), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg)
    )
    w_opt = sol["x"].full().flatten()
    status = solver.stats()["return_status"]

    return w_opt, Pt_interpolated, F_totale_collecte, ind_masse, labels, Pt_ancrage_interpolated, dict_fixed_params, sol.get("f"), status


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
        7101,
        7120,
        # 7170,
    ]  # This range repends on the trial. To find it, one should use the code plateforme_verification_toutesversions.py.
    dt = 1 / 500  # Hz

    initial_guess = InitialGuessType.GACO

    dict_fixed_params = Param_fixe()
    Fs_totale_collecte, Pts_collecte, labels, ind_masse, Pts_ancrage = get_list_results_dynamic(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )

    ########################################################################################################################

    for idx, frame in enumerate(list(range(jump_frame_index_interval[0], jump_frame_index_interval[1]))):
        trial_name_this_time = f"results_multiple_static_optim_in_a_row/{trial_name}/frame{frame}"
        w_opt, Pt_interpolated, F_totale_collecte, ind_masse, labels, Pt_ancrage_interpolated, dict_fixed_params, cost, status = Optimisation(
            Fs_totale_collecte[idx, :],
            Pts_collecte[idx, :, :],
            labels,
            ind_masse,
            Pts_ancrage[idx, :, :],
            weight,
            initial_guess,
            dict_fixed_params,
            trial_name_this_time,
        )

        Ma = np.array(w_opt[0:5])
        Pt = np.reshape(w_opt[5:-15], (n * m, 3))
        F_athl = np.reshape(w_opt[-15:], (5, 3))
        print(F_athl)

        # ENREGISTREMENT PICKLE#
        if status == "Solve_Succeeded":
            ends_with = "CVG"
        else:
            ends_with = "DVG"
        path = f"results_multiple_static_optim_in_a_row/frame{frame}_{ends_with}.pkl"

        with open(path, "wb") as file:
            data = {"w_opt": w_opt,
                    "Ma": Ma,
                    "Pt": Pt,
                    "F_athl": F_athl,
                    "labels": labels,
                    "Pt_collecte": Pts_collecte[idx, :, :],
                    "Pt_interpolated": Pt_interpolated,
                    "Pt_ancrage": Pts_ancrage[idx, :, :],
                    "Pt_ancrage_interpolated": Pt_ancrage_interpolated,
                    "ind_masse": ind_masse,
                    "cost": cost,
                    "dict_fixed_params": dict_fixed_params,
                    "trial_name": trial_name,
                    "frame": frame,
                    }
            pickle.dump(data, file)

        bt1, bt2, btc1, btc2 = spring_bouts_collecte(Pt.T, Pt_ancrage_interpolated)
        M, F_spring, F_spring_croix, F_masses = static_forces_calc(bt1, bt2, btc1, btc2, dict_fixed_params)
        F_point = static_force_in_each_point(F_spring, F_spring_croix, F_masses)
        F_point[ind_masse, :] += F_athl[0, :]
        F_point[ind_masse + 1, :] += F_athl[1, :]
        F_point[ind_masse - 1, :] += F_athl[2, :]
        F_point[ind_masse + 15, :] += F_athl[3, :]
        F_point[ind_masse - 15, :] += F_athl[4, :]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1.1, 1.8, 1])
        ax.plot(0, 0, -1.2, "ow")

        # ax.plot(Pt_interpolated[0, :],
        #         Pt_interpolated[1, :],
        #         Pt_interpolated[2, :],
        #         "xb",
        #         label="initial guess"
        #         )

        ax.plot(
            Pt_ancrage_interpolated[:, 0],
            Pt_ancrage_interpolated[:, 1],
            Pt_ancrage_interpolated[:, 2],
            "ok",
            mfc="none",
            alpha=0.5,
            markersize=3,
            label="Model Frame",
        )

        ax.plot(
            Pts_ancrage[idx, :, 0],
            Pts_ancrage[idx, :, 1],
            Pts_ancrage[idx, :, 2],
            ".k",
            markersize=3,
            label="Experimental Frame",
        )
        ax.plot(
            Pts_collecte[idx, 0, :],
            Pts_collecte[idx, 1, :],
            Pts_collecte[idx, 2, :],
            ".b",
            markersize=3,
            label="Experimental Trampoline"
        )
        ax.plot(
            Pts_collecte[idx, 0, ind_masse],
            Pts_collecte[idx, 1, ind_masse],
            Pts_collecte[idx, 2, ind_masse],
            ".g",
            markersize=3,
            label="Lowest point on the interval",
        )
        ax.plot(
            Pt[:, 0],
            Pt[:, 1],
            Pt[:, 2],
            "ob",
            mfc="none",
            markersize=3,
            label="Optimized point positions",
        )

        for ind in range(m*n):
            plt.plot(np.vstack((Pt[ind, 0], Pt[ind, 0] + F_point[ind, 0] / 100000)),
                     np.vstack((Pt[ind, 1], Pt[ind, 1] + F_point[ind, 1] / 100000)),
                     np.vstack((Pt[ind, 2], Pt[ind, 2] + F_point[ind, 2] / 100000)),
                     "-r")

        for ind, index in enumerate([ind_masse, ind_masse-1, ind_masse+1, ind_masse-15, ind_masse+15]):
            plt.plot(np.vstack((Pt[index, 0], Pt[index, 0] + F_athl[ind, 0] / 100000)),
                     np.vstack((Pt[index, 1], Pt[index, 1] + F_athl[ind, 1] / 100000)),
                     np.vstack((Pt[index, 2], Pt[index, 2] + F_athl[ind, 2] / 100000)),
                     "-m")

        ax.legend()
        plt.savefig(f"results_multiple_static_optim_in_a_row/solution_frame{frame}_{ends_with}.png")
        plt.show()

    l_repos = dict_fixed_params["l_repos"]
    l_repos_croix = dict_fixed_params["l_repos_croix"]
    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage_interpolated)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    spring_elongation = np.linalg.norm(Spring_bout_2 - Spring_bout_1, axis=1) - l_repos
    spring_croix_elongation = np.linalg.norm(Spring_bout_croix_2 - Spring_bout_croix_1, axis=1) - l_repos_croix
    print(np.sort(spring_elongation))
    print(np.sort(spring_croix_elongation))

    Pt_integres, erreur_relative, erreur_absolue, static_force_in_each_points, v_all = multiple_shooting_integration(
        1, Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params
    )
    print("erreur relative : ", erreur_relative)
    print("erreur absolue : ", erreur_absolue)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(0, 0, -1.2, "ow")

    ax.plot(
        Pt_ancrage_interpolated[:, 0],
        Pt_ancrage_interpolated[:, 1],
        Pt_ancrage_interpolated[:, 2],
        "ok",
        mfc="none",
        alpha=0.5,
        markersize=3,
        label="Model Frame",
    )

    ax.plot(
        Pts_collecte[idx+1, 0, :],
        Pts_collecte[idx+1, 1, :],
        Pts_collecte[idx+1, 2, :],
        ".b",
        markersize=3,
        label="Experimental Trampoline frame + 1"
    )

    ax.plot(
        Pt_integres[:, 0],
        Pt_integres[:, 1],
        Pt_integres[:, 2],
        "ob",
        mfc="none",
        markersize=3,
        label="Integrated point positions",
    )

    ax.legend()
    plt.savefig(f"results_multiple_static_optim_in_a_row/integration_frame{frame}_{ends_with}.png")
    plt.show()



if __name__ == "__main__":
    n = 15
    m = 9
    main()
