"""

"""

import pygmo as pg
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
from optim_dynamique_withoutC_casadi import get_list_results_dynamic, Pt_bounds, F_bounds

def cost_function(X, Ma, F_athl, Pt_collecte, Pt_ancrage_interpolated, dict_fixed_params, labels, ind_masse):

    K, _, _ = Param_variable(Ma, ind_masse)
    _, F_point = Calcul_Pt_F(X, Pt_ancrage_interpolated, dict_fixed_params, K, ind_masse, Ma)
    Pt = list2tab(X)
    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage_interpolated)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    spring_elongation = np.linalg.norm(Spring_bout_2 - Spring_bout_1) - dict_fixed_params["l_repos"]
    spring_elongation_croix = np.linalg.norm(Spring_bout_croix_2 - Spring_bout_croix_1) - dict_fixed_params["l_repos_croix"]

    F_point[ind_masse, :] += F_athl[0:3].T
    F_point[ind_masse+1, :] += F_athl[3:6].T
    F_point[ind_masse-1, :] += F_athl[6:9].T
    F_point[ind_masse+15, :] += F_athl[9:12].T
    F_point[ind_masse-15, :] += F_athl[12:15].T

    Pt = list2tab(X)

    Difference = 0
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

    return Difference

def equality_constraints_function(Ma, Masse_centre):
    g = [Ma[0] + Ma[1] + Ma[2] + Ma[3] + Ma[4] - Masse_centre]
    return g

def inequality_constraints_function(X, Pt_ancrage_interpolated, dict_fixed_params):
    Pt = list2tab(X)
    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage_interpolated)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    spring_elongation = - np.linalg.norm(Spring_bout_2 - Spring_bout_1) - dict_fixed_params["l_repos"]
    spring_elongation_croix = - np.linalg.norm(Spring_bout_croix_2 - Spring_bout_croix_1) - dict_fixed_params["l_repos_croix"]
    g = [spring_elongation] + [spring_elongation_croix]
    return g

class global_optimisation:
    """
    This class does a few steps of the global optimization (genetic algorithm) using pygmo.
        - fitness: The function returning the fitness of the solution
        - get_nobj: The function returning the number of objectives
        - get_bounds: The function returning the bounds on the weightings
    """

    def __init__(self, Pt_collecte, Pt_ancrage_interpolated, dict_fixed_params, labels, ind_masse, Masse_centre, Pt_repos, Pt_ancrage_repos):
        self.Pt_collecte = Pt_collecte
        self.Pt_ancrage_interpolated = Pt_ancrage_interpolated
        self.dict_fixed_params = dict_fixed_params
        self.labels = labels
        self.ind_masse = ind_masse
        self.Masse_centre = Masse_centre
        self.Pt_repos = Pt_repos
        self.Pt_ancrage_repos = Pt_ancrage_repos

    def fitness(self, x):
        """
        This function returns how well did the weightings allow to fit the data to track.
        The OCP is solved in this function.
        """
        global i_inverse
        i_inverse += 1
        print(
            f"+++++++++++++++++++++++++++ {i_inverse}th evaluation of the cost function +++++++++++++++++++++++++++"
        )
        Ma = x[0:5]
        X = x[5:n*m*3+5]
        F_athl = x[-15:]

        obj = cost_function(X, Ma, F_athl, self.Pt_collecte, self.Pt_ancrage_interpolated, self.dict_fixed_params, self.labels, self.ind_masse)

        equality_const = equality_constraints_function(Ma, self.Masse_centre)

        inequality_const = inequality_constraints_function(X, self.Pt_ancrage_interpolated, self.dict_fixed_params)

        return [obj, equality_const, inequality_const]

    def get_nobj(self):
        """
        Number of objectives
        """
        return 1

    def get_nic(self):
        """
        Number of inequality constraints
        """
        return 2

    def get_nec(self):
        """
        Number of equality constraints
        """
        return 1

    def get_bounds(self):

        _, lbw_m, ubw_m = m_bounds(self.Masse_centre)

        initial_guess = InitialGuessType.SURFACE_INTERPOLATION
        _, lbw_Pt, ubw_Pt, _, _ = Pt_bounds(initial_guess, self.Pt_collecte,
                                        self.Pt_ancrage_interpolated, self.Pt_repos,
                                        self.Pt_ancrage_repos, self.dict_fixed_params)
        lbw_Pt = [lbw_Pt[i][j] for i in range(n*m) for j in range(3)]
        ubw_Pt = [ubw_Pt[i][j] for i in range(n*m) for j in range(3)]

        _, lbw_F, ubw_F = F_bounds()
        lbw_F = list(lbw_F)
        ubw_F = list(ubw_F)

        lbx = lbw_m + lbw_Pt + lbw_F
        ubx = ubw_m + ubw_Pt + ubw_F

        return (lbx, ubx)

    def gradient(self, x):
        grad = pg.estimate_gradient_h(lambda x: self.fitness(x), x)
        return grad

def solve(prob):

    global i_inverse
    i_inverse = 0

    algo = pg.algorithm(pg.simulated_annealing())
    pop = pg.population(prob, size=100)

    epsilon = 1e-8
    diff = 10000
    w_opt = None
    while i_inverse < 100 and diff > epsilon:
        olf_pop_f = np.min(pop.get_f())
        pop = algo.evolve(pop)
        new_pop_f = np.min(pop.get_f())
        diff = olf_pop_f - new_pop_f
        w_opt = pop.get_x()[np.argmin(pop.get_f())]

    return w_opt, new_pop_f


##########################################################################################################################
def main():

    global i_inverse

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

    dict_fixed_params = Param_fixe()
    Fs_totale_collecte, Pts_collecte, labels, ind_masse, Pts_ancrage = get_list_results_dynamic(
        participant, static_trial_name, empty_trial_name, trial_name, jump_frame_index_interval
    )

    ########################################################################################################################

    for idx, frame in enumerate(list(range(jump_frame_index_interval[0], jump_frame_index_interval[1]))):

        Pt_ancrage_repos, Pt_repos = Points_ancrage_repos(dict_fixed_params)
        initial_guess = InitialGuessType.SURFACE_INTERPOLATION
        _, _, _, Pt_interpolated, Pt_ancrage_interpolated = Pt_bounds(initial_guess,
                                                                      Pts_collecte[idx, :, :],
                                                                      Pts_ancrage[idx, :, :],
                                                                      Pt_repos,
                                                                      Pt_ancrage_repos,
                                                                      dict_fixed_params)

        prob = pg.problem(global_optimisation(
            Pts_collecte[idx, :, :],
            Pt_ancrage_interpolated,
            dict_fixed_params,
            labels,
            ind_masse,
            weight,
            Pt_repos,
            Pt_ancrage_repos,
        ))
        w_opt, cost = solve(prob)

        Ma = np.array(w_opt[0:5])
        Pt = np.reshape(w_opt[5:-15], (n * m, 3))
        F_athl = np.reshape(w_opt[-15:], (5, 3))
        print(F_athl)
        print(i_inverse)

        path = f"results_multiple_static_optim_in_a_row/frame{frame}_SA.pkl"
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
