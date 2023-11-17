"""

"""

import pygmo as pg
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import sys

sys.path.append("../")
from enums import InitialGuessType

sys.path.append("../Statique/casadi/")
from Optim_35_essais_kM_regul_koblique import Param_fixe, list2tab, Spring_bouts, Spring_bouts_croix, tab2list
from modele_dynamique_nxm_DimensionsReelles import (
    Points_ancrage_repos,
    multiple_shooting_integration,
)
from Optim_multi_essais_kM_regul_koblique import m_bounds, k_bounds
from optim_dynamique_withoutC_casadi import get_list_results_dynamic, Pt_bounds, F_bounds
from iterative_stabilisation import position_the_points_based_on_the_force

def cost_function(Ma, F_athl, K, Pt_collecte, Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params, labels, ind_masse, WITH_K_OBLIQUE):

    n = 15
    m = 9

    Pt, F_point_after_step = position_the_points_based_on_the_force(Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params, Ma, F_athl, K, ind_masse, WITH_K_OBLIQUE, PLOT_FLAG=True)

    Difference = 0
    for i in range(3):
        for ind in range(n * m):
            if "t" + str(ind) in labels:
                ind_collecte = labels.index("t" + str(ind))
                if not np.isnan(Pt_collecte[i, ind_collecte]):
                    Difference += 500 * (Pt[ind, i] - Pt_collecte[i, ind_collecte]) ** 2

    F_total = np.linalg.norm(F_point_after_step.sum(axis=0))
    Difference += F_total / 100000

    return Difference

def equality_constraints_function(Ma, Masse_centre):
    """
    The total mass must be the athlete weight.
    """
    g = [Ma[0] + Ma[1] + Ma[2] + Ma[3] + Ma[4] - Masse_centre]
    return g

def inequality_constraints_function(F_athl, F_collecte):
    """
    The total force must be similar to the force measured on the trampoline.
    g <= 0
    """
    norm_athl = np.linalg.norm(np.sum(F_athl.reshape(5, 3), axis=0))
    norm_collecte = np.linalg.norm(F_collecte)
    g = [norm_athl - norm_collecte * 1.5, norm_collecte * 0.5 - norm_athl]
    return g

def k_bounds(WITH_K_OBLIQUE):
    # RESULTS FROM THE STATIC OPTIMIZATION
    k1 = 1.21175669e05
    k2 = 3.20423906e03
    k3 = 4.11963416e03
    k4 = 2.48125477e03
    k5 = 7.56820743e03
    k6 = 4.95811865e05
    k7 = 1.30776275e-03
    k8 = 3.23131678e05
    k_oblique1 = 7.48735556e02
    k_oblique2 = 1.08944449e-04
    k_oblique3 = 3.89409909e03
    k_oblique4 = 1.04226031e-04

    w0_k = [k1, k2, k3, k4, k5, k6, k7, k8]
    if WITH_K_OBLIQUE:
        w0_k += [k_oblique1, k_oblique2, k_oblique3, k_oblique4]

    lbw_k = [1e2] * len(w0_k)
    ubw_k = [1e5] * len(w0_k)

    return w0_k, lbw_k, ubw_k

class global_optimisation:
    """
    This class does a few steps of the global optimization (genetic algorithm) using pygmo.
        - fitness: The function returning the fitness of the solution
        - get_nobj: The function returning the number of objectives
        - get_bounds: The function returning the bounds on the weightings
    """

    def __init__(self, Pt_collecte, Pt_interpolated, Pt_ancrage_interpolated, dict_fixed_params, labels, ind_masse, Masse_centre, Pt_repos, Pt_ancrage_repos, F_collecte, WITH_K_OBLIQUE):
        self.Pt_collecte = Pt_collecte
        self.Pt_interpolated = Pt_interpolated
        self.Pt_ancrage_interpolated = Pt_ancrage_interpolated
        self.dict_fixed_params = dict_fixed_params
        self.labels = labels
        self.ind_masse = ind_masse
        self.Masse_centre = Masse_centre
        self.Pt_repos = Pt_repos
        self.Pt_ancrage_repos = Pt_ancrage_repos
        self.F_collecte = F_collecte
        self.WITH_K_OBLIQUE = WITH_K_OBLIQUE

    def fitness(self, x):
        """
        This function returns how well did the weightings allow to fit the data to track.
        The OCP is solved in this function.
        """

        Ma = x[0:5]
        if self.F_collecte is not None:
            K = x[5:-15]
            F_athl = x[-15:]
        else:
            F_athl = None
            K = x[5:]

        Difference = cost_function(Ma, F_athl, K, self.Pt_collecte, self.Pt_interpolated, self.Pt_ancrage_interpolated, self.dict_fixed_params, self.labels, self.ind_masse, self.WITH_K_OBLIQUE)
        obj = [Difference]

        equality_const = equality_constraints_function(Ma, self.Masse_centre)

        inequality_const = []
        if self.F_collecte is not None:
            inequality_const = inequality_constraints_function(F_athl, -self.F_collecte)

        return obj + equality_const + inequality_const

    def get_nobj(self):
        """
        Number of objectives
        """
        return 1

    def get_nec(self):
        """
        Number of equality constraints
        """
        return 1

    def get_nic(self):
        """
        Number of equality constraints
        """
        return 2 if self.F_collecte is not None else 0

    def get_bounds(self):

        _, lbw_m, ubw_m = m_bounds(self.Masse_centre, None, None)

        _, lbw_k, ubw_k = k_bounds(self.WITH_K_OBLIQUE)

        lbx = lbw_m + lbw_k
        ubx = ubw_m + ubw_k

        if self.F_collecte is not None:
            _, lbw_F, ubw_F = F_bounds(None, None)
            lbw_F = list(lbw_F)
            ubw_F = list(ubw_F)
            lbx += lbw_F
            ubx += ubw_F

        return (lbx, ubx)

    def gradient(self, x):
        grad = pg.estimate_gradient_h(lambda x: self.fitness(x), x)
        return grad

def solve(prob, global_optim):

    bfe = True
    seed = 42
    pop_size = 500
    num_gen = 150

    algo = pg.gaco()
    if bfe:
        algo.set_bfe(pg.bfe())
    algo = pg.algorithm(algo)  #pg.ihs
    bfe_pop = pg.default_bfe() if bfe else None
    pop = pg.population(prob=prob, size=pop_size, seed=seed)

    # Initialize lists with the best individual per generation
    list_of_champion_f = [pop.champion_f]
    list_of_champion_x = [pop.champion_x]
    for i in range(num_gen):
        print(f'Evolution: {i + 1} / {num_gen} \n')
        pop = algo.evolve(pop)
        list_of_champion_x.append(pop.champion_x)
        list_of_champion_f.append(pop.champion_f)


        w_current = pop.champion_x
        Ma = np.array(w_current[0:5])
        K = np.array(w_current[5:-15])
        F_athl = np.reshape(w_current[-15:], (5, 3))
        print(f"{i}th generation: K = {K}, cost = {pop.champion_f}")
        # Pt = position_the_points_based_on_the_force(global_optim.Pt_interpolated,
        #                                             global_optim.Pt_ancrage_interpolated,
        #                                             global_optim.dict_fixed_params,
        #                                             Ma,
        #                                             F_athl,
        #                                             K,
        #                                             global_optim.ind_masse)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # ax.set_box_aspect([1.1, 1.8, 1])
        # ax.plot(0, 0, -1.2, "ow")
        #
        # ax.plot(global_optim.Pt_interpolated[0, :],
        #         global_optim.Pt_interpolated[1, :],
        #         global_optim.Pt_interpolated[2, :],
        #         "xb",
        #         label="initial guess"
        #         )
        # ax.plot(
        #     global_optim.Pt_ancrage_interpolated[:, 0],
        #     global_optim.Pt_ancrage_interpolated[:, 1],
        #     global_optim.Pt_ancrage_interpolated[:, 2],
        #     "ok",
        #     mfc="none",
        #     alpha=0.5,
        #     markersize=3,
        #     label="Model Frame",
        # )
        # ax.plot(
        #     global_optim.Pt_collecte[0, :],
        #     global_optim.Pt_collecte[1, :],
        #     global_optim.Pt_collecte[2, :],
        #     ".b",
        #     markersize=3,
        #     label="Experimental Trampoline"
        # )
        # ax.plot(
        #     Pt[:, 0],
        #     Pt[:, 1],
        #     Pt[:, 2],
        #     "ob",
        #     mfc="none",
        #     markersize=3,
        #     label="Optimized point positions",
        # )
        #
        # for ind in range(m*n):
        #     plt.plot(np.vstack((Pt[ind, 0], Pt[ind, 0] + F_point[ind, 0] / 100000)),
        #              np.vstack((Pt[ind, 1], Pt[ind, 1] + F_point[ind, 1] / 100000)),
        #              np.vstack((Pt[ind, 2], Pt[ind, 2] + F_point[ind, 2] / 100000)),
        #              "-r")
        #
        # for ind, index in enumerate([ind_masse, ind_masse-1, ind_masse+1, ind_masse-15, ind_masse+15]):
        #     plt.plot(np.vstack((Pt[index, 0], Pt[index, 0] + F_athl[ind, 0] / 100000)),
        #              np.vstack((Pt[index, 1], Pt[index, 1] + F_athl[ind, 1] / 100000)),
        #              np.vstack((Pt[index, 2], Pt[index, 2] + F_athl[ind, 2] / 100000)),
        #              "-m")
        #
        # ax.legend()
        # plt.savefig(f"results_multiple_static_optim_in_a_row/{trial_name}/solution_frame{frame}_global_k.png")
        # plt.show()

    print('Evolution finished')

    f_opt = np.min(list_of_champion_f)
    best_champion_idx = np.where(list_of_champion_f == f_opt)[0][0]
    w_opt = list_of_champion_x[best_champion_idx]

    return w_opt, f_opt


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
        7101,
        7120,
        # 7170,
    ]  # This range repends on the trial. To find it, one should use the code plateforme_verification_toutesversions.py.
    dt = 1 / 500  # Hz

    # if trial_name is not a folder, create it
    if not os.path.isdir(f"results_multiple_static_optim_in_a_row/{trial_name}"):
        os.mkdir(f"results_multiple_static_optim_in_a_row/{trial_name}")


    dict_fixed_params = Param_fixe()
    Fs_totale_collecte, Pts_collecte, labels, ind_masse, Pts_ancrage = get_list_results_dynamic(
        participant, empty_trial_name, trial_name, jump_frame_index_interval
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
                                                                      dict_fixed_params,
                                                                      trial_name)

        global_optim = global_optimisation(
                    Pts_collecte[idx, :, :],
                    Pt_interpolated,
                    Pt_ancrage_interpolated,
                    dict_fixed_params,
                    labels,
                    ind_masse,
                    weight,
                    Pt_repos,
                    Pt_ancrage_repos,
                    Fs_totale_collecte[idx, :],
                    WITH_K_OBLIQUE=False,
                )
        prob = pg.problem(global_optim)
        w_opt, cost = solve(prob, global_optim)

        Ma = np.array(w_opt[0:5])
        K = np.array(w_opt[5:-15])
        F_athl = np.reshape(w_opt[-15:], (5, 3))
        print(F_athl)

        path = f"results_multiple_static_optim_in_a_row/{trial_name}/frame{frame}_gaco.pkl"
        with open(path, "wb") as file:
            data = {"w_opt": w_opt,
                    "Ma": Ma,
                    "K": K,
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

        embed()


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
    plt.savefig(f"results_multiple_static_optim_in_a_row/{trial_name}/integration_frame{frame}_gaco.png")
    plt.show()



if __name__ == "__main__":
    n = 15
    m = 9
    main()
