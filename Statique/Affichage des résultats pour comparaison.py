'''
Permet de comparer les valeurs des raideurs

Code permettant de récuperer les donnees des fichiers pickle issus des tests pour chaque essai individuel
Permet de traiter les donnees : k, m , pt, f ...

Finalement on fait un multistart avec des C.I entre le 1 et le 3e quartile
on importe les résultats et on choisit les meilleurs k
'''

#IMPORT
import pickle
import matplotlib.pyplot as plt
import numpy as np


######DONNEES DES ESSAIS INDIVIDUELS######
#initialisation des variables
k = []
M = []
Pt = []
labels = []
Pt_collecte = []
Pt_ancrage = []
ind_masse = []
erreur_position = []
erreur_force = []
f = []

#labeled statique
for i in range (1,12):
    file_name = 'labeled_statique_D' + str(i)
    base = '/home/lim/Documents/Jules/statique/results/result_1essai_regk*e-7_diff*1000/'
    path_name = str(base) + str(file_name) + '.pkl'

    #chargement des donnees
    with open(path_name, 'rb') as file:
        k.append(pickle.load(file))
        M.append(pickle.load(file))
        Pt.append(pickle.load(file))
        labels.append(pickle.load(file))
        Pt_collecte.append(pickle.load(file))
        Pt_ancrage.append(pickle.load(file))
        ind_masse.append(pickle.load(file))
        erreur_position.append(pickle.load(file))
        erreur_force.append(pickle.load(file))
        f.append(pickle.load(file))
#left
for i in range (1,12):
    file_name = 'labeled_statique_left_D' + str(i)
    base = '/home/lim/Documents/Jules/statique/results/result_1essai_regk*e-7_diff*1000/'
    path_name = str(base) + str(file_name) + '.pkl'

    #chargement des donnees
    with open(path_name, 'rb') as file:
        k.append(pickle.load(file))
        M.append(pickle.load(file))
        Pt.append(pickle.load(file))
        labels.append(pickle.load(file))
        Pt_collecte.append(pickle.load(file))
        Pt_ancrage.append(pickle.load(file))
        ind_masse.append(pickle.load(file))
        erreur_position.append(pickle.load(file))
        erreur_force.append(pickle.load(file))
        f.append(pickle.load(file))
#centrefront
for i in range (1,12):
    file_name = 'labeled_statique_centrefront_D' + str(i)
    base = '/home/lim/Documents/Jules/statique/results/result_1essai_regk*e-7_diff*1000/'
    path_name = str(base) + str(file_name) + '.pkl'

    #chargement des donnees
    with open(path_name, 'rb') as file:
        k.append(pickle.load(file))
        M.append(pickle.load(file))
        Pt.append(pickle.load(file))
        labels.append(pickle.load(file))
        Pt_collecte.append(pickle.load(file))
        Pt_ancrage.append(pickle.load(file))
        ind_masse.append(pickle.load(file))
        erreur_position.append(pickle.load(file))
        erreur_force.append(pickle.load(file))
        f.append(pickle.load(file))
#leftfront
for i in range (1,12):
    file_name = 'labeled_statique_leftfront_D' + str(i)
    base = '/home/lim/Documents/Jules/statique/results/result_1essai_regk*e-7_diff*1000/'
    path_name = str(base) + str(file_name) + '.pkl'

    #chargement des donnees
    with open(path_name, 'rb') as file:
        k.append(pickle.load(file))
        M.append(pickle.load(file))
        Pt.append(pickle.load(file))
        labels.append(pickle.load(file))
        Pt_collecte.append(pickle.load(file))
        Pt_ancrage.append(pickle.load(file))
        ind_masse.append(pickle.load(file))
        erreur_position.append(pickle.load(file))
        erreur_force.append(pickle.load(file))
        f.append(pickle.load(file))

##############################################################################################

#moyenne k et des f des essais individuels

k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = [], [], [], [], [], [], [], [], [], [], [], []
K = []
for i in range (0,44):
    k1.append(k[i][0])
    k2.append(k[i][1])
    k3.append(k[i][2])
    k4.append(k[i][3])
    k5.append(k[i][4])
    k6.append(k[i][5])
    k7.append(k[i][6])
    k8.append(k[i][7])
    k9.append(k[i][8])
    k10.append(k[i][9])
    k11.append(k[i][10])
    k12.append(k[i][11])
K += (k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12) #Grande liste contenant tous les k1, puis tous les k2..

moyenne = []
et = []
mediane = []
quartile = []
quartile2 = []
moyennef = (np.mean(f))
for i in range (0,12):
    moyenne.append(np.mean(K[i][:]))
    et.append(np.std(K[i]))
    mediane.append(np.median(K[i][:]))
    quartile.append(np.quantile(K[i][:], 0.25))
    quartile2.append(np.quantile(K[i][:], 0.75))
    print('Moyenne des k' + str(i+1) + ' : ' + str(moyenne[i]) + ' // ' + 'Médiane des k' + str(i + 1) + ' : ' + str(mediane[i]) + ' // ' + '1er quartile ' + str(i+1) + ' : ' + str(quartile[i]) + ' // ' + '3e quartile ' + str(i+1) + ' : ' + str(quartile2[i]) + ' // ' + 'ecart type ' + str(i+1) + ' : ' + str(et[i]))


####################################################################################################
####################################################################################################

'''AFFICHAGE DES RESULTATS'''

#essais individuels
liste_k_name = ['k1 coin horizontal', 'k2 cadre/toile horizontal', 'k3 coin vertical', 'k4 cadre/toile vertical', 'k5 bord horizontal', 'k6 horizontal', 'k7 bord vertical', 'k8 vertical', 'k1ob coin', 'k2ob bord vertical', 'k3ob bord horizontal', 'k4ob autre']
for i in range(0,11):
    if i==1:
        plt.plot(liste_k_name, k[i][:], 'xb', markersize=5, label = 'Zone : statique')
    else:
        plt.plot(liste_k_name, k[i][:], 'xb', markersize=5)
for i in range (11,22):
    if i == 11:
        plt.plot(liste_k_name, k[i][:], 'xg', markersize=5, label='Zone : left')
    else:
        plt.plot(liste_k_name, k[i][:], 'xg', markersize=5)
for i in range (22,33):
    if i == 22:
        plt.plot(liste_k_name, k[i][:], 'xc', markersize=5, label='Zone : centrefront')
    else:
        plt.plot(liste_k_name, k[i][:], 'xc', markersize=5)
for i in range (33,44):
    if i == 33:
        plt.plot(liste_k_name, k[i][:], 'xy', markersize=5, label='Zone : leftfront')
    else:
        plt.plot(liste_k_name, k[i][:], 'xy', markersize=5)

##############################################################################################
##############################################################################################
"""Recuperation des donnees des simulations avec plusieurs essais"""

Solution_quartile1 = []
labels_quartile1 = []
Pt_collecte_quartile1 = []
Pt_ancrage_quartile1 = []
ind_masse_quartile1 = []
err_position_quartile1 = []
err_force_quartile1 = []
dict_fixed_params_quartile1 = {}
path = '/home/lim/Documents/Jules/statique/results/result_multi_essais/optim_sur_35_essais_regkob_quartile1.pkl'
with open(path, 'rb') as file:
    Solution_quartile1.append(pickle.load(file))
    labels_quartile1.append(pickle.load(file))
    Pt_collecte_quartile1.append(pickle.load(file))
    Pt_ancrage_quartile1.append(pickle.load(file))
    ind_masse_quartile1.append(pickle.load(file))
    err_position_quartile1.append(pickle.load(file))
    err_force_quartile1.append(pickle.load(file))
    f_quartile1 = pickle.load(file)
    dict_fixed_params_quartile1 = pickle.load(file)

k_quartile1 = Solution_quartile1[0][:12]
#print(k_quartile1)

Solution_quartile3 = []
labels_quartile3 = []
Pt_collecte_quartile3 = []
Pt_ancrage_quartile3= []
ind_masse_quartile3 = []
err_position_quartile3 = []
err_force_quartile3 = []
dict_fixed_params_quartile3 = {}
path = '/home/lim/Documents/Jules/statique/results/result_multi_essais/optim_sur_35_essais_quartile3.pkl'
with open(path, 'rb') as file:
    Solution_quartile3.append(pickle.load(file))
    labels_quartile3.append(pickle.load(file))
    Pt_collecte_quartile3.append(pickle.load(file))
    Pt_ancrage_quartile3.append(pickle.load(file))
    ind_masse_quartile3.append(pickle.load(file))
    err_position_quartile3.append(pickle.load(file))
    err_force_quartile3.append(pickle.load(file))
    f_quartile3 = pickle.load(file)
    dict_fixed_params_quartile3 = pickle.load(file)
k_quartile3 = Solution_quartile3[0][:12]

Solution_35essais = []
labels_35essais = []
Pt_collecte_35essais = []
Pt_ancrage_35essais= []
ind_masse_35essais = []
err_position_35essais = []
err_force_35essais = []
dict_fixed_params_35essais = {}
path = '/home/lim/Documents/Jules/statique/results/result_multi_essais/optim_sur_35_essais_regkob_mediane.pkl'
with open(path, 'rb') as file:
    Solution_35essais.append(pickle.load(file))
    labels_35essais.append(pickle.load(file))
    Pt_collecte_35essais.append(pickle.load(file))
    Pt_ancrage_35essais.append(pickle.load(file))
    ind_masse_35essais.append(pickle.load(file))
    err_position_35essais.append(pickle.load(file))
    err_force_35essais.append(pickle.load(file))
    f_35essais = pickle.load(file)
    dict_fixed_params_35essais = pickle.load(file)

k_35essais = Solution_35essais[0][:12]
#print(k_35essais)

###################################################################################################
###################################################################################################
"""Affichage des resulttats"""

#plusieurs essais
plt.plot(liste_k_name, mediane, '_r', markersize = 15 ,label = 'Médiane essais individuels')
# plt.plot(liste_k_name, quartile, 'r', markersize = 1 ,label = 'premier quartile - valeurs initiales k')
#plt.plot(liste_k_name, k_quartile1, 'ob', markersize = 6 , label = '1er quartile')
# plt.plot(liste_k_name, k_quartile3, 'om', markersize = 6, label = '3e quartile')
# plt.plot(liste_k_name, k_35essais, 'ok', markersize = 6, label = 'k issues de l optimisation des 35 essais, initialisés  a la médiane')


###################################################################################################
###################################################################################################
"""Recuperation des donnees du multistart"""

Solution_multi = []
labels_multi = []
Pt_collecte_multi = []
Pt_ancrage_multi= []
ind_masse_multi = []
err_position_multi = []
err_force_multi = []
f_multi = []
dict_fixed_params_multi = []
trial_name_multi = []
temps_multi = []
init_k_multi = []
statu_multi = []

path = '/home/lim/Documents/Jules/statique/results/result_multistart/'
for i in range (0,57):
    essai = 'optim_numero' + str(i) + '.pkl'
    name = path + essai
    with open(name, 'rb') as file:
        Solution_multi.append(pickle.load(file))
        labels_multi.append(pickle.load(file))
        Pt_collecte_multi.append(pickle.load(file))
        Pt_ancrage_multi.append(pickle.load(file))
        ind_masse_multi.append(pickle.load(file))
        err_position_multi.append(pickle.load(file))
        err_force_multi.append(pickle.load(file))
        f_multi.append(pickle.load(file))
        dict_fixed_params_multi.append(pickle.load(file))
        trial_name_multi.append(pickle.load(file))
        temps_multi.append(pickle.load(file))
        init_k_multi.append(pickle.load(file))
        statu_multi.append(pickle.load(file))

f_ok=[]
for i in range(0,len(statu_multi)):
    if (statu_multi[i] == 'Solve_Succeeded' or statu_multi[i] == 'Solved_To_Acceptable_Level') and f_multi[i] < f_35essais:
        print(f_multi[i])
        print('   indice : ' + str(i))
        #print(i)

# plt.plot(liste_k_name, Solution_multi[3][0:12], 'ob', markersize = 6, label = 'meilleurs k multistart')

#k
# for i in range (0,12):
#     print('Médiane des k' + str(i + 1) + ' : ' + str(mediane[i]) + ' / / ' + 'optim 35 k' + str(i + 1) + ' : ' + str(k_35essais[i]))

#valeur fonction objectif
# print('moyenne objectif individuels : ' + str(moyennef))
# print('fonction objectif 1 quartile : ' + str(f_quartile1))
# print('fonction objectif 3 quartile : ' + str(f_quartile3))
print('fonction objectif 35 essais : ' + str(f_35essais))
print('fonction objectif best multistart : ' + str(f_multi[3]))

plt.title('Comparaison des k')
plt.xlabel('type de ressort')
plt.xticks(fontsize = 7)
plt.ylabel('raideurs obtenues')
print(Solution_multi[3][:12])

plt.legend()
plt.show()
