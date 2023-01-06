"""
Ce code permet de calculer la vraie erreur de position entre les points collectés et les points simulés
il s'agit d'une erreur moyenne calculée a partir des points de la toile collecté
moy = (somme distance entre pt coll et pt simu) / (nb de point coll)

il s'agit d'une erreur absolue
"""


import pickle
import numpy as np

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

#Calcul vraie erreur
normes = []
for essai in range (0,11):
    norme = []
    for ligne in range (0,135):
        if 't' + str(ligne) in labels[essai]:
            ind_collecte = labels[essai].index('t' + str(ligne))
            if (np.isnan(np.max(Pt_collecte[essai][:, ind_collecte])))== False:
                norme.append(np.sqrt((Pt[essai][ligne,0]-Pt_collecte[essai][0, ind_collecte])**2 + (Pt[essai][ligne,1]-Pt_collecte[essai][1, ind_collecte])**2 + (Pt[essai][ligne,2]-Pt_collecte[essai][2, ind_collecte])**2))
    normes.append(norme)

em = []
for i in range(0,11):
    erreur_position_moyenne =np.sum(normes[i])/len(normes[i])
    em.append(erreur_position_moyenne)
    print('erreur de position moyenne pour l essai D' + str(i) + ' : ' + str(erreur_position_moyenne) + ' m, calculée sur ' + str(len(normes[i])) + ' points')

print(normes)
