
This repository aims to fit a trampoline model to experimental data obtained using motion capture and force plates. 
The code is linked with an article (see [in the writting] for more details). *** Add pictures of the data colletion + results of the fitted model***.

### Installation ####
conda install -c conda-forge ezc3d pygmo casadi

### How it works ###
1. Global optimization (using pygmo/gaco) of the elasticity coefficients (k) of the model's springs with multiple static trials. [Optim_multi_essais_kM.py](Statique/iterative_stabilisation/Optim_multi_essais_kM.py)
2. Local optimization (using casadi/ipopt) to refine the elasticity coefficients with the results from 1. as the initial guess. [...](...)
3. Global optimization of the damping coefficients (due to air resistance) whiile keeping the elasticity coefficients from 2. with multiple dynamic trials. [Optim_multi_essais_kM.py](Dynamique/iterative_stabilisation/Optim_multi_essais_kM.py)
4. Local optimization to refine the damping coefficients with the results from 3. as the initial guess. [...](...)

The model is then validated using the test pool of static and dynamic trials by measuring the distance between the predicted mass positions and the experimental markers. 
- [...](...) 
- 
The figures presented in the article are generated here [...](...)?



! The results from the folder Statique/results/ (Jule's results) should be deleted !
