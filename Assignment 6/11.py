#------------------------------------------------------------
# Importing all libraries
from cmath import pi
from uncertainties import ufloat
#------------------------------------------------------------
# Data
Einsteins_theory = 1.74 # arc-seconds (Model-1)
Newtonian_gravity = Einsteins_theory/2 # arc-seconds (Model-2)
Eddington = ufloat(1.61, 0.40) # arc-seconds (Model-1)
Crommelin = ufloat(1.98, 0.16) # arc-seconds (Model-2)
#------------------------------------------------------------
# Probabilities
Prob_Einsteins_theory = Einsteins_theory/pi
Prob_Newtonian_gravity = Newtonian_gravity/pi
Prob_Eddington = Eddington/pi
Prob_Crommelin = Crommelin/pi
#------------------------------------------------------------
# Bayes factor
Bayes_factor = (Prob_Crommelin/Prob_Eddington) *(Prob_Newtonian_gravity/Prob_Einsteins_theory)
print('Bayes factor: {}' .format(Bayes_factor))
