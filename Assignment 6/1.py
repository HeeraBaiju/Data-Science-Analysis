import statistics as statistics
from uncertainties import ufloat
from uncertainties.umath import *  
from cmath import pi



Eddington = ufloat(1.61, 0.40)
Crommelin = ufloat(1.98, 0.16)
ProbabilityEddingtonvalue = Eddington/pi
ProbabilityCrommelinvalue = Crommelin/pi
Priorodds = ProbabilityEddingtonvalue/ProbabilityCrommelinvalue
print('The value of prior odds = ', Priorodds)

PH1givenD_Einstein = 1.74/pi
PH0givenD_Newton = 0.87/pi
Posteriorodds = PH0givenD_Newton/PH1givenD_Einstein
print( 'The value of posterior odds, ', Posteriorodds)

Bayesfactor = Posteriorodds/Priorodds
print('the value of Bayes factor is ', Bayesfactor)
