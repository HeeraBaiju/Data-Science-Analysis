import math
from scipy.stats import norm


#TODO 1.Testing ⍴ =0(Fisher Transformation) of Normal Populations
#TODO 2.Confidence Interval for ⍴(Population correlation coefficient of Normal Populations


def correlation_normal_population():
    alpha = float(input("Level of significance: "))
    n = float(input("No of samples: "))
    r = float(input("Sample Correlation Coefficient: "))
    test_statistics_value = (math.sqrt(n-3)/2)*math.log((1+r)/(1-r), math.exp(1))

    print(f"\nSelect the alternative Hypothesis to test with null ⍴ = 0")
    print(f"1. ⍴ < 0 ")
    print(f"2. ⍴ > 0")
    print(f"3. ⍴ ≠ 0")

    selection = int(input())

    if selection == 1:
        critical_value = -1*round(norm.isf(alpha, loc=0, scale=1), 4)
        print(f"\nThe null must be rejected if Z<{critical_value}\n")
        print("Calculations\n")
        print(f"Z = {test_statistics_value}\n")
        print("Decision\n")
        if test_statistics_value < critical_value:
            print(f"Null ⍴ = 0 must be Rejected at level of significance {alpha} and Accept ⍴ < 0")
        else:
            print(f"Failure to reject Null ⍴ = 0 ")

    if selection == 2:
        critical_value = round(norm.isf(alpha, loc=0, scale=1), 4)
        print(f"The null must be rejected if Z>{critical_value}\n")
        print("Calculations\n")
        print(f"Z = {test_statistics_value}\n")
        print("Decision\n")
        if test_statistics_value > critical_value:
            print(f"Null ⍴ = 0 must be Rejected at level of significance {alpha} and Accept ⍴ > 0")
        else:
            print(f"Failure to reject Null ⍴ = 0 ")

    if selection == 3:
        critical_value_lower = -1*round(norm.isf(alpha / 2, loc=0, scale=1), 4)
        critical_value_upper = round(norm.isf(alpha / 2, loc=0, scale=1), 4)
        print(f" The null must be rejected if Z<{critical_value_lower} or Z>{critical_value_upper}\n")
        print("Calculations\n")
        print(f"Z = {test_statistics_value}\n")
        print("Decision\n")
        if test_statistics_value > critical_value_upper or test_statistics_value < critical_value_lower:
            print(f"Null ⍴ = 0 must be Rejected at level of significance {alpha} and Accept ⍴ ≠ 0")
        else:
            print(f"Failure to reject Null ⍴ = 0")


def population_interval_for_correlation():
    n = float(input("No of samples: "))
    r = float(input("Sample Correlation Coefficient: "))

    print("\n1.Confidence interval in Percentage")
    print("2.Level of significance given")
    choice = int(input())
    print("\n")
    alpha = 0
    if choice == 1:
        p = float(input("1.Confidence interval in Percentage: "))
        alpha = 1 - p / 100
    else:
        alpha = float(input("Level of significance: "))

    z_ = (1/2)*math.log((1+r)/(1-r), math.exp(1))
    lower_uz = z_ - (round(norm.isf(alpha / 2, loc=0, scale=1), 4) / math.sqrt(n-3))
    upper_uz = z_ + (round(norm.isf(alpha / 2, loc=0, scale=1), 4) / math.sqrt(n - 3))
    lower_r = (math.exp(lower_uz) - math.exp(-1*lower_uz)) / (math.exp(lower_uz) + math.exp(-1*lower_uz))
    upper_r = (math.exp(upper_uz) - math.exp(-1*upper_uz)) / (math.exp(upper_uz) + math.exp(-1*upper_uz))

    print(f"\n{round(lower_r ,4)} < r < {round(upper_r, 4)}")