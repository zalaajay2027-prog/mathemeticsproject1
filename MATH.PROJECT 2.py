import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
S = 200

df = pd.DataFrame({
    "Study_hours": np.random.randint(1, 15, S),
    "Attendance": np.random.randint(50, 100, S),
    "Group_discussion": np.random.choice(["Yes", "No"], S),
    "Previous_test_score": np.random.randint(20, 100, S)
})

df["Final_exam_pass"] = np.where(
    (df["Study_hours"] > 7) &
    (df["Attendance"] > 75) &
    (df["Previous_test_score"] > 50),
    "Pass", "Fail"
)

print("\n***** Dataset Preview *****")
print(df.head())

P_pass = (df["Final_exam_pass"] == "Pass").mean()
print("\nProbability of Passing:", P_pass)


P_high_study = (df["Study_hours"] > 10).mean()
P_high_attendance = (df["Attendance"] > 80).mean()
P_group_yes = (df["Group_discussion"] == "Yes").mean()

print("\nEvent Probabilities:")
print("P(Study > 10):", P_high_study)
print("P(Attendance > 80):", P_high_attendance)
print("P(Group Discussion = Yes):", P_group_yes)


P_Empirical = (df["Study_hours"] > 10).mean()

P_Theoretical = 4 / 14   

print("\nEmpirical Probability:", P_Empirical)
print("\nTheoretical Probability:", P_Theoretical)


p = P_pass

prob_dist = {
    0: (1 - p) ** 3,
    1: 3 * p * (1 - p) ** 2,
    2: 3 * p**2 * (1 - p),
    3: p**3
}

dist_df = pd.DataFrame({
    "X": list(prob_dist.keys()),
    "P(X)": list(prob_dist.values())
})

print("\n***** Probability Distribution *****")
print(dist_df)

Mean = sum(x * prob_dist[x] for x in prob_dist)
Variance = sum((x - Mean) ** 2 * prob_dist[x] for x in prob_dist)

print("\nMean:", Mean)
print("Variance:", Variance)

A = df["Study_hours"] > 10
B = df["Attendance"] > 80

Pro_A = A.mean()
Pro_B = B.mean()
Pro_A_andpro_B = (A & B).mean()

print("\nVenn Probabilities:")
print("Pro_A:", Pro_A)
print("Pro_B:", Pro_B)
print("P(A ∩ B):", Pro_A_andpro_B)


Contingency_Table = pd.crosstab(df["Group_discussion"], df["Final_exam_pass"])
print("\n***** Contingency Table *****")
print(Contingency_Table)

Total = len(df)

P_joint = Contingency_Table.loc["Yes", "Pass"] / Total
P_pass_marginal = (df["Final_exam_pass"] == "Pass").mean()
P_conditional = Contingency_Table.loc["Yes", "Pass"] / Contingency_Table.loc["Yes"].sum()

print("\nJoint Probability P(Yes AND Pass):", P_joint)
print("Marginal Probability P(Pass):", P_pass_marginal)
print("Conditional Probability P(Pass | Yes):", P_conditional)


P_yes = (df["Group_discussion"] == "Yes").mean()

print("\n ***** Check Independence *****")
print("P(Pass|Yes):", P_conditional)
print("P(Pass):", P_pass_marginal)

if abs(P_conditional - P_pass_marginal) < 0.05:
    print("Approximately Independent")
else:
    print("Dependent Events")

P_high_attendance = 0.6
P_high_attendance_pass = 0.7
P_high_attendance_fail = 0.4

P_fail = 1 - P_pass


P_pass_high_attendance = (P_high_attendance_pass * P_pass) / P_high_attendance

print("\n ***** Bayes Result *****")
print("P(Pass | High Attendance):", P_pass_high_attendance)


plt.figure()
sns.boxplot(x="Final_exam_pass", y="Study_hours", data=df)
plt.title("Study Hours vs Pass/Fail")
plt.show()


plt.figure()
sns.histplot(data=df, x="Attendance", hue="Final_exam_pass", kde=True)
plt.title("Attendance Distribution")
plt.show()


plt.figure()
sns.scatterplot(x="Previous_test_score", y="Study_hours",
                hue="Final_exam_pass", data=df)
plt.title("Score vs Study Hours")
plt.show()


plt.figure()
sns.countplot(x="Final_exam_pass", data=df)
plt.title("Pass vs Fail Count")
plt.show()


plt.figure()
sns.countplot(x="Group_discussion", hue="Final_exam_pass", data=df)
plt.title("Group Discussion vs Result")
plt.show()


