import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

np.random.seed(42)

n = 150

data = {
    "Household_ID": [f"H{str(i).zfill(4)}" for i in range(1, n+1)],
    "Age_of_Household_Head": np.random.randint(25, 70, n),
    "Household_Income": np.random.randint(10000, 100000, n),
    "Education_Level": np.random.choice(["Primary", "Secondary", "Graduate", "Post-Graduate"], n),
    "Family_Size": np.random.randint(1, 8, n),
    "Owns_House": np.random.choice(["Yes", "No"], n),
    "Urban_Rural": np.random.choice(["Urban", "Rural"], n)
}

df = pd.DataFrame(data)

print(df.head())

print("Mean:\n", df[['Age_of_Household_Head','Household_Income']].mean())
print("\nMedian:\n", df[['Age_of_Household_Head','Household_Income']].median())
print("\nMode:\n", df[['Age_of_Household_Head','Household_Income']].mode().iloc[0])

income = df['Household_Income']

print("Range:", income.max() - income.min())
print("Variance:", income.var())
print("Standard Deviation:", income.std())

Q1 = income.quantile(0.25)
Q3 = income.quantile(0.75)
IQR = Q3 - Q1

print("IQR:", IQR)


sns.histplot(df['Household_Income'], kde=True)
plt.title("Income Distribution")
plt.show()


mu, std = df['Household_Income'].mean(), df['Household_Income'].std()

xmin, xmax = df['Household_Income'].min(), df['Household_Income'].max()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

plt.hist(df['Household_Income'], bins=30, density=True)
plt.plot(x, p)
plt.title("Normal Distribution")
plt.show()

print("Skewness:", df['Household_Income'].skew())
print("Kurtosis:", df['Household_Income'].kurt())


sns.histplot(df['Household_Income'], kde=True)
plt.title("Income Distribution with KDE")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()


sns.boxplot(x='Education_Level', y='Family_Size', data=df)
plt.title("Family Size by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Family Size")
plt.show()

sns.kdeplot(x=df['Age_of_Household_Head'], y=df['Household_Income'], cmap="Blues", fill=True)
plt.title("Age vs Income Distribution")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()

