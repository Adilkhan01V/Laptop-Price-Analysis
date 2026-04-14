# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("laptop_prices.csv")

# =========================
# BASIC INFO (EDA)
# =========================
print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# SELECT IMPORTANT COLUMNS
# =========================
df = df[['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage', 'Price_euros']].dropna()

# =========================
# HISTOGRAM + KDE
# =========================
features = ['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage', 'Price_euros']

for col in features:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution (Skewness: {round(df[col].skew(), 2)})")
    plt.show()

# =========================
# BOXPLOT
# =========================
for col in features:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# =========================
# HEATMAP
# =========================
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# OBJECTIVE 1: RAM vs PRICE
# =========================
sns.regplot(x='Ram', y='Price_euros', data=df)
plt.title("RAM vs Price")
plt.show()

print("Correlation:", df['Ram'].corr(df['Price_euros']))

# Linear Regression
X = df[['Ram']]
y = df['Price_euros']

model = LinearRegression()
model.fit(X, y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# =========================
# OBJECTIVE 2: CPU vs PRICE
# =========================
sns.regplot(x='CPU_freq', y='Price_euros', data=df)
plt.title("CPU vs Price")
plt.show()

print("Correlation:", df['CPU_freq'].corr(df['Price_euros']))

# =========================
# OBJECTIVE 3: STORAGE vs PRICE
# =========================
sns.regplot(x='PrimaryStorage', y='Price_euros', data=df)
plt.title("Storage vs Price")
plt.show()

print("Correlation:", df['PrimaryStorage'].corr(df['Price_euros']))

# =========================
# OBJECTIVE 4: PAIRPLOT
# =========================
sns.pairplot(df[['Ram', 'CPU_freq', 'PrimaryStorage', 'Weight', 'Price_euros']])
plt.show()

# =========================
# OBJECTIVE 5: HYPOTHESIS TESTING (T-TEST)
# =========================

# Create groups
low_ram = df[df['Ram'] <= 8]['Price_euros']
high_ram = df[df['Ram'] > 8]['Price_euros']

# Perform t-test
t_stat, p_value = ttest_ind(low_ram, high_ram)

print("\n===== T-TEST RESULT =====")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Decision
alpha = 0.05

if p_value < alpha:
    print("Reject Null Hypothesis (Significant Difference)")
else:
    print("Fail to Reject Null Hypothesis (No Significant Difference)")