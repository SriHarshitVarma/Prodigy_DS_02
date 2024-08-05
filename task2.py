import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = pd.read_csv("train.csv")

# Exploratory Data Analysis (EDA)
print(titanic_data.head())
print(titanic_data.info())
print(titanic_data.describe())

# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna('S', inplace=True)

# Feature Engineering
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1
titanic_data['Title'] = titanic_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Explore categorical features
print(titanic_data['Pclass'].value_counts())
print(titanic_data['Sex'].value_counts())
print(titanic_data['Embarked'].value_counts())
print(titanic_data['Title'].value_counts())

# Visualizations
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival by Pclass')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival by Sex')
plt.show()

sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival by Embarked')
plt.show()

sns.histplot(titanic_data['Age'])
plt.title('Age Distribution')
plt.show()

sns.histplot(titanic_data['Fare'])
plt.title('Fare Distribution')
plt.show()

sns.boxplot(x='Pclass', y='Age', hue='Survived', data=titanic_data)
plt.title('Age Distribution by Pclass and Survival')
plt.show()

# Correlation matrix
correlation_matrix = titanic_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Survival probability by Pclass, Sex, and Age
sns.catplot(x='Pclass', hue='Survived', col='Sex', data=titanic_data, kind='count')
plt.show()

# Pairplot for numerical features
sns.pairplot(titanic_data[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
plt.show()
