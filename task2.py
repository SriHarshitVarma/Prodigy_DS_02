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

# Create new features
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1

# Explore categorical features
print(titanic_data['Pclass'].value_counts())
print(titanic_data['Sex'].value_counts())
print(titanic_data['Embarked'].value_counts())

# Visualizations
# Survival rate
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

# Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival by Pclass')
plt.show()

# Survival by Sex
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival by Sex')
plt.show()

# Survival by Embarked
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival by Embarked')
plt.show()

# Distribution of Age
sns.histplot(titanic_data['Age'])
plt.title('Age Distribution')
plt.show()

# Fare distribution
sns.histplot(titanic_data['Fare'])
plt.title('Fare Distribution')
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
