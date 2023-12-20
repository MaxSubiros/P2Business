from cmath import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objs as go
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/dsindy/kaggle-titanic/master/data/train.csv'
dataset = pd.read_csv(url)

## DESCRIPCIÓ DEL DATASET (GENÈRIC)

# Resum del dataset (les 5 primeres files)
#print(dataset.head())

# Resum estadístic del dataset (count,mean,min,max,etc.) per cada columna
#print(dataset.describe())

# Veure quines columnes tenim
#print(dataset.columns)

# Veure el tipus de dada de cada columna 
#print(dataset.info())

# Veure quants valors nuls tenim per columna
#print(dataset.isnull().sum())

## MIREM LA CORRELACIÓ ENTRE LES VARIABLES

# Correlació entre les variables
numeric_dataset = dataset.select_dtypes(include=[np.number])
print(numeric_dataset.corr(method='pearson'))

#Crear gràfica per veure la correlació entre les variables
plt.figure(figsize=(15,12.5))
sns.heatmap(dataset.select_dtypes(include=[np.number]).corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlació entre les variables')
plt.show()

# Correlació entre variables i Survived
print(numeric_dataset.corrwith(dataset['Survived']))

# Eliminem les columnes que no ens interessen
dataset = dataset.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Convertim les columnes amb valors categòrics a numèrics
dataset = pd.get_dummies(dataset)

# Calculem la correlació de cada variable amb 'Survived' i fem la gràfica
correlation = dataset.corrwith(dataset['Survived']).sort_values(ascending=False)

plt.figure(figsize=(15, 7))
correlation.plot(kind='bar', color='blue')
plt.title('Correlació amb Survived')
plt.xlabel('Variables')
plt.ylabel('Correlació')
plt.show()

# Llista de columnes a eliminar (surten NaN o <0.05 de correlació)
cols_to_drop = ['SibSp', 'PassengerId']

# Eliminem les columnes de la llista anterior
dataset = dataset.drop(cols_to_drop, axis=1)

# Eliminem les files que tinguin algun NaN
dataset = dataset.dropna()


