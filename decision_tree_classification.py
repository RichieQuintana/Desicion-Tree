# Árbol de Decisión para Clasificación - Bank Marketing Dataset
# Código completo y corregido para evitar errores de dimensiones

# Importación de bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargar el dataset (asegúrate de tener el archivo 'bank.csv' en tu directorio)
dataset = pd.read_csv('bank.csv')

# Preprocesamiento: convertir variables categóricas a numéricas
label_encoder = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_cols:
    if col in dataset.columns:
        dataset[col] = label_encoder.fit_transform(dataset[col])

# Codificar la variable objetivo (deposit: 'yes'/'no' -> 1/0)
dataset['deposit'] = label_encoder.fit_transform(dataset['deposit'])

# Seleccionar características (X) y variable objetivo (y)
# Usamos 3 características numéricas para el modelo, pero 2 para visualización
X = dataset[['age', 'balance', 'duration']].values  # Características para el modelo
y = dataset['deposit'].values  # Variable objetivo

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado de características (para el modelo completo con 3 características)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Entrenar el modelo de Árbol de Decisión (con 3 características)
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train_scaled, y_train)

# Predicción de ejemplo (edad=30, saldo=2000, duración=300)
ejemplo = sc.transform([[30, 2000, 300]])
prediccion = classifier.predict(ejemplo)
print(f"Predicción para [edad=30, saldo=2000, duración=300]: {'Sí' if prediccion[0] == 1 else 'No'}")

# Evaluación del modelo
y_pred = classifier.predict(X_test_scaled)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2f}")

# Seleccionar solo 'age' y 'balance' para visualización
X_plot_train = X_train[:, :2]  # age (col 0) y balance (col 1)
X_plot_test = X_test[:, :2]

# Escalador específico para las 2 características de visualización
sc_plot = StandardScaler()
X_plot_train_scaled = sc_plot.fit_transform(X_plot_train)

# Entrenar un modelo AUXILIAR solo para visualización 
classifier_plot = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_plot.fit(X_plot_train_scaled, y_train)

# Generar gráfico de fronteras de decisión (conjunto de entrenamiento)
X_set, y_set = sc_plot.inverse_transform(X_plot_train_scaled), y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=100)
)
plt.contourf(X1, X2, classifier_plot.predict(sc_plot.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.5, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Scatter plot de los puntos reales
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), 
                label='Sí' if j == 1 else 'No',
                edgecolors='black', s=20)

plt.title('Árbol de Decisión - Fronteras de Decisión (Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Saldo')
plt.legend()
plt.show()