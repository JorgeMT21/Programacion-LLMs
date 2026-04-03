import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import median_absolute_error

def generar_caso_de_uso_evaluar_regresion_robusta():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_regresion_robusta(df, columnas, columna_objetivo, test_size).
    """
    
    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    n_rows = random.randint(50, 100)
    n_features = random.randint(2, 5)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    
    # ---------------------------------------------------------
    # 2. Generar datos con relación lineal y algunos outliers
    # ---------------------------------------------------------
    X = np.random.randn(n_rows, n_features)
    coef = np.random.uniform(-4, 4, size=n_features)
    y = X @ coef + np.random.normal(0, 1, size=n_rows)
    
    # Introducir algunos outliers en la variable objetivo
    n_outliers = random.randint(2, 5)
    indices_outliers = np.random.choice(n_rows, size=n_outliers, replace=False)
    y[indices_outliers] += np.random.uniform(15, 30, size=n_outliers)
    
    df = pd.DataFrame(X, columns=columnas)
    columna_objetivo = 'target'
    df[columna_objetivo] = y
    
    test_size = random.choice([0.2, 0.25, 0.3])
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columnas': columnas,
        'columna_objetivo': columna_objetivo,
        'test_size': test_size
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    
    X_data = df[columnas].to_numpy()
    y_data = df[columna_objetivo].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42
    )
    
    model = TheilSenRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    output_data = median_absolute_error(y_test, y_pred)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_regresion_robusta()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas predictoras: {entrada['columnas']}")
    print(f"Columna objetivo: {entrada['columna_objetivo']}")
    print(f"Test size: {entrada['test_size']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Median Absolute Error esperado: {salida_esperada}")
