import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

def generar_caso_de_uso_evaluar_modelo_poisson():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_modelo_poisson(df, columnas, columna_objetivo, test_size).
    """
    
    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    n_rows = random.randint(50, 120)
    n_features = random.randint(2, 5)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    columna_objetivo = 'target'
    test_size = random.choice([0.2, 0.25, 0.3])
    
    # ---------------------------------------------------------
    # 2. Generar datos aleatorios
    # ---------------------------------------------------------
    X = np.random.randn(n_rows, n_features)
    
    # Generar medias positivas para una variable objetivo de conteo
    coef = np.random.uniform(-0.5, 0.8, size=n_features)
    eta = X @ coef
    mu = np.exp(eta)  # media positiva
    
    y = np.random.poisson(lam=mu)
    
    df = pd.DataFrame(X, columns=columnas)
    df[columna_objetivo] = y
    
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = PoissonRegressor()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    output_data = mean_absolute_error(y_test, y_pred)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_modelo_poisson()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas predictoras: {entrada['columnas']}")
    print(f"Columna objetivo: {entrada['columna_objetivo']}")
    print(f"Test size: {entrada['test_size']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Mean Absolute Error esperado: {salida_esperada}")
