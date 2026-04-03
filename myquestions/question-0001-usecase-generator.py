import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

def generar_caso_de_uso_calcular_scores_chi2():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_scores_chi2(df, columnas, columna_objetivo).
    """
    
    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    n_rows = random.randint(30, 80)
    n_features = random.randint(3, 6)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    
    # ---------------------------------------------------------
    # 2. Generar datos aleatorios
    # ---------------------------------------------------------
    X = np.random.randn(n_rows, n_features)
    
    # Generar objetivo categórico binario a partir de algunas features
    pesos = np.random.uniform(-2, 2, size=n_features)
    logits = X @ pesos
    y = (logits > np.median(logits)).astype(int)
    
    df = pd.DataFrame(X, columns=columnas)
    columna_objetivo = 'target'
    df[columna_objetivo] = y
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columnas': columnas,
        'columna_objetivo': columna_objetivo
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    
    X_expected = df[columnas].to_numpy()
    y_expected = df[columna_objetivo].to_numpy()
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_expected)
    
    scores, _ = chi2(X_scaled, y_expected)
    
    output_data = scores
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_calcular_scores_chi2()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas predictoras: {entrada['columnas']}")
    print(f"Columna objetivo: {entrada['columna_objetivo']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Array de numpy) ===")
    print(f"Shape del array de scores: {salida_esperada.shape}")
    print("Scores chi-cuadrado:")
    print(salida_esperada)
