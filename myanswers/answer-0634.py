def evaluar_precision_positiva(y_real, y_pred):
    """
    Solución del caso de uso:
    calcula la precisión de la clase positiva, es decir,
    de todos los valores predichos como 1, qué proporción realmente eran 1.
    """

    # Calcular precision para la clase positiva
    precision = precision_score(
        y_real,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # Devolver como float
    return float(precision)


# ============================================================
# COMPROBACIÓN DE LA FUNCIÓN SOLUCIÓN
# ============================================================

input_data, output_data = generar_caso_de_uso_evaluar_precision_positiva()

resultado = evaluar_precision_positiva(**input_data)

# Como el generador devuelve None, calculamos aquí el valor esperado
salida_esperada = float(
    precision_score(
        input_data["y_real"],
        input_data["y_pred"],
        pos_label=1,
        zero_division=0
    )
)

print("=== INPUT DEL CASO DE USO ===")
print("y_real:")
print(input_data["y_real"])

print("\ny_pred:")
print(input_data["y_pred"])

print("\n=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)

print("\n=== OUTPUT ESPERADO ===")
print(salida_esperada)

print("\n=== COMPROBACIÓN ===")
print(np.isclose(resultado, salida_esperada))
