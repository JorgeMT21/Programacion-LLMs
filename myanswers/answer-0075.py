def comprimir_sensores_correlacionados(X, n_clusters):
    """
    Solución del caso de uso:
    reduce la cantidad de sensores o características usando FeatureAgglomeration.
    """

    # 1. Crear el modelo FeatureAgglomeration con el número de clusters indicado
    model = FeatureAgglomeration(n_clusters=n_clusters)

    # 2. Ajustar el modelo y transformar la matriz X
    X_comprimido = model.fit_transform(X)

    # 3. Devolver la matriz transformada
    return X_comprimido


# ============================================================
# COMPROBACIÓN DE LA FUNCIÓN SOLUCIÓN
# ============================================================

resultado = comprimir_sensores_correlacionados(**input_data)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)
print("\nForma del resultado:")
print(resultado.shape)

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(output_data)
print("\nForma esperada:")
print(output_data.shape)

print("\n=== COMPROBACIÓN ===")
print(np.allclose(resultado, output_data))
