def agrupar_danos(df, n_clusters):
    """
    Solución del caso de uso:
    agrupa puntos de daño estructural usando KMeans
    y devuelve las etiquetas de cluster.
    """

    # 1. Seleccionar las columnas x e y
    X = df[["x", "y"]]

    # 2. Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)

    # 3. Obtener las etiquetas de cluster
    labels = kmeans.fit_predict(X)

    # 4. Devolver las etiquetas
    return labels


# ============================================================
# COMPROBACIÓN DE LA FUNCIÓN SOLUCIÓN
# ============================================================

input_data, output_data = generar_caso_de_uso_agrupar_danos()

resultado = agrupar_danos(**input_data)

print("=== INPUT DEL CASO DE USO ===")
print(f"Número de clusters: {input_data['n_clusters']}")
print(input_data["df"].head())

print("\n=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(output_data)

print("\n=== COMPROBACIÓN EXACTA ===")
print(np.array_equal(resultado, output_data))
