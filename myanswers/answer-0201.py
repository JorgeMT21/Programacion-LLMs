def analizar_eficiencia_rutas(df):
    """
    Solución del caso de uso:
    analiza el desempeño de las rutas de entrega, eliminando envíos fallidos,
    calculando retrasos y conservando solo rutas con al menos 3 entregas.
    """

    # 1. Filtrar las entregas que no fallaron
    df_valid = df[df["status"] != "failed"].copy()

    # 2. Calcular la columna delay
    df_valid["delay"] = df_valid["actual_duration"] - df_valid["planned_duration"]

    # 3. Agrupar por ruta y calcular promedio de delay y número de entregas
    resultado = (
        df_valid.groupby("route_id")
        .agg(
            delay_promedio=("delay", "mean"),
            entregas=("route_id", "count")
        )
        .reset_index()
    )

    # 4. Conservar únicamente rutas con al menos 3 entregas completadas
    resultado = resultado[resultado["entregas"] >= 3]

    # 5. Ordenar por delay promedio de mayor a menor y reiniciar índice
    resultado = (
        resultado
        .sort_values("delay_promedio", ascending=False)
        .reset_index(drop=True)
    )

    return resultado


# ============================================================
# COMPROBACIÓN DE LA FUNCIÓN SOLUCIÓN
# ============================================================

caso, descripcion = generar_caso_de_uso_analizar_eficiencia_rutas()

resultado = analizar_eficiencia_rutas(**caso["input"])

print("=== DESCRIPCIÓN DEL CASO ===")
print(descripcion)

print("\n=== INPUT DEL CASO DE USO ===")
print(caso["input"]["df"])

print("\n=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(caso["output"])

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(
        resultado,
        caso["output"],
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10
    )
    print(True)

except AssertionError as error:
    print(False)
    print(error)
