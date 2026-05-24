import pandas as pd

def analizar_eficiencia_rutas(df=None, input=None, output=None):
    if df is None and input is not None:
        df = input["df"]

    df_valid = df[df["status"] != "failed"].copy()

    df_valid["delay"] = df_valid["actual_duration"] - df_valid["planned_duration"]

    resultado = (
        df_valid.groupby("route_id")
        .agg(
            delay_promedio=("delay", "mean"),
            entregas=("route_id", "count")
        )
        .reset_index()
    )

    resultado = (
        resultado[resultado["entregas"] >= 3]
        .sort_values("delay_promedio", ascending=False)
        .reset_index(drop=True)
    )

    return resultado
