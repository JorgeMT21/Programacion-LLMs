from sklearn.metrics import precision_score

def evaluar_precision_positiva(y_real, y_pred):
    precision = precision_score(
        y_real,
        y_pred,
        pos_label=1,
        zero_division=0
    )
    return float(precision)
