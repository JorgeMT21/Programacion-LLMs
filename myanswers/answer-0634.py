import numpy as np
from sklearn.metrics import precision_score

def generar_caso_de_uso_evaluar_precision_positiva():
    size = np.random.randint(50, 100)
    y_real = np.random.randint(0, 2, size)
    y_pred = np.random.randint(0, 2, size)

    input_data = {
        "y_real": y_real,
        "y_pred": y_pred
    }

    output_data = float(
        precision_score(
            y_real,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    return input_data, output_data
