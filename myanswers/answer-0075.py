from sklearn.cluster import FeatureAgglomeration

def comprimir_sensores_correlacionados(X, n_clusters):
    model = FeatureAgglomeration(n_clusters=n_clusters)
    X_comprimido = model.fit_transform(X)

    return X_comprimido
