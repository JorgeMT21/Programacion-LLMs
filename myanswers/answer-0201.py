from sklearn.cluster import KMeans

def agrupar_danos(df, n_clusters):
    X = df[["x", "y"]]

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    return labels
