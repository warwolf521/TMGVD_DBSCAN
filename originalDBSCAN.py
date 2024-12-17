import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
import time
from memory_profiler import memory_usage
import csv
import pandas as pd

def dbscan(data, minPts, eps):
    """
    Implementacion de DBSCAN.

    Parametros:
    - data: Array de numpy de forma (n, 3), donde n es el número de puntos.
    - minPts: Número mínimo de vecinos requeridos para formar una región densa.
    - eps: Radio para la búsqueda de vecindario.

    Returns:
    - labels: Array de numpy de forma (n,) con etiquetas de cluster (-1 para ruido).
    """
    n = data.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    clusterId = 0

    def regionQuery(pointIdx):
        """Find neighbors within eps of a given point."""
        distances = np.linalg.norm(data - data[pointIdx], axis=1)
        return np.where(distances <= eps)[0]
        
    for pointIdx in range(n):
        if not visited[pointIdx]:
            visited[pointIdx] = True
            neighbors = regionQuery(pointIdx)
            if len(neighbors) < minPts:
                labels[pointIdx] = -1
            else:
                clusterId += 1
                labels[pointIdx] = clusterId
                i = 0
                while i < len(neighbors):
                    neighborIdx = neighbors[i]
                    if not visited[neighborIdx]:
                        visited[neighborIdx] = True
                        newNeighbors = regionQuery(neighborIdx)
                        if len(newNeighbors) >= minPts:
                            neighbors = np.append(neighbors, newNeighbors)
                    if labels[neighborIdx] == -1:
                        labels[neighborIdx] = clusterId
                    i += 1
    return labels

def findMaxCurvature(data, k=1):
    """
    Encontrar el valor óptimo de Eps analizando la curvatura de la curva K-dist.

    Parametros:
    - data: matriz numpy de forma (n, 3), donde n es el número de puntos.
    - k: Número de vecinos para calcular la curvatura (por defecto=1).

    Returns:
    - optimalEps: Valor óptimo de Eps.
    """

    distMatrix = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    sortedDists = np.sort(distMatrix, axis=1)[:, k]
    sortedDists = np.sort(sortedDists)

    for degree in range(1, 70):
        poly = Polynomial.fit(range(len(sortedDists)), sortedDists, deg=degree)
        yFit = poly(range(len(sortedDists)))
        residuals = sortedDists - yFit
        ssResid = np.sum(residuals ** 2)
        ssTotal = (len(sortedDists) - 1) * np.var(sortedDists)
        r2 = 1 - (ssResid / ssTotal)
        if r2 >= 0.99:
            break

    derivative = poly.deriv()
    curvature = np.abs(derivative(range(len(sortedDists)))) 
    maxCurvatureIdx = np.argmax(curvature)
    optimalEps = sortedDists[maxCurvatureIdx]

    return optimalEps

def calculateMinPts(data, eps, beta=0.5):
    """
    Calcular el valor de MinPts basado en el número promedio de vecinos dentro de Eps.

    Parametros:
    - data: matriz numpy de forma (n, 3), donde n es el número de puntos.
    - eps: Radio para la búsqueda de vecindario.
    - beta: Factor de ajuste para la supresión de ruido (por defecto=0.5).

    Returns:
    - minPts: Valor de MinPts calculado.
    """

    n = data.shape[0]
    neighborsCount = []

    for i in range(n):
        distances = np.linalg.norm(data - data[i], axis=1)
        neighborsCount.append(np.sum(distances <= eps))

    averageNeighbors = np.mean(neighborsCount)
    minPts = int(np.ceil(beta * averageNeighbors))

    return minPts

def adaptiveDBSCAN(data, k=None, beta=0.5):
    """
    Adaptive DBSCAN implementacion que optimiza Eps y MinPts usando el índice CH.

    Parameters:
    - data: matriz numpy de forma (n, 3), donde n es el número de puntos.
    - k: Número máximo de iteraciones para encontrar el valor óptimo de Eps (por defecto=n-1).
    - beta: Factor de ajuste para la supresión de ruido (por defecto=0.5).

    Returns:
    - labels: Array de numpy de forma (n,) con etiquetas de cluster (-1 para ruido).
    - bestEps: El valor optimizado de Eps.
    - bestMinPts: El valor optimizado de MinPts.
    """
    bestEps = 0
    bestMinPts = 0
    bestCh = -np.inf
    bestLabels = None

    if k is None:
        k = data.shape[0] - 1  # Default to n-1 if not provided

    for i in range(1, k + 1):
        print("K:" + str(i))
        eps = findMaxCurvature(data, k=i)
        print("Optimal Eps:" + str(eps))
        minPts = calculateMinPts(data, eps, beta)
        print("Optimal MinPts:" + str(minPts))
        labels = dbscan(data, minPts, eps)
        numberClusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("Nº Cluster labels:" + str(numberClusters))

        if numberClusters > 1:  # Ensure there are at least two clusters
            try:
                chScore = calinski_harabasz_score(data, labels)
                print("Calinski-Harabasz Index:" + str(chScore))
                print("-------------------------------------------------")
                if chScore > bestCh:
                    bestCh = chScore
                    bestEps = eps
                    bestMinPts = minPts
                    bestLabels = labels
            except Exception as e:
                continue

    return bestLabels, bestEps, bestMinPts

def readTxtFile(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                x, y, z, label = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                points.append([x, y, z, label])
            elif len(parts) == 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2]), 0])  # Default label 0 if not provided
    return np.array(points)

def plotClusters(data, labels):
    uniqueLabels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(uniqueLabels))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Configurar gráfico 3D
    
    for k, col in zip(uniqueLabels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        classMemberMask = (labels == k)
        
        xyz = data[classMemberMask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=tuple(col), edgecolor='k', s=10)

    ax.set_title("DBSCAN Clustering (3D)")
    plt.show()

def saveToCSV(data, file, totalExecutionTime, iterationExecutionTime, avgTimePerPoint, peakMemory):
        # Guardar métricas en un archivo CSV
    metrics = {
        "Dataset": file,
        "Number of Points": data.shape[0],
        "Total Execution Time (s)": totalExecutionTime,
        "Iteration Execution Time (s)": iterationExecutionTime,
        "Average Time per Point (s)": avgTimePerPoint,
        "Peak Memory Usage (MiB)": peakMemory
    }

    try:
        # Si el archivo ya existe, cargarlo y agregar la nueva fila
        df = pd.read_csv(csvFile)
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    except FileNotFoundError:
        # Si el archivo no existe, crearlo
        df = pd.DataFrame([metrics])

    df.to_csv(csvFile, index=False)

    print(f"Métricas guardadas en {csvFile}")

if __name__ == "__main__":
    file = "datasets/armadillo_points.txt"
    csvFile = "original_armadillo.csv"
    data = readTxtFile(file)
    totalPoints = len(data)
    numPoints = 2000
    indices = np.linspace(0, totalPoints - 1, numPoints, dtype=int)
    data = data[indices]

    maxK = 10
    
    startTime = time.process_time()
    memUsage = memory_usage((adaptiveDBSCAN, (data[:, :3], maxK, 1)))
    endTime = time.process_time()

    labels, eps, minPts = adaptiveDBSCAN(data[:, :3], maxK, beta=1)

    peakMemory = max(memUsage)
    totalExecutionTime = endTime - startTime
    iterationExecutionTime = totalExecutionTime / maxK
    avgTimePerPoint = totalExecutionTime / len(data)
    

    # Imprimir métricas
    print()
    print("DBSCAN clustering results:")
    print(f"Number of points: {data.shape[0]}")
    print(f"Tiempo total de ejecución: {totalExecutionTime:.6f} segundos")
    print(f"Tiempo promedio por iteracion: {iterationExecutionTime:.6f} segundos")
    print(f"Tiempo promedio por punto: {avgTimePerPoint:.6f} segundos")
    print(f"Peak Memory Usage: {peakMemory} MiB")

    saveToCSV(data, file, totalExecutionTime, iterationExecutionTime, avgTimePerPoint, peakMemory)

    plotClusters(data[:, :3], labels)