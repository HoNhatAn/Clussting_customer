import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
def normalize(X):
    n,d=X.shape
    temp=np.zeros((n,d))
    for i in range(d):
        temp[:,i]=(X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    return temp
def Getdata():
    X=pd.read_csv("D:/Data.csv", header=None).values
    X = X[1:X.shape[0], :].astype(float)
    return X
def export_data(X):
    path = input("Nhap ten file: ")
    headers=['Height(Inches)','Weight(Pounds)','Label']
    path="D:/"+path+".csv"
    with open(path,'w',newline="") as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        writer = csv.writer(f)
        writer.writerows(X)
def display(X):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Original data points")
    plt.plot(X[:, 0], X[:, 1], 'bo', markersize=2)
    plt.show()
def distance(X, centers):
    d = np.zeros(shape=(X.shape[0], centers.shape[0]))
    for j in range(centers.shape[0]):
        for i in range(centers.shape[1]):
            #Tính tổng bình phương
            d[:, j] = d[:, j] + (X[:, i] - centers[j, i]) ** 2
    return np.sqrt(d)
def nearest_centers_point(X, centers):
    D = distance(X, centers)
    label = np.argmin(D, axis=1)
    return label
def check_matrix_equal(M, N):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (M[i, j] != N[i, j]):
                return False
    return True
def kmeans_init_centers(X, n_cluster):
    return X[np.random.choice(X.shape[0], n_cluster, replace=False)]
def kmeans_visualize(X, centers, labels, n_cluster, title):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(n_cluster):
        data = X[labels == i]
        plt.plot(data[:, 0], data[:, 1], plt_colors[i] + '^', markersize=4, label='cluster_' + str(i))
        plt.plot(centers[i][0], centers[i][1], plt_colors[i + 4] + 'o', markersize=10, label='center_' + str(i))
        plt.legend()
    plt.show()
def kmeans_update_centers(X, labels, n_cluster):
    centers = np.zeros((n_cluster, X.shape[1]))
    for k in range(n_cluster):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0)
    return centers
def kmeans(init_centes, init_labels, X, n_cluster):
    centers = init_centes
    labels = init_labels
    times = 0
    while True:
        # ham tinh khoang cach gan nhat
        labels = nearest_centers_point(X,centers)
        if times==0:
            kmeans_visualize(X, centers, labels, n_cluster, 'Assigned label for data at time = ' + str(times + 1))
        new_centers = kmeans_update_centers(X, labels, n_cluster)
        if check_matrix_equal(centers, new_centers):
            break
        centers = new_centers
        #kmeans_visualize(X, centers, labels, n_cluster, 'Update center possition at time = ' + str(times + 1))
        times += 1
    kmeans_visualize(X, centers, labels, n_cluster, "Finish")
    return centers, labels, times
def frequentcy(List,X):
    value=dict()
    for i in List:
        value[i] = List.count(i)
    percent=dict()
    for i,j in value.items():
        percent[i]=100*j/X.shape[0]
    return value,percent
def add_labels(labels,X):
    ones = np.zeros((X.shape[0],1))
    X0 = np.concatenate((ones, X), axis=1)
    X0[:, 0] = X0[:, 0] + labels[:]
    X0 = X0.T
    for i in range(0,X0.shape[0]-1):
        X0[[i, i+1]] = X0[[i+1, i]]
    X0 = X0.T
    return X0
X=normalize(Getdata())
display(X)
n_cluster = 3
init_centers = kmeans_init_centers(X, n_cluster)
init_labels = np.zeros(X.shape[0])
kmeans_visualize(X, init_centers, init_labels, n_cluster,'Init centers in the first run. Assigned all data as cluster 0')
centers, labels, times = kmeans(init_centers, init_labels, X, n_cluster)
print('finished', times, 'times')
X1=add_labels(labels,Getdata())
print(X1)
value,percent=frequentcy(labels.tolist(),X)
print(X1[9,:])
print(value)
print(percent)
export_data(X1)
