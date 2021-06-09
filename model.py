import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import networkx as nx

###################################################

# reading dataframe
data = pd.read_csv("log2.csv")
# removing instances where Bytes are beyond 99th percentile
data = data[data['Bytes'] <= np.percentile(data['Bytes'], 99)]
# removing instances where Packets are beyond 99th percentile
data = data[data['Packets'] <= np.percentile(data['Packets'], 99)]
# removing instances where Elapsed Time (sec) are beyond 99th percentile
data = data[data['Elapsed Time (sec)'] <= np.percentile(data['Elapsed Time (sec)'], 99)]
# adding translation features
data['Source Port Translation'] = (data['Source Port'] != data['NAT Source Port']).astype('int')
data['Destination Port Translation'] = (data['Destination Port'] != data['NAT Destination Port']).astype('int')

# building bidirectional graph by using port numbers on host devices
HOST_NW = nx.DiGraph(name = "Host")
HOST_NW.add_edges_from(data[['Source Port', 'Destination Port']].values)
joblib.dump(HOST_NW, 'host_nw.pkl')
# building bidirectional graph by using port numbers on NAT 
NAT_NW = nx.DiGraph(name = "NAT")
NAT_NW.add_edges_from(data[['NAT Source Port', 'NAT Destination Port']].values)
joblib.dump(NAT_NW, 'nat_nw.pkl')

def common_ports(nw, src, dst):
    """
    Counts no. of common ports connected directly between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        No. of common ports from intersection of set of neighbors of both src and dst ports
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst))))
    except:
        return 0

# adding features to the dataset
data['Host CP'] = data.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
data['NAT CP'] = data.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)

def jaccard_index(nw, src, dst):
    """
    Counts Jaccard index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Jaccard Index
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))) / len(set(nw.neighbors(src)).union(set(nw.neighbors(dst))))
    except:
        return 0

# adding features to the dataset
data['Host JI'] = data.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
data['NAT JI'] = data.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)

def salton_index(nw, src, dst):
    """
    Counts Salton index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Salton Index
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))) / np.sqrt(len(set(nw.neighbors(src))) * len(set(nw.neighbors(dst))))
    except:
        return 0
        
# adding features to the dataset
data['Host SL'] = data.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
data['NAT SL'] = data.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)

def sorensen_index(nw, src, dst):
    """
    Counts Sorensen index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Sorensen Index
    """
    try:
        return len(set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))) / (len(set(nw.neighbors(src))) + len(set(nw.neighbors(dst))))
    except:
        return 0
        
# adding features to the dataset
data['Host SI'] = data.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
data['NAT SI'] = data.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)

def adamic_adar_index(nw, src, dst):
    """
    Counts Adamic-Adar index between src and dst ports.

    Args:
        nw: Network instance
        src: Source Port
        dst: Destination Port

    Returns:
        Adamic-Adar Index
    """
    try:
        ports = set(nw.neighbors(src)).intersection(set(nw.neighbors(dst)))
        return 1/np.sum([np.log10(set(nw.neighbors(port))) for port in ports])
    except:
        return 0
        
# adding features to the dataset
data['Host AA'] = data.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
data['NAT AA'] = data.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)

# networkx library provides a function to calculate page rank of ports in the networks. It returns a dictionary where keys are ports and values are pageranks.
host_page_rank = nx.pagerank(HOST_NW)
nat_page_rank = nx.pagerank(NAT_NW)

# adding features to the dataset
data['Host Source PR'] = data.apply(lambda row: host_page_rank.get(row['Source Port'], 0), axis = 1)
data['Host Destination PR'] = data.apply(lambda row: host_page_rank.get(row['Destination Port'], 0), axis = 1)
data['NAT Source PR'] = data.apply(lambda row: nat_page_rank.get(row['NAT Source Port'], 0), axis = 1)
data['NAT Destination PR'] = data.apply(lambda row: nat_page_rank.get(row['NAT Destination Port'], 0), axis = 1)

# seperating Action class as target feature
y = data['Action']
X = data.drop(['Action'], axis = 1)

# splitting data for calibration
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 859)

# applying RobustScaler
robust_scaler = RobustScaler()
robust_scaler_features = ['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received', 'Host Source PR', 'Host Destination PR', 'NAT Source PR', 'NAT Destination PR']
X_train_robust_scaled = robust_scaler.fit_transform(X_train[robust_scaler_features])
X_cv_robust_scaled = robust_scaler.transform(X_cv[robust_scaler_features])
joblib.dump(robust_scaler, 'robust_scaler.pkl')

# applying StandardScaler
std_scaler = StandardScaler()
std_scaler_features = ['Host CP', 'NAT CP', 'Host JI', 'NAT JI', 'Host SL', 'NAT SL', 'Host SI', 'NAT SI', 'Host AA', 'NAT AA']
X_train_std_scaled = std_scaler.fit_transform(X_train[std_scaler_features])
X_cv_std_scaled = std_scaler.transform(X_cv[std_scaler_features])
joblib.dump(std_scaler, 'std_scaler.pkl')

# stacking the scaled features
X_train_preprocessed = np.hstack((X_train[X_train.columns[:4]], X_train_robust_scaled, X_train_std_scaled, X_train[['Source Port Translation', 'Destination Port Translation']]))
X_cv_preprocessed = np.hstack((X_cv[X_cv.columns[:4]], X_cv_robust_scaled, X_cv_std_scaled, X_cv[['Source Port Translation', 'Destination Port Translation']]))

# training the classifier on data
lgbm = LGBMClassifier(n_estimators = 750, max_depth = 4, objective = 'multiclass', class_weight = 'balanced', n_jobs = -1, random_state = 859)
lgbm.fit(X_train_preprocessed, y_train)

# calibrating the classifier on validation data
calibrator_lgbm = CalibratedClassifierCV(lgbm, method = 'isotonic', cv = 'prefit')
calibrator_lgbm.fit(X_cv_preprocessed, y_cv)
joblib.dump(calibrator_lgbm, 'calibrator_lgbm.pkl')