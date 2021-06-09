from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import joblib
import networkx as nx
from lightgbm import LGBMClassifier

# loading pickle objects
robust_scaler = joblib.load('robust_scaler.pkl')
robust_scaler_features = ['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received', 'Host Source PR', 'Host Destination PR', 'NAT Source PR', 'NAT Destination PR']

std_scaler = joblib.load('std_scaler.pkl')
std_scaler_features = ['Host CP', 'NAT CP', 'Host JI', 'NAT JI', 'Host SL', 'NAT SL', 'Host SI', 'NAT SI', 'Host AA', 'NAT AA']

calibrator_lgbm = joblib.load('calibrator_lgbm.pkl')
HOST_NW = joblib.load('host_nw.pkl')
NAT_NW = joblib.load('nat_nw.pkl')

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################


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
    
# networkx library provides a function to calculate page rank of ports in the networks. It returns a dictionary where keys are ports and values are pageranks.
host_page_rank = nx.pagerank(HOST_NW)
nat_page_rank = nx.pagerank(NAT_NW)


def preprocess(feature_vector):
    """
    Preprocesses the feature matrix of firewall logs.

    Args:
        feature_vector: Input feature matrix of firewall logs

    Returns:
        Preprocessed feature vector
    """
    try:
        # reshaping into row vector if feature matrix of single instance is given
        # creating empty dataframe
        feature_names = ['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'Bytes', 'Bytes Sent',
                         'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']
        data_matrix = pd.DataFrame(feature_vector, columns = feature_names)
        # applying engineered features
        data_matrix['Source Port Translation'] = (data_matrix['Source Port'] != data_matrix['NAT Source Port']).astype('int')
        data_matrix['Destination Port Translation'] = (data_matrix['Destination Port'] != data_matrix['NAT Destination Port']).astype('int')
        data_matrix['Host CP'] = data_matrix.apply(lambda row: common_ports(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT CP'] = data_matrix.apply(lambda row: common_ports(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host JI'] = data_matrix.apply(lambda row: jaccard_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT JI'] = data_matrix.apply(lambda row: jaccard_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host SL'] = data_matrix.apply(lambda row: salton_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT SL'] = data_matrix.apply(lambda row: salton_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host SI'] = data_matrix.apply(lambda row: sorensen_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT SI'] = data_matrix.apply(lambda row: sorensen_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host AA'] = data_matrix.apply(lambda row: adamic_adar_index(HOST_NW, row['Source Port'], row['Destination Port']), axis = 1)
        data_matrix['NAT AA'] = data_matrix.apply(lambda row: adamic_adar_index(NAT_NW, row['NAT Source Port'], row['NAT Destination Port']), axis = 1)
        data_matrix['Host Source PR'] = data_matrix.apply(lambda row: host_page_rank.get(row['Source Port'], 0), axis = 1)
        data_matrix['Host Destination PR'] = data_matrix.apply(lambda row: host_page_rank.get(row['Destination Port'], 0), axis = 1)
        data_matrix['NAT Source PR'] = data_matrix.apply(lambda row: nat_page_rank.get(row['NAT Source Port'], 0), axis = 1)
        data_matrix['NAT Destination PR'] = data_matrix.apply(lambda row: nat_page_rank.get(row['NAT Destination Port'], 0), axis = 1)
        # scaling the data
        data_matrix_robust_scaled = robust_scaler.transform(data_matrix[robust_scaler_features])
        data_matrix_std_scaled = std_scaler.transform(data_matrix[std_scaler_features])
        data_matrix_preprocessed = np.hstack((data_matrix[data_matrix.columns[:4]], data_matrix_robust_scaled, data_matrix_std_scaled, data_matrix[['Source Port Translation', 'Destination Port Translation']]))
        return data_matrix_preprocessed
    except:
        print("The last dimension of the data should be 11")



@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the firewall Action depending on the feature_matrix of firewall logs

    Returns:
        Predicted Action class
    """
    inp = request.form.to_dict()
    feature_vector = np.array([inp['source_p'], inp['dest_p'], inp['nat_source_p'], inp['nat_dest_p'], inp['bytes'], inp['bts_sent'], inp['bts_received'], inp['pkts'], inp['elapsed_t'], inp['pkts_sent'], inp['pkts_received']])
    data_preprocessed = preprocess(feature_vector.reshape(1, -1))
    prediction = calibrator_lgbm.predict(data_preprocessed.reshape(1, -1))
    return jsonify({'Predicted Action': str(prediction)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)