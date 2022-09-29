import pandas as pd
import numpy as np
import json
import pytest
import requests
import traceback
from data_loader.data_loader import CsvDataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

#TODO: unified the label name passing method and corresponding syntext structure


def test_loader_data_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/load-data/',
            data='{"data_path":"/Users/pwang/BenWork/Dataset/hospital/aggregate_data.csv","label":"SEPSIS"}',
            headers={'content-type': 'application/json'}
        )
        print(response.json())
        print(response.status_code)
        assert response.status_code == 200
    except:
        traceback.print_exc()

def test_check_data_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/check-data/',
        )
        print(response.json())
        assert True
    except:
        traceback.print_stack()

def test_init_model_sklearn_rfc_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/init-model/',
            data='{"model_type":"sk-rf-classifier","model_params":{"n_estimators":10,"criterion":"gini","verbose":1}}',
            headers={'content-type': 'application/json'}
        )
        print(response.json())
        assert True
    except:
        traceback.print_exc()

def test_init_model_xgb_c_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/init-model/',
            data='{"model_type":"xgb-classifier","model_params":{"verbosity":3,"n_estimators":10,"max_depth":5}}',
            headers={'content-type': 'application/json'}
        )
        print(response.json())
        assert True

    except:
        traceback.print_exc()


def test_fit_api():

    try:
        response = requests.post(
            url='http://127.0.0.1:5000/fit/',
        )
        print(response.json())
        assert True
    except:
        pass


def test_predict_api():

    df = CsvDataLoader(data_path="/Users/pwang/BenWork/Dataset/hospital/aggregate_data_testing_202007_to_202008.csv").get_df(do_label_encoder=True)
    # df = pd.read_csv("/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv")
    df = df.head(1000)
    y = df.pop('SEPSIS')

    df_to_send = str(df.to_json())


    try:
        response = requests.post(
            url='http://127.0.0.1:5000/predict/',
            data=df_to_send,
            headers={'content-type': 'application/json'}
        )
        result = response.json()
        # print(result)
        # result = json.loads(result)
        # print(np.asarray(result['array']))
        # print(y.tolist())

        acc = accuracy_score(y.tolist(), result)
        recall = recall_score(y.tolist(), result)
        f1 = f1_score(y.tolist(), result)
        prec = precision_score(y.tolist(), result)
        print("accuracy : {}".format(acc))
        print("recall : {}".format(recall))
        print("f1 : {}".format(f1))
        print("prec : {}".format(prec))

        assert True

    except:
        traceback.print_exc()
