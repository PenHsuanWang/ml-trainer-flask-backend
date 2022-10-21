from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, Response
import json
from json import JSONEncoder
import traceback

import pandas
import numpy

from ml_trainer.data_loader.data_loader import CsvDataLoader
from ml_trainer.model.model import ModelSklearnRFClassifier, ModelXGBClassifier


class ModelTrainerServer:

    def __init__(self):

        """
        The Model Trainer Server implemented by Flask backend.
        Manage the Model Training and expose several API for invoke.
        class level object include `data_loader`, `model`, `dataframe`, `label series`, and `Flask`
        """

        self._data_loader = None
        self._model = None
        self._df_x = None
        self._y = None


        # TODO: MLFlow

        self._MODEL_KEY_NAME_MAPPING = {
            "sk-rf-classifier": ModelSklearnRFClassifier,
            "xgb-classifier": ModelXGBClassifier
        }

        self.app = Flask(__name__)


    def ml_trainer_hello_message(self):  # put application's code here
        # return render_template('home/index.html')
        return 'Hello, this is the app for ml model trainer implement by callback function.<br>' \
               'Provide the data path and initialization the model configuration to do model training.'

    def load_data(self):
        """
        Implementation of API to receive `data_path` for data loading into class level `DataFrame` object
        :return:
        """

        try:
            data_path = request.get_json()['data_path']
            label_name = request.get_json()['label']
            print("target data path is {}".format(data_path))

            if data_path.split(".")[-1] == 'csv':
                self._data_loader = CsvDataLoader(data_path=data_path)

                self._df_x = self._data_loader.get_df(do_label_encoder=True).drop(columns=[label_name])
                self._y = self._data_loader.get_column(label_name)

                return Response(
                    json.dumps(
                        'load {} data success'.format(data_path)
                    ),
                    status=200,
                    headers={'content-type': 'application/json'}
                )

        except KeyError:
            traceback.print_exc()
            return Response(
                json.dumps(
                    'The key name error, please specified the keys `data_path` and `label` during the request sending.\n' +
                    'e.g. \n'
                    'data = {"data_path":<path-of-datafile-going-to-load>, "label":<label-name>}'
                ),
                status=503,
                headers={'content-type': 'application/json'}
            )

        except:
            traceback.print_exc()

        return Response(
            status=200
        )

    def check_data(self):
        """
        Looking the head of loaded data to check the status of dataloader is done successfully.
        Using REST API to invoke this function. provided the key `head_limit` to specify the limit of head to check, default limit is 5
        :return:
        """
        limit = None
        try:
            limit = request.get_json()['head_limit']
        except KeyError:
            print("do not found the head_limit key from the http request. default limit is 5")

        if limit is None:
            limit = 5

        print(self._df_x.head(limit))
        print(self._y.head(limit))

        return Response(
            json.dumps(
                str(self._df_x.head(limit)) +
                '\n' +
                str(self._y.head(limit))
            ),
            status=200,
            headers={'content-type':'application/json'}
        )

    def init_model(self):

        """
        Implement function of the model initialization. Setup the function for following fitting and prediction usage.
        Using REST API to invoke this function, provided the key
        `model_type`: specified the model, `sklearn`, `XGBoost` is supported.
        `model_params`: specified the model hyperparameter.

        :return:
        """

        print("going to initialized the model")

        try:
            model_type = request.get_json()['model_type']
            model_params = request.get_json()['model_params']
            scale_pos_weight = self._data_loader.get_pos_weight("SEPSIS", "True", "False")

            print("Model type: {}".format(model_type))

            model_class = self._MODEL_KEY_NAME_MAPPING.get(model_type)

            if model_class is None:
                print("Error: Model {} not found in table".format(model_type))
                print("please choose one of the the model key names below, which is available in this service")
                print((self._MODEL_KEY_NAME_MAPPING.keys()))

            try:
                """
                Model initialization, passing parameters extracted from http request.
                """
                model = model_class(
                    **model_params,
                    scale_pos_weight=scale_pos_weight
                )

                self._model = model
                print("finish of model initialization")

                return Response(
                    json.dumps("finish of model initialization"),
                    status=200,
                    headers={'content-type': 'application/json'}
                )


            except:
                print("fail to initialize the model {}".format(model_class))
                traceback.print_exc()


        except:
            traceback.print_exc()


    def fit(self):

        print("going to fit the model")

        try:

            # MLFlow
            self._model.fit(self._df_x, self._y)
            #
            print("trained model successfully")

            return Response(
                status=200
            )

        except:
            traceback.print_exc()
            print('Unexpect error happen when doing model fitting')


    def predict(self):

        class NumpyArrayEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, numpy.ndarray):
                    return obj.tolist()
                return JSONEncoder.default(self, obj)

        try:
            data = request.get_json()

            data = json.dumps(data)
            df = pandas.read_json(data)
            result = self._model.predict(df)

            return Response(
                json.dumps(result, cls=NumpyArrayEncoder),
                status=200,
                headers={'content-type': 'application/json'}
            )

        except:
            traceback.print_exc()



    def run(self):
        self.app.add_url_rule(rule='/', endpoint='ml_training_message', view_func=self.ml_trainer_hello_message)
        self.app.add_url_rule(rule='/load-data/', endpoint='load_data', view_func=self.load_data, methods=['POST'])
        self.app.add_url_rule(rule='/check-data/', endpoint='check_data', view_func=self.check_data, methods=['POST'])
        self.app.add_url_rule(rule='/init-model/', endpoint='init_model', view_func=self.init_model, methods=['POST'])
        self.app.add_url_rule(rule='/fit/', endpoint='fit', view_func=self.fit, methods=['POST'])
        self.app.add_url_rule(rule='/predict/', endpoint='predict', view_func=self.predict, methods=['POST'])
        print("Start flask server")
        self.app.run()





if __name__ == '__main__':

    server = ModelTrainerServer()
    server.run()


