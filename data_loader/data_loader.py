
import os

from abc import ABC

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from collections import Counter


class DataExtractor:

    def __init__(self, raw_df):

        self._complete_set_df = raw_df

    def get_df(self, do_label_encoder: True, random_sampling=-1):
        return self._complete_set_df

    def get_df_x_y(self, label: str) -> (pd.DataFrame, pd.DataFrame):
        """
        provided label name to pop from complete set of dataframe loaded by dataloader.
        :param label: str, column name of label
        :return: tuple (pd.DataFrame: x, pd.DataFrame: y)
        """

        x = self._complete_set_df
        try:
            y = x.pop(label)

            return x, y
        except:
            print("Cannot extraction label: {} from dataframe: ".format(label))
            print("complete set of dataframe : ")
            print(x)


class DataLoader(DataExtractor, ABC):

    def __init__(self, *args, **kwargs):
        """
        Abstraction of DataLoader for reading from static data.
        Different from DataAcquisitor, here is basically reading static data from file or DB table.
        :param input_data_path:
        """

        do_label_encoding = kwargs.get('do_label_encoding')
        if do_label_encoding is None:
            do_label_encoding = True

        self._raw_df = None
        self._load_df(do_label_encoding)
        super().__init__(self._raw_df)


        self._df_x = None
        self._df_y = None



    @staticmethod
    def __check_data_path_valid(data_path):
        """
        look up the provided data path is valid or not,
        implement in concrete object.
        :return:
        """
        raise NotImplementedError

    def _load_df(self, do_label_encoding: bool):
        """
        load df inner function provided to construct self._raw_df,
        based on different source, the method implement at concrete class.
        :param do_label_encoding: configuration to make loaded df do label (object type of column) encoding.
        :return:
        """
        raise NotImplementedError

    def _do_label_encoder(self):
        for col in self._raw_df.columns:
            if self._raw_df[col].dtype == 'object':
                self._raw_df[col] = self._raw_df[col].fillna(self._raw_df[col].mode())
                self._raw_df[col] = LabelEncoder().fit_transform(self._raw_df[col].astype('str'))
            else:
                self._raw_df[col] = self._raw_df[col].fillna(self._raw_df[col].median())



    def get_resample_x_y(self, label='', resampler=RandomUnderSampler(random_state=42, sampling_strategy='auto')):
        """
        resampler using imblearn module's member,
        here can be RandomUnderSampler or RandomOverSampler
        :param label:
        :param resampler:
        :return:
        """

        if resampler is not None:
            df_x, df_y = self.get_df_x_y(label=label)
            print(df_x)
            resample_df_x, resample_df_y = resampler.fit_resample(df_x, df_y)


            print("successfully done with data resampling")
            print("The shape of original dataframe: {}".format(sorted(Counter(df_y).items())))
            print("The shape after resampling: {}".format(sorted(Counter(resample_df_y).items())))

            return resample_df_x, resample_df_y

        else:
            print("please specify the data resampler, imblearn under sampler or over sampler ..")



class CsvDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        """
        implementation of data loader which responsible for csv file.

        :param args:
        :param kwargs:
        """

        # check of input file is acceptable
        self._data_path = kwargs.get('data_path')
        self.__check_data_path_valid(self._data_path)

        super(CsvDataLoader, self).__init__(*args, **kwargs)

    def _load_df(self, do_label_encoding=True):
        print("invoke from child class")
        self._raw_df = pd.read_csv(self._data_path)

        if do_label_encoding:
            print("going to do label encoding")
            self._do_label_encoder()

    @staticmethod
    def __check_data_path_valid(data_path):
        """
        check the provided data path is existed, and it is csv file
        :return:
        """

        if os.path.isfile(data_path):
            if data_path.split('.')[-1] != 'csv':
                raise RuntimeError('provided file {} is not csv'.format(data_path))
        else:
            raise FileNotFoundError('provided file {} is not exist'.format(data_path))


    def show_dataframe(self, row_limit):
        print(self._df.head(row_limit))


if __name__ == '__main__':

    loader = CsvDataLoader(data_path='/Users/pwang/BenWork/OnlineML/onlineml/data/airline/airline_data.csv')
    loader.show_dataframe(row_limit=100)

