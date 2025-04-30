import pandas as pd
import torch
from torch.utils.data import Dataset

"""
    A custom dataset class for loading and preprocessing data for training.

    This class is designed to handle two types of datasets: 'source' and 'target'. 
    It reads data and labels from CSV files, optionally normalizes the data, and 
    prepares the data for model training. The class supports both single-label 
    (target) and multi-label (source) datasets.
    
    Attributes:
        data_path (str): Path to the CSV file containing the input data.
        label_path (str): Path to the CSV file containing the labels.
        dataset_type (str): Type of dataset, either 'target'  or 'source' .
        is_normalized (bool): Indicates whether to normalize the input data using MinMaxScaler.
        feature (list): A list to store the preprocessed feature-label pairs.
"""
class trainDataset(Dataset):
    def __init__(self, data_path=None, label_path=None, dataset_type=None):
        super(trainDataset, self).__init__()
        self.path = data_path
        self.data_path = data_path
        self.feature = []
        self.dataset_type = dataset_type
        df_1 = pd.read_csv(data_path)
        df_2 = pd.read_csv(label_path)
        if dataset_type == 'target':
            for i in range(len(df_1)):
                x_1 = df_1.iloc[i, :].values
                y_1 = df_2.iloc[i]['label']
                self.feature.append((x_1, y_1))
        if dataset_type == 'source':
            for i in range(len(df_1)):
                x_1 = df_1.iloc[i, :].values
                y_1 = df_2.iloc[i][['x', 'y']].astype(float).values
                self.feature.append((x_1, y_1))
    def __getitem__(self, item) -> tuple:
            features, data = self.feature[item]
            features = torch.from_numpy(features).float()
            data = torch.as_tensor(data)
            return features, data
    def __len__(self) -> int:
            return len(self.feature)


class ValDataset(Dataset):
    def __init__(self, data_path=None, label_path=None, dataset_type=None):
        super(ValDataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.feature = []
        df_1 = pd.read_csv(data_path)
        df_2 = pd.read_csv(label_path)
        if dataset_type == 'target':
            for i in range(len(df_1)):
                x_1 = df_1.iloc[i, :].values
                y_1 = df_2.iloc[i][['X', 'Y']].astype(float).values
                self.feature.append((x_1, y_1))
        if dataset_type == 'source':
            for i in range(len(df_1)):
                x_1 = df_1.iloc[i, :].values
                y_1 = df_2.iloc[i, :].astype(float).values
                self.feature.append((x_1, y_1))
    def __getitem__(self, item) -> tuple:
            features, data = self.feature[item]
            features = torch.from_numpy(features).float()
            return features, data
    def __len__(self) -> int:
            return len(self.feature)


"""
    A custom dataset class for loading and preprocessing data for validation of major cell type classification in spatial transcriptomics.

    This class is designed to handle two types of datasets: 'source' and 'target'. 
    It reads data and labels from CSV files, optionally normalizes the data, and 
    prepares the data for model training. The class supports both single-label 
    (target) and multi-label (source) datasets.

    Attributes:
        data_path (str): Path to the CSV file containing the input data.
        label_path (str): Path to the CSV file containing the labels.
        dataset_type (str): Type of dataset, either 'target' or 'source' .
        is_normalized (bool): Indicates whether to normalize the input data using MinMaxScaler.
        feature (list): A list to store the preprocessed feature-label pairs.
 """
class ValDataset_cls(Dataset):
    def __init__(self, data_path=None, label_path=None):
        super(ValDataset_cls, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.feature = []
        df_1 = pd.read_csv(data_path)
        df_2 = pd.read_csv(label_path)
        for i in range(len(df_1)):
            x_1 = df_1.iloc[i, :].values
            y_1 = df_2.iloc[i]['label']
            self.feature.append((x_1, y_1))
    def __getitem__(self, item) -> tuple:
            features, data = self.feature[item]
            features = torch.from_numpy(features).float()
            data = torch.as_tensor(data)
            return features, data
    def __len__(self) -> int:
            return len(self.feature)

