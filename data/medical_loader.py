import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

class StrokeDataset(Dataset):
    def __init__(self, file_path="./datasets/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(file_path)

        self.data.drop(columns=["id"], inplace=True)

        self.data["bmi"] = self.data["bmi"].fillna(self.data["bmi"].median())

        categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        categorical_features = encoder.fit_transform(self.data[categorical_columns])
        categorical_df = pd.DataFrame(categorical_features, columns=encoder.get_feature_names_out(categorical_columns))
        self.data = pd.concat([self.data.drop(columns=categorical_columns), categorical_df], axis=1)

        self.data["age_hypertension"] = self.data["age"] * self.data["hypertension"]
        self.data["bmi_glucose"] = self.data["bmi"] * self.data["avg_glucose_level"]

        self.features = self.data.drop(columns=["stroke"]).values
        self.labels = self.data["stroke"].values

        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        dataset_size = len(self.features)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size

        self.train_data, self.test_data = random_split(
            list(zip(self.features, self.labels)), [train_size, test_size]
        )

    @property
    def input_dim(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return 2
