import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import RobustScaler

class MalwareDataset(Dataset):
    def __init__(self, file_path="./datasets/dataset_malwares.csv"):
        self.data = pd.read_csv(file_path)

        self.data.drop(columns=["Name"], inplace=True)

        redundant_features = ["SectionMinVirtualsize", "SectionMaxVirtualsize"]
        self.data.drop(columns=redundant_features, inplace=True)

        self.labels = self.data["Malware"].values  # 0 (benign), 1 (malware)
        self.features = self.data.drop(columns=["Malware"]).values  # Features

        scaler = RobustScaler()
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
