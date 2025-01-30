import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

class FinancialDistressDataset(Dataset):
    def __init__(self, file_path="./datasets/fin_distress.csv"):
        self.data = pd.read_csv(file_path)

        self.data.drop(columns=["Company", "Time"], inplace=True)

        self.data["Financial Distress"] = (self.data["Financial Distress"] < -0.50).astype(int)  # 1 = distressed, 0 = healthy

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        categorical_features = encoder.fit_transform(self.data[["x80"]])
        categorical_df = pd.DataFrame(categorical_features, columns=encoder.get_feature_names_out(["x80"]))
        self.data = pd.concat([self.data.drop(columns=["x80"]), categorical_df], axis=1)

        self.features = self.data.drop(columns=["Financial Distress"]).values  
        self.labels = self.data["Financial Distress"].values  

        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        smote = SMOTE()
        self.features, self.labels = smote.fit_resample(self.features, self.labels)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        dataset_size = len(self.features)
        train_size = int(0.7 * dataset_size)
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
