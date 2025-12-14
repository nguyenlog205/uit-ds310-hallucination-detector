from utils import load_configs
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset():
    def __init__(
        self,
        config_path: str
    ):
        config = load_configs(config_path)
        self.dataset_path = config['dataset_path']
        self.dataset = pd.read_csv(self.dataset_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.iloc[index]

