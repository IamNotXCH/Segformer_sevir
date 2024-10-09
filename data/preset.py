import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class SEVIRDataset(Dataset):
    def __init__(self, sevir_catalog, sevir_data_dir, seq_len=25):
        self.catalog = pd.read_csv(sevir_catalog)
        self.data_dir = sevir_data_dir
        self.seq_len = seq_len

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        # 获取每种类型的数据文件路径
        event = self.catalog.iloc[idx]
        vil_data = self.load_data(event, 'vil')
        ir069_data = self.load_data(event, 'ir069')
        ir107_data = self.load_data(event, 'ir107')
        vis_data = self.load_data(event, 'vis')
        lght_data = self.load_data(event, 'lght')

        # 数据预处理和标准化
        vil_data = self.preprocess_data(vil_data)
        ir069_data = self.preprocess_data(ir069_data)
        ir107_data = self.preprocess_data(ir107_data)
        vis_data = self.preprocess_data(vis_data)
        lght_data = self.preprocess_data(lght_data)

        # 将五种数据类型合并为一个Tensor
        data = np.stack([vil_data, ir069_data, ir107_data, vis_data, lght_data], axis=0)

        return torch.tensor(data, dtype=torch.float32)

    def load_data(self, event, data_type):
        # 根据数据类型和事件信息加载HDF5数据
        file_path = f"{self.data_dir}/{data_type}/{event['file_name']}"
        with h5py.File(file_path, 'r') as f:
            data = f[data_type][:self.seq_len]  # 裁剪到指定的时间序列长度
        return data

    def preprocess_data(self, data):
        # 数据标准化，这里假设每个数据类型都需要不同的标准化处理
        return (data - np.mean(data)) / np.std(data)

# 创建DataLoader
dataset = SEVIRDataset(sevir_catalog='path_to_catalog.csv', sevir_data_dir='path_to_data')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

