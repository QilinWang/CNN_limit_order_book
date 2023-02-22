import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import data_prep

def StandardScaler(tensor):
    m = tensor.mean(0, keepdim=True)
    s = tensor.std(0, unbiased=False, keepdim=True)
    tensor -= m
    tensor /= s + 1e-8
    return tensor


class LobDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, df, input_width, shift, label_width, stride=1):
        self.input_width = (
            input_width  # input_width: # of time steps that are fed into the models
            # input_width_p1 + input_width_p2
        )
        self.shift = shift  # shift: # of timesteps separating the input and the (final) predictions
        self.label_width = label_width  # label_width: # of time steps in the predictions

        self.window_size = self.input_width + self.shift  # [120,24,24] -> window_size=144
        self.label_start = self.window_size - self.label_width  # [120,24,24] -> label_start=144-24=120

        self.length = df.shape[0]
        self.input_slice = slice(0, self.input_width)
        self.label_slice = slice(self.label_start, None)

        self.mask_slice = None
        if self.shift != self.label_width:
            self.mask_slice = slice(self.input_width, self.label_start)

        self.stride = stride

        # splits = [total[i:i+self.window_size] for i in range(0,self.length - self.window_size + 1,self.stride)]
        # df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

        inputs = [df[i : i + self.input_width] for i in range(0, self.length - self.window_size + 1, self.stride)]
        labels = [
            df[i + self.label_start : i + self.window_size] for i in range(0, self.length - self.window_size + 1, self.stride)
        ]

        inputs_tensor = torch.from_numpy(np.concatenate(np.expand_dims(inputs, axis=0), axis=0)).to(dtype=torch.float32)
        labels_tensor = torch.from_numpy(np.concatenate(np.expand_dims(labels, axis=0), axis=0)).to(dtype=torch.float32)
        inputs_tensor = StandardScaler(inputs_tensor)
        labels_tensor = StandardScaler(labels_tensor)

        self.X = inputs_tensor[:, :, :-1]  # mid_price not included
        self.y = labels_tensor[:, :, :-1]
        self.target = labels_tensor[:, :, -1].mean(dim=-1, keepdim=True)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.X[index], self.y[index], self.target[index]  # 


class LobData(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data_new/SB_20210625_20210810",
        batch_size: int = 32,
        input_width=120,
        shift=24,
        label_width=24,
        stride=1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_width = input_width

        self.shift = shift  # shift: # of timesteps separating the input and the (final) predictions
        self.label_width = label_width  # label_width: # of time steps in the predictions

        self.window_size = self.input_width + self.shift  # [120,24,24] -> window_size=144
        self.label_start = self.window_size - self.label_width  # [120,24,24] -> label_start=144-24=120
        self.stride = stride


    def setup(self, stage: str):
        train_df, valid_df, test_df = data_prep.read_parquet(self.data_dir)
        self.test_ds = LobDataset(test_df, input_width=self.input_width, shift=self.shift, label_width=self.label_width, stride=self.stride)
        self.train_ds = LobDataset(train_df, input_width=self.input_width, shift=self.shift, label_width=self.label_width, stride=self.stride)
        self.valid_ds = LobDataset(valid_df, input_width=self.input_width, shift=self.shift, label_width=self.label_width, stride=self.stride)

        # lob_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
