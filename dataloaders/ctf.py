from typing import List
import os
import bisect
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from dataloaders.datasets import SequenceDataset, default_data_path
from ctf4science.data_module import load_dataset, load_validation_dataset

class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data, loc=None):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean

class CTFSequenceDataset(SequenceDataset):
    @property
    def n_tokens_time(self):
        # Shape of the dates: depends on `timeenc` and `freq`
        return self.dataset_train.n_tokens_time  # data_stamp.shape[-1]

    @property
    def d_input(self):
        return self.dataset_train.d_input

    @property
    def d_output(self):
        return self.dataset_train.d_output

    @property
    def l_output(self):
        return self.dataset_train.pred_len

    @staticmethod
    def collate_fn(batch, resolution, **kwargs):
        x, y, *z = zip(*batch)
        x = torch.stack(x, dim=0)[:, ::resolution]
        y = torch.stack(y, dim=0)
        z = [torch.stack(e, dim=0)[:, ::resolution] for e in z]
        return x, y, *z

    def setup(self):
        self.data_dir = default_data_path.parent.parent.parent.parent / 'data' / self._name_ / 'train'

        self.dataset_train = _Dataset_CTF(
            name=self._name_,
            flag="train",
            size=self.size,
            scale=self.scale,
            inverse=self.inverse,
            pair_id=self.pair_id,
            validation=self.validation
        )

        self.dataset_val = _Dataset_CTF(
            name=self._name_,
            flag="train",
            size=self.size,
            scale=self.scale,
            inverse=self.inverse,
            pair_id=self.pair_id,
            validation=self.validation
        )

        self.dataset_test = _Dataset_CTF(
            name=self._name_,
            flag="test",
            scale=self.scale,
            inverse=self.inverse,
            pair_id=self.pair_id,
            validation=self.validation
        )

        pass

class CTFDataset(Dataset):
    def __init__(
        self,
        name=None,
        flag="train",
        size=None,
        scale=True,
        inverse=False,
        pair_id=None,
        validation=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.name = name
        self.flag = flag
        self.set_type = type_map[flag]

        self.scale = scale
        self.inverse = inverse
        self.forecast_horizon = self.pred_len

        self.validation = validation

        self.pair_id = pair_id
        self.__read_data__()
        
        # if data_path == 'national_illness.csv':
        #     breakpoint()

    def _borders(self, df_raw):
        num_train = int(len(df_raw)) # 100% of training
        num_vali = len(df_raw) - int(len(df_raw) * 0.8) # 20% of training for validation
        border1s = [0, num_train - num_vali, 0] # 0% for testing
        border2s = [num_train, num_train, 0]

        return border1s, border2s

    def __read_data__(self):
        self.lens = list()
        self.start_idxs = list()
        self.end_idxs = list()
        self.data_x = list()
        self.data_y = list()
        self.data_stamp = list()
        self.scaler = StandardScaler()

        # Set up scaler on all data first
        if self.scale:
            all_data = list()
            if self.validation:
                data_mats, _, _ = load_validation_dataset(self.name, self.pair_id, transpose=True)
            else:
                data_mats, _ = load_dataset(self.name, self.pair_id, transpose=True)
            for i, data_mat in enumerate(data_mats):
                data_mat = np.swapaxes(data_mat, 0, 1)
                data_mat = torch.Tensor(data_mat.astype(np.float32))

                border1s, border2s = self._borders(data_mat)
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                train_data = data_mat[border1s[0] : border2s[0]]
                all_data.append(train_data)
            all_data = torch.cat(all_data)
            df_data = pd.DataFrame(all_data)
            self.scaler.fit(df_data.values)

        for i, data_mat in enumerate(data_mats):
            data_mat = np.swapaxes(data_mat, 0, 1)
            data_mat = torch.Tensor(data_mat.astype(np.float32))
            df_data = pd.DataFrame(data_mat)

            border1s, border2s = self._borders(df_data)
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                data = self.scaler.transform(df_data.values)  # Scaled down, should not be Y
            else:
                data = df_data.values

            self.data_x.append(data[border1:border2])
            if self.inverse:
                self.data_y.append(df_data.values[border1:border2])
            else:
                self.data_y.append(data[border1:border2])

            self.data_stamp.append(np.arange(0,df_data.shape[0]).reshape(-1,1))

            data_len = self.data_x[-1].shape[0]
            self.lens.append(data_len - self.seq_len - self.pred_len + 1)

            if len(self.start_idxs) == 0:
                self.start_idxs.append(0)
            else:
                self.start_idxs.append(self.end_idxs[-1]+1)
            if len(self.end_idxs) == 0:
                self.end_idxs.append(self.lens[-1]-1)
            else:
                self.end_idxs.append(self.start_idxs[-1]+self.lens[-1]-1)

    def __getitem__(self, index):
        # Get appropriate matrix data
        matrix_idx = bisect.bisect_right(self.start_idxs, index)-1
        data_x = self.data_x[matrix_idx]
        data_y = self.data_y[matrix_idx]
        data_stamp = self.data_stamp[matrix_idx]

        # Do normal __get_item__
        s_begin = index - self.start_idxs[matrix_idx]
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_x = np.concatenate(
            [seq_x, np.zeros((self.pred_len, data_x.shape[-1]))], axis=0
        )

        if self.inverse:
            seq_y = data_y[s_end:r_end]
        else:
            seq_y = data_y[s_end:r_end]

        mark = data_stamp[s_begin:s_end]
        mark = np.concatenate([mark, np.zeros((self.pred_len, mark.shape[-1]))], axis=0)

        mask = np.concatenate([np.zeros(self.seq_len), np.zeros(self.pred_len)], axis=0)
        mask = mask[:, None]

        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        mark = mark.astype(np.float32)
        mask = mask.astype(np.int64)

        return torch.tensor(seq_x), torch.tensor(seq_y), torch.tensor(mark), torch.tensor(mask)

    def __len__(self):
        return sum(self.lens)

    def inverse_transform(self, data, loc=None):
        # Need to do inverse scaling for each training matrix
        return self.scaler.inverse_transform(data, loc)

    @property
    def d_input(self):
        if self.features == 'M':
            return 1
        return self.data_x.shape[-1]

    @property
    def d_output(self):
        if self.features in ["M", "S"]:
            return self.data_x.shape[-1]
        elif self.features == "MS":
            return 1
        else:
            raise NotImplementedError

    @property
    def n_tokens_time(self):
        if self.freq == 'h':
            return [13, 32, 7, 24]
        elif self.freq == 't':
            return [13, 32, 7, 24, 4]
        else:
            raise NotImplementedError

class _Dataset_CTF(CTFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def n_tokens_time(self):
        assert self.freq == "h"
        return [13, 32, 7, 24]

class PDE_KS(CTFSequenceDataset):
    _name_ = "PDE_KS"

    init_defaults = {
        "size": None,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    names = {
        1: "X1train.mat",
        2: "X2train.mat",
        3: "X3train.mat",
        4: "X4train.mat",
        5: "X5train.mat",
        6: "X6train.mat",
        7: "X7train.mat",
        8: "X8train.mat",
        9: "X9train.mat",
        10: "X10train.mat"
    }

class ODE_Lorenz(CTFSequenceDataset):
    _name_ = "ODE_Lorenz"

    init_defaults = {
        "size": None,
        "scale": True,
        "inverse": False,
        "timeenc": 0,
        "freq": "h",
        "cols": None,
    }

    names = {
        1: "X1train.mat",
        2: "X2train.mat",
        3: "X3train.mat",
        4: "X4train.mat",
        5: "X5train.mat",
        6: "X6train.mat",
        7: "X7train.mat",
        8: "X8train.mat",
        9: "X9train.mat",
        10: "X10train.mat"
    }

def load_data(config_dataset, config_loader):
    if config_dataset['_name_'] in ["ODE_Lorenz", "Lorenz_Official"]:
        dataset = ODE_Lorenz(**config_dataset)
    elif config_dataset['_name_'] in ["PDE_KS", "KS_Official"]:
        dataset = PDE_KS(**config_dataset)
    dataset._name_ = config_dataset['_name_']
    dataset.setup()
    
    train_loader = dataset.train_dataloader(**config_loader)
    # Eval loaders are dictionaries where key is resolution, value is dataloader
    # - Borrowed from S4 dataloaders. For now just set resolution to 1
    val_loader   = dataset.val_dataloader(**config_loader)[None]
    test_loader  = dataset.test_dataloader(**config_loader)[None]
    return train_loader, val_loader, test_loader  # , dataset


def visualize_data(dataloaders, splits=['train', 'val', 'test'],
                   save=False, args=None, title=None):
    None