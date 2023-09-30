"""
Fix Me: define your own dataloader with the function: get_dataloader
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch_geometric
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
import copy
import warnings
import random
import math
from config import args
from collections import deque

class MyDataset(torch_geometric.data.Dataset):
    def __init__(self, data_file, transform=None, pre_transform=None, pre_filter=None):
        self.data_file = data_file
        super().__init__(self.data_file, transform, pre_transform, pre_filter)
        data = torch.load(data_file)
        # data = torch.load(f'{self.data_dir}/dataset{data_id}.pt')

        self.weight = data.sample_weight.clip(min=0) * (data.y*1e3 + 1)
        # self.weight = data.sample_weight.abs() * (data.y*1e3 + 1)
        self.data = data

    @property
    def processed_file_names(self):
        return [self.data_file]


    def len(self) -> int:
        self.size = len(self._data)
        return len(self._data)

    def get(self, idx: int) -> Data:
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])
        
        data = self._data[idx].clone()
        data.weight = self.weight[idx]
        # data.edge_attr = data.edge_attr * 0.
        self._data_list[idx] = data
        return copy.copy(self._data_list[idx])
        # return  data

    @property
    def data(self) -> Any:
        msg1 = ("It is not recommended to directly access the internal "
                "storage format `data` of an 'InMemoryDataset'.")
        msg2 = ("The given 'InMemoryDataset' only references a subset of "
                "examples of the full dataset, but 'data' will contain "
                "information of the full dataset.")
        msg3 = ("The data of the dataset is already cached, so any "
                "modifications to `data` will not be reflected when accessing "
                "its elements. Clearing the cache now by removing all "
                "elements in `dataset._data_list`.")
        msg4 = ("If you are absolutely certain what you are doing, access the "
                "internal storage via `InMemoryDataset._data` instead to "
                "suppress this warning. Alternatively, you can access stacked "
                "individual attributes of every graph via "
                "`dataset.{attr_name}`.")
        msg = msg1
        if self._indices is not None:
            msg += f' {msg2}'
        if self._data_list is not None:
            msg += f' {msg3}'
            self._data_list = None
        msg += f' {msg4}'

        warnings.warn(msg)
        return self._data

    @data.setter
    def data(self, value: Any):
        self._data = value
        self._data_list = None


# constants
n_folds= args.num_slices_train + args.num_slices_test + args.num_slices_apply
idx = deque(range(n_folds))
idx.rotate(args.ifold)
idx_dict = {
    'train': list(idx)[:-2],
    'test': list(idx)[-2],
    'apply': list(idx)[-1],
}


if args.apply_only == 0:
    trainset_list = [MyDataset(f'{args.fileList[0]}/dataset{dataid}.pt') for dataid in idx_dict['train']]
    test_id = idx_dict["test"]
    testset = MyDataset( f'{args.fileList[0]}/dataset{test_id}.pt') 

if args.apply_file_list==0:
    apply_id = idx_dict["apply"]
    applyset = MyDataset( f'{args.fileList[0]}/dataset{apply_id}.pt') 
else:
    applyset = [MyDataset(f) for f in args.apply_file_list]

def get_dataloader(loaderType, data_slice_id, num_slices, data_size, batch_size):
    loader = None
    if loaderType=="train":
        loader = torch_geometric.loader.DataLoader(trainset_list[data_slice_id], batch_size=batch_size, shuffle=True)
    elif loaderType=='test':
        loader = torch_geometric.loader.DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif loaderType=='apply':
        loader = torch_geometric.loader.DataLoader(applyset[data_slice_id], batch_size=batch_size, shuffle=False)
    return loader
