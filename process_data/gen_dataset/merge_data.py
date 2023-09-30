import torch
import torch_geometric
import numpy as np
import glob
import os

class data_source:
    def __init__(self, data_file_list) -> None:
        self.data_file_list = data_file_list 
    def get_data_list(self, ):
        if  not hasattr(self, 'data_list'):
            self.data_list = sum([torch.load(data).to_data_list() for data in self.data_file_list],[])
        return self.data_list
    @staticmethod
    def save_data_lists_as_slices(data_list, dest_file_name):
        data, slices, _ = torch_geometric.data.collate.collate(data_list[0].__class__,
                    data_list=data_list, increment=False,add_batch=False)
        torch.save((data, slices), f"{dest_file_name}")
    @staticmethod
    def save_data_lists_as_batch(data_list, dest_file_name):
        torch.save(torch_geometric.data.Batch.from_data_list(data_list), dest_file_name)

# save data in the following format
source_dir = "/lustre/collider/mocen/project/multilepton/data/"
data_format = "dataset_SR%i.pt"
dest_data_format = "../dataset%i.pt"

sample_name_format = "sample_name%i.npy"
dest_name_format = "../sample_name%i.npy"

fold = 3
for i in range(fold):
    data_files = glob.glob(source_dir+"/*/*/pt_data/" + data_format%i)
    source = data_source(data_files)
    data_source.save_data_lists_as_batch(source.get_data_list(), dest_data_format%i)

    name_files = glob.glob(source_dir+"/*/*/pt_data/" + sample_name_format%i)
    sample_name = [np.load(f) for f in name_files]
    sample_name = np.concatenate(sample_name)
    np.save(dest_name_format%i, sample_name)
    
# data_format = "dataset.pt"
# data_files = glob.glob(source_dir+"/*/*/pt_data/" + data_format)
# source = data_source(data_files)
# data_source.save_data_lists_as_batch(source.get_data_list(), "./tmp.pt")