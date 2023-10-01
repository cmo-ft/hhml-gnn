import uproot as ur
import glob
import torch_geometric as tg
# from torch_cluster import knn_graph
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os


parser = argparse.ArgumentParser(description='Analysis Framework for HH->Multilepton-> 3 leptons')
parser.add_argument('-i', '--input', type=str, default='./data.root', help='input root file')
parser.add_argument('-o', '--output', type=str, default='./', help='output pt file')
args = parser.parse_args()


filename = args.input
target_filename = args.output
dsid = os.path.basename(os.path.dirname(filename))
dsid = int(dsid)

lep_list = [0,1,2]
# global features and node features of graph
ft_name_glob = ["met_met", "HT", "HT_lep", "HT_jets", "minOSMll", "minOSSFMll", 'M_lll', 'M_llljj']
ft_name_PtEtaPhiE = [[f"lep_Pt_{x}", f"lep_Eta_{x}", f"lep_Phi_{x}", f"lep_E_{x}"] for x in lep_list] + \
        [[f"{x}lead_jetPt", f"{x}lead_jetEta", f"{x}lead_jetPhi", f"{x}lead_jetE"] for x in ['', 'sub']]
ft_name_PtEtaPhiE = [item for sublist in ft_name_PtEtaPhiE for item in sublist]

ft_name_parId = [[f"lep_Charge_{x}", f"lep_Mass_{x}"] for x in lep_list] 
ft_name_parId = [item for sublist in ft_name_parId for item in sublist]


edges = [[], []]
name_map = ['l0', 'l1', 'l2', 'j0', 'j1']
ft_name_edge = []
for i in range(5):
    for j in range(5):
        if i==j or (i>2 and j>2):
            continue
        edges[1] += [i]
        edges[0] += [j]
        suffix = (name_map[j]+name_map[i]) if i>j else (name_map[i]+name_map[j])
        ft_name_edge += ["dR_" + suffix]
        ft_name_edge += ["M_" + suffix]

edges = torch.tensor(edges)     


# generate datalist for a root file
def generate_pt_list(root_file_name: str):
    pt_list = []
    sample_name_list = []
    try:
        f = ur.open(root_file_name)
        data = f['nominal'].arrays(library='pd')#.groupby('entry').first()
    except:
        print(f"Error: file {root_file_name} cannot be opened", flush=True)
        data = pd.DataFrame([])
    if len(data)!=0:    
        for ievent in range(len(data)):
            sample_name = data.loc[ievent,"Sample_Name"]
            y = 1 if ("HH" in sample_name) else 0
            EvtNum = torch.tensor(data.loc[ievent, 'EvtNum'])
            if_SR = torch.tensor(data.loc[ievent, "Evt_SR"]==1)
            # ifold = int(data.loc[ievent, 'EvtNum'] % nfolds)
            # sample weight
            sample_weight = data.loc[ievent, 'weight']
            # global feature
            ft_glob = torch.tensor(data.loc[ievent, ft_name_glob]).view(1,-1).float()

            # node feature
            ft_lzvec = torch.tensor(data.loc[ievent, ft_name_PtEtaPhiE]).view(5, 4)
            ft_parId = torch.zeros(5,2)
            ft_parId[:3] = torch.tensor(data.loc[ievent, ft_name_parId]).view(3,2)
            ft_parId = ft_parId.view(5,2)

            ft_nodes = torch.cat([ft_parId, ft_lzvec], dim=1).float()

            # edge feature
            ft_edge = torch.tensor(data.loc[ievent, ft_name_edge]).view(-1,2).float()

            pt = tg.data.Data(
                x=ft_nodes, edge_index=edges, u=ft_glob, y=y, edge_attr=ft_edge, sample_weight=sample_weight,
                EvtNum=EvtNum, sr=if_SR, dsid=dsid
                )
            pt_list.append(pt)


    return pt_list, sample_name_list



# store sample name
pt_list, sample_name_list = generate_pt_list(filename)

if len(pt_list)!=0:
    torch.save(tg.data.Batch.from_data_list(pt_list), target_filename)
print(f'from {filename} to {target_filename} done.')
