
from scipy.special import softmax
import numpy as np
import ROOT 
import ROOT.RDataFrame as RDF
import glob
import os
import argparse
from collections import deque

snap_option = ROOT.RDF.RSnapshotOptions()
snap_option.fMode = "UPDATE"
snap_option.fOverwriteIfExists = True

nfolds = 3

parser = argparse.ArgumentParser(description='Merge score into root file')
parser.add_argument('-i', '--input', required=True, type=str, help='directory of input file')
parser.add_argument('-o', '--output', type=str, help='output directory')
args = parser.parse_args()



source_dir = args.input
target_dir = args.output

rdf = RDF("nominal", source_dir+"/data.root")
columns = list(rdf.GetColumnNames())

score = []
for i in range(nfolds):
    ROOT.gInterpreter.ProcessLine(f"vector<double> score{i};")
    out = np.load(source_dir+f"/pt_data/fold{i}/outApply_GPU0.npy")
    pred = softmax(out[:,1:], axis=1)[:,1]
    # pred = np.zeros(len(out))
    score += [pred]


for iscore in range(len(score[0])):
    for ifold in range(nfolds):
        ROOT.gInterpreter.ProcessLine(f"""
                score{ifold}.push_back({(score[ifold][iscore])});
        """)
    
idx_str = ""
for ifold in range(nfolds):
    rdf = rdf.Define(f"score{ifold}", f"score{ifold}[rdfentry_]")
    columns += [f"score{ifold}"]

    idx = deque(range(nfolds))
    idx.rotate(ifold)
    idx_dict = {
        'train': list(idx)[:-2],
        'test': list(idx)[-2],
        'apply': list(idx)[-1],
    }
    idx_str += f'( (EvtNum % {nfolds} == {idx_dict["apply"]}) * score{ifold}) {"+" if ifold < nfolds - 1 else ""} '
    # rdf = rdf.Define(f"score", f"0")
    
    
rdf = rdf.Define("score", idx_str)
columns += [f"score"]

rdf.Snapshot("nominal", target_dir+"/data.root", columns, snap_option)


