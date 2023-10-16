import ROOT
import os
import glob
from tqdm import tqdm

target_dir = "/lustre/collider/mocen/project/multilepton/data/"
data_dirs = glob.glob("/lustre/collider/mocen/project/multilepton/data/alldata/*/*/")

for dr in tqdm(data_dirs):
    hadd_list = ""
    for rf in glob.glob(dr+"/*.root"):
        try:
            f = ROOT.TFile(rf)
            entries = f.Get("nominal").GetEntries()
            f.Close()
        except:
            print(f"File {f} opening error.")
            entries = 0

        if entries==0:
            # # delete the empty file
            # os.system(f"rm {rf}")
            hadd_list += ""
        else:
            hadd_list += rf + " " 
    
    if hadd_list=="":
        continue
    
    sections = dr.split('/')
    tmp_target_dir = f"{target_dir}/{sections[-3]}/{sections[-2]}/"
    os.makedirs(tmp_target_dir, exist_ok=True)
    os.system(f"hadd {tmp_target_dir}/data.root {hadd_list}")