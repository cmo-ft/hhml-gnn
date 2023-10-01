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
parser.add_argument('-i', '--input', required=True, type=str, help='path of input .pt file')
parser.add_argument('-o', '--output', type=str, help='output directory')
args = parser.parse_args()



source_data_pt = args.input                                 # e.g. /path/to/user.sparajul.31752835._000002.output_output.pt
source_data_prefix = os.path.splitext(source_data_pt)[0]    # e.g. /path/to/user.sparajul.31752835._000002.output_output
source_data_root = source_data_prefix + '.root'             # e.g. /path/to/user.sparajul.31752835._000002.output_output.root
data_prefix = os.path.basename(source_data_prefix)          # e.g. user.sparajul.31752835._000002.output_output.pt
target_dir = args.output
target_file = target_dir + data_prefix + ".root"            # e.g. /target/dir/user.sparajul.31752835._000002.output_output.root

print(f'From {source_data_root} to {target_file}')

rdf = RDF("nominal", source_data_root)
columns = list(rdf.GetColumnNames())


# load score into cppyy
cpp_define_load_data = """
void load_data(string datafilename, vector<double>& mydata){
    std::ifstream input(datafilename);
    if (!input) {
        std::cerr << "Failed to open data.txt" << std::endl;
        return;
    }
    float value;
    while (input >> value) {
        mydata.push_back(value);
    }
    input.close();
}
"""
ROOT.gInterpreter.Declare(cpp_define_load_data)
for i in range(nfolds):
    ROOT.gInterpreter.ProcessLine(f"vector<double> score{i};")
    tmpscore = np.load(source_data_prefix+f"_score{i}.npy")
    np.savetxt(f"{data_prefix}{i}.txt", tmpscore, delimiter="\n")
    ROOT.gInterpreter.ProcessLine(f'load_data("{data_prefix}{i}.txt", score{i});')
    os.remove(f"{data_prefix}{i}.txt")

    
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

rdf.Snapshot("nominal", target_file, columns, snap_option)


