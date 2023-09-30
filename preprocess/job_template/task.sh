#!/bin/bash
source ~/cvmfs.env
rootfile=${1}
# rootfile=/publicfs/atlas/atlasnew/higgs/HHML/v5/mc16a/user.sparajul.700168.Sh_2210_ttW.DAOD_HIGG8D1.e8273_s3126_r9364_p4416.HHML_v5_19122022_mc16a_output_root/user.sparajul.31752801._000032.output.root
echo $rootfile
python /cefs/higgs/mocen/project/atlas/multilepton/hhml-gnn/preprocess/main.py -i ${rootfile} -o ./
