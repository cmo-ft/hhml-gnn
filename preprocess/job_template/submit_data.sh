#!/bin/bash
source_dir="/publicfs/atlas/atlasnew/higgs/HHML/Data/*040621_data*/"
target_dir="/cefs/higgs/mocen/project/atlas/multilepton/sample/data/"
log_dir=./

skip_num=0
# end_num=1097

i=0
for datadir in `ls -d ${source_dir}/`; do
	# e.g.: datadir=/publicfs/atlas/atlasnew/higgs/HHML/Data/group.phys-hdbs.AllYear.physics_Main.DAOD_HIGG8D1.grp15_v01_p4309.HHML_040621_data15_v5_output_root
	cd $target_dir
	tmp=`basename $datadir`
	tmp=20`echo $tmp |  awk -F 'grp' '{split($2, a, "_"); print a[1]}'`
	mkdir -p $tmp && cd $tmp

	# loop all root files
	for rootfile in `ls $datadir/*.root`; do
		i=$(($i +  1))
		if [[ $i -le $skip_num ]]; then continue; fi
		
		hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/task.sh -o ${log_dir}/out -e ${log_dir}/err -argu "${rootfile} data"
		# hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/jobs/roottask.sh -mem 3000  -o ${log_dir}/out -e ${log_dir}/err -argu ${rootfile}
		# hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/jobs/roottask.sh -wt short -mem 3000  -o ${log_dir}/out -e ${log_dir}/err -argu ${rootfile}

		# if [[ $i == $end_num ]]; then exit; fi
	done
done

