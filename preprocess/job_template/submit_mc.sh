#!/bin/bash
source_dir="/publicfs/atlas/atlasnew/higgs/HHML/v5/"
source_dir=${source_dir}"/mc16*/"
target_dir="/cefs/higgs/mocen/project/atlas/multilepton/sample/mc/"
log_dir=./

skip_num=0
# end_num=1097

i=0
for mcdir in `ls -d $source_dir/`; do
	cd $target_dir
	tmp=`basename $mcdir`
	mkdir -p $tmp && cd $tmp

	# loop all channel id directory
	for datadir in `ls -d $mcdir/*_root/`; do
		# e.g.: datadir=/publicfs/atlas/atlasnew/higgs/HHML/v5/mc16a/user.sparajul.700168.Sh_2210_ttW.DAOD_HIGG8D1.e8273_s3126_r9364_p4416.HHML_v5_19122022_mc16a_output_root/
		tmp=`basename $datadir`
		channelId=`echo $tmp | awk -F'.' '{print $3}'`
		if [[ `ls ${channelId}/*.root 2>/dev/null | wc -l ` -gt 0 ]]; then continue; fi
		mkdir -p $channelId && cd $channelId

		# loop all root files
		for rootfile in `ls $datadir/*.root`; do
			i=$(($i +  1))
			if [[ $i -le $skip_num ]]; then continue; fi
			
			hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/task.sh -o ${log_dir}/out -e ${log_dir}/err -argu ${rootfile}
			# hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/jobs/roottask.sh -mem 3000  -o ${log_dir}/out -e ${log_dir}/err -argu ${rootfile}
			# hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/jobs/roottask.sh -wt short -mem 3000  -o ${log_dir}/out -e ${log_dir}/err -argu ${rootfile}

			# if [[ $i == $end_num ]]; then exit; fi
		done
		cd ..
	done
	sleep 1h
done



# source_dir="/publicfs/atlas/atlasnew/higgs/HHML/v5/mc16a/user.sparajul.410470.PhPy8EG.DAOD_HIGG8D1.e6337_s3126_r9364_p4308.HHML_v5_19122022_mc16a_output_root/"
# source_dir="/publicfs/atlas/atlasnew/higgs/HHML/v5/mc16a//user.sparajul.410472.PhPy8EG.DAOD_HIGG8D1.e6348_s3126_r9364_p4308.HHML_v5_19122022_mc16a_output_root/"
# target_dir="/cefs/higgs/mocen/project/atlas/multilepton/data/mc16a/"

# source_dir="/publicfs/atlas/atlasnew/higgs/HHML/v5/mc16d//user.sparajul.410470.PhPy8EG.DAOD_HIGG8D1.e6337_s3126_r10201_p4308.HHML_v5_19122022_1_mc16d_output_root/"
# target_dir="/cefs/higgs/mocen/project/atlas/multilepton/data/mc16d/"

# source_dir="/publicfs/atlas/atlasnew/higgs/HHML/v5/mc16e//group.phys-hdbs.410470.PhPy8EG.DAOD_HIGG8D1.e6337_s3126_r10724_p4308.HHML_v5_19122022_sys_e_output_root/"
# target_dir="/cefs/higgs/mocen/project/atlas/multilepton/data/mc16e/"


# ids=(113 149 158 344 368 402 428 457 508 591)
# for id in ${ids[@]}; do
# idx="000${id}"
# echo $idx
# for rootfile in `ls $source_dir/*${idx}*.root`; do
# 	i=$(($i +  1))
# 	if [[ $i -le $skip_num ]]; then continue; fi
# 	hep_sub /cefs/higgs/mocen/project/atlas/multilepton/preprocess/jobs/roottask.sh -mem 3000  -o ${log_dir}/out -e ${log_dir}/err -argu ${rootfile}
# done
# done
