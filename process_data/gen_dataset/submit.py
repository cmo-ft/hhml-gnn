import htcondor
import os
import itertools
import time
import shutil
import glob

max_materialize = 495

def wait_for_complete(wait_time, constraint, schedd, itemdata, submit_result,sub_job):
    time.sleep(1)
    print(constraint)
    while True:
        ads = schedd.query(
            constraint=constraint,
            projection=["ClusterId", "ProcId", "Out", "JobStatus"],
        )
        if len(itemdata) == 0: return
        if len(ads) < max_materialize:
            sub_data = itemdata[:max_materialize - len(ads)]
            print(len(itemdata))
            submit_result += [schedd.submit(sub_job, itemdata=iter(sub_data))]
            print(f"==> Submitting {len(sub_data)} jobs to cluster {submit_result[-1].cluster()}")
            itemdata = itemdata[max_materialize - len(ads):]
            constraint = '||'.join([f'ClusterId == {id.cluster()}' for id in submit_result])
            print(len(itemdata))
            # print(constraint)

        n_runs = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] == 2])
        n_idle = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] == 1])
        n_other = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] > 2])
        print(f"-- {n_idle} idle, {n_runs}/{init_N} running ({len(itemdata)} left)... (wait for another {wait_time} seconds)")
        if n_other > 0:
            print(f"-- {n_other} jobs in other status, please check")
        if n_other > 0 and (n_runs + n_idle == 0):
            print(f"-- {n_other} jobs in other status, other's done, please check")
            return

        time.sleep(wait_time)


cur_dir=os.getcwd()
schedd = htcondor.Schedd()

    # submit jobs using htcondor
sub_job = htcondor.Submit({
        "executable": "/lustre/collider/mocen/software/condaenv/hailing/bin/python",
        "arguments": f"./gen_data_from_ROOT.py -i $(in_file) -o $(out_file)",
        "output": f"log/$(job_tag)_$(cur).out",
        "error": f"log/$(job_tag)_$(cur).err",
        # "log": f"log/$(job_tag)_$(ProcID).log",
        "log": f"log/$(job_tag)_$(ClusterID).log",
        "rank": '(OpSysName == "CentOS")',
        'initialdir': "./",
        "getenv": 'True',
})

submit_result = []
in_files = glob.glob("/lustre/collider/mocen/project/multilepton/data/*/*/*.root")
out_files = [os.path.dirname(f)+"/dataset.pt" for f in in_files]
itemdata = []
for f in in_files:
    dir_names = f.split('/')
    mctype, dsid = dir_names[-3], dir_names[-2]
    out_file = os.path.splitext(f)[0] + ".pt"
    itemdata += [
        {'cur': f'{mctype}_{dsid}', 'job_tag': 'gen_dataset', 'in_file': f, 'out_file': out_file}
    ]

init_N = len(itemdata)

sub_data = itemdata[:max_materialize]
submit_result += [schedd.submit(sub_job, itemdata=iter(sub_data))]
print(f"==> Submitting {len(sub_data)} jobs to cluster {submit_result[-1].cluster()}")

itemdata = itemdata[max_materialize:]
print(len(itemdata))

constraint = '||'.join([f'ClusterId == {id.cluster()}' for id in submit_result])

# waiting for job complete
# wait_for_complete(15, submit_result.cluster(), schedd, itemdata)
wait_for_complete(30, constraint, schedd, itemdata, submit_result, sub_job)
