import htcondor
import os
import itertools
import time
import shutil
import glob



def wait_for_complete(wait_time, cluster_id, schedd, itemdata):
    time.sleep(15)
    while True:
        ads = schedd.query(
            constraint=f"ClusterId == {cluster_id}",
            projection=["ClusterId", "ProcId", "Out", "JobStatus"],
        )
        if len(ads) == 0: return
        n_runs = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] == 2])
        n_idle = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] == 1])
        n_other = len([ad["JobStatus"] for ad in ads if ad["JobStatus"] > 2])
        print(f"-- {n_idle} idle, {n_runs}/{len(itemdata)} running... (wait for another {wait_time} seconds)")
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
        "arguments": f"",
        "output": f"log/$(job_tag)_$(cur).out",
        "error": f"log/$(job_tag)_$(cur).err",
        # "log": f"log/$(job_tag)_$(ProcID).log",
        "log": f"log/$(job_tag)_$(ClusterID).log",
        "rank": '(OpSysName == "CentOS")',
        'initialdir': "./",
        "getenv": 'True',
        'max_materialize': 400,
        'max_idle': 50,
})

in_files = glob.glob("/lustre/collider/mocen/project/multilepton/data/*/*/data.root")
out_files = [os.path.dirname(f)+"/dataset.pt" for f in in_files]
itemdata = []
for f in in_files:
    dir_names = f.split('/')
    mctype, dsid = dir_names[-3], dir_names[-2]
    itemdata += [
        {'cur': f'{mctype}_{dsid}', 'job_tag': 'gen_dataset'}
    ]

submit_result = schedd.submit(sub_job, itemdata=iter(itemdata))
print(f"==> Submitting {len(itemdata)} jobs to cluster {submit_result.cluster()}")

# waiting for job complete
wait_for_complete(15, submit_result.cluster(), schedd, itemdata)
