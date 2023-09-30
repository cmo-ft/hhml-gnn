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
souce_dir='/lustre/collider/mocen/project/multilepton/data/'
out_dir="/lustre/collider/mocen/project/multilepton/dnn/trex/apply_all/data_with_score/"
schedd = htcondor.Schedd()


# submit jobs using htcondor
sub_job = htcondor.Submit({
        "executable": "/lustre/collider/mocen/project/multilepton/dnn/trex/apply_all/merge_score/run.sh",
        "arguments": f" $(data_dir)",
        "output": f"log/$(job_tag)_$(cur).out",
        "error": f"log/$(job_tag)_$(cur).err",
        "log": f"log/$(job_tag)_$(ClusterID).log",
        "rank": '(OpSysName == "CentOS")',
        'initialdir': "/lustre/collider/mocen/project/multilepton/dnn/trex/apply_all/merge_score",
        "getenv": 'True',
        'max_materialize': 400,
        'max_idle': 50,
})



itemdata = []
datatype = ['mc16a', 'mc16d', 'mc16e']
for dt in datatype:
    for dr in glob.glob(souce_dir+f"{dt}/*/data.root"):
        # e.g.: data_dir="/lustre/collider/mocen/project/multilepton/dnn/trex/apply_all/data_with_score/"
        data_dir = os.path.dirname(dr)

        base_name = os.path.basename(data_dir)
        itemdata += [
            {'cur': f'{dt}_{base_name}', 'job_tag': 'merge_score', 'data_dir': data_dir}
        ]


submit_result = schedd.submit(sub_job, itemdata=iter(itemdata))
print(f"==> Submitting {data_dir} jobs to cluster {submit_result.cluster()}")

# waiting for job complete
wait_for_complete(15, submit_result.cluster(), schedd, itemdata)