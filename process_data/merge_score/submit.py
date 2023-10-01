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




# submit jobs using htcondor
sub_job = htcondor.Submit({
        "executable": "./run.sh",
        "arguments": f" $(data_pt) $(out_dir)",
        "output": f"log/$(job_tag)_$(cur).out",
        "error": f"log/$(job_tag)_$(cur).err",
        "log": f"log/$(job_tag)_$(ClusterID).log",
        "rank": '(OpSysName == "CentOS")',
        'initialdir': "./",
        "getenv": 'True',
})


cur_dir=os.getcwd()
souce_dir='/lustre/collider/mocen/project/multilepton/data/'
out_dir="/lustre/collider/mocen/project/multilepton/dnn/data_with_score/"
schedd = htcondor.Schedd()


submit_result = []
itemdata = []
datatype = ['mc16a', 'mc16d', 'mc16e', 'data']
for dt in datatype:
    for dp in glob.glob(souce_dir+f"{dt}/*/*.pt"):
        # e.g.: data_dir="/lustre/collider/mocen/project/multilepton/data/mc16a/304014/"
        data_dir = os.path.dirname(dp)

        dsid = os.path.basename(data_dir)
        itemdata += [
            {'cur': f'{dt}_{dsid}', 'job_tag': 'merge_score', 'data_pt': dp, 'out_dir': out_dir}
        ]


init_N = len(itemdata)

sub_data = itemdata[:max_materialize]
submit_result += [schedd.submit(sub_job, itemdata=iter(sub_data))]
print(f"==> Submitting {len(sub_data)} jobs to cluster {submit_result[-1].cluster()}")

itemdata = itemdata[max_materialize:]
print(len(itemdata))

constraint = '||'.join([f'ClusterId == {id.cluster()}' for id in submit_result])

# # waiting for job complete
wait_for_complete(30, constraint, schedd, itemdata, submit_result, sub_job)