import htcondor
import os
import itertools
import time
import shutil

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



del os.environ['PYTHONPATH']

cur_dir=os.getcwd()
data_dir='/lustre/collider/zhangyulei/CEPC/llp/LLP_Data'
schedd = htcondor.Schedd()


#mass=['1']
#lifetime=['01']
#fit_type=['fixed', 'floated']
N = 500

    # submit jobs using htcondor
sub_job = htcondor.Submit({
        "executable": "../gnn_template/apply.sh",
        "arguments": f"$(cur)",
        "output": f"log/$(job_tag).out",
        "error": f"log/$(job_tag).err",
        "log": f"log/$(ClusterID).log",
        "rank": '(OpSysName == "CentOS")',
        # 'initialdir': cur_dir,
        "getenv": 'True',
})

ifolds = [0, 1, 2]
submit_result = []
itemdata = []
for ifold in ifolds:
    itemdata += [
        {'cur': f'{ifold}', 'job_tag': f'fold{ifold}'}
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
