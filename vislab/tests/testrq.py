import rq
import numpy as np
import time
import subprocess
import shlex
import sys
import redis
import vislab.tests.testrqaux

def get_redis_client():
    host, port = ["0.0.0.0", 6379]
    try:
        connection = redis.Redis(host, port)
        connection.ping()
    except redis.ConnectionError:
        raise Exception(
            "Need a Redis server running on {}, port {}".format(host, port))
    return connection



def run_workers_cmd(q_name, num_workers):
    print("Starting {} workers...".format(num_workers))
    cmd = "rqworker --burst {}".format(q_name)
    print(cmd)
    pids = []
    for i in range(num_workers):
        # time.sleep(np.random.rand())  # stagger the jobs a little bit
        pids.append(subprocess.Popen(shlex.split(cmd),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE))

        out, err = pids[-1].communicate()
        print '==='

        print out
        print err

def run_workers_py(q):
    for i in xrange(2):
        print 'Started worker {}'.format(i)
        worker = rq.Worker([q], connection=q.connection)
        worker.work(burst=True)  # Runs enqueued job


name = 'test_q'
async = True
redis_conn = get_redis_client()
queue = rq.Queue(name, connection=redis_conn, async=async)
queue.empty()
jobs = [queue.enqueue_call(
                func=vislab.tests.testrqaux.foo, args=[i],
                timeout=10000, result_ttl=999) for i in xrange(1)]

t = time.time()
if async:
    run_workers_py(queue)
    # run_workers_cmd(name, 1)

    # Wait until all jobs are completed.
    known_jobs = {}
    while True:
        for job in jobs:
            if job not in known_jobs:
                if job.is_finished:
                    from pprint import pprint
                    print ''
                    pprint(vars(job))
                    known_jobs[job] = 0
                elif job.is_failed:
                    from pprint import pprint
                    print ''
                    pprint(vars(job))
                    known_jobs[job] = 1
        num_failed = sum(known_jobs.values())
        num_succeeded = len(known_jobs) - num_failed
        msg = "\r{:.1f} s passed, {} succeeded / {} failed".format(
            time.time() - t, num_succeeded, num_failed)
        msg += " out of {} total".format(len(jobs))
        sys.stdout.write(msg)
        sys.stdout.flush()
        if num_succeeded + num_failed == len(jobs):
            break
        time.sleep(1)
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('Done with all jobs.')

fq = rq.Queue('failed', connection=redis_conn)
failed_jobs = [j for j in fq.get_jobs() if j.origin == name]
print("{} jobs failed and went into the failed queue.".format(
    len(failed_jobs)))
