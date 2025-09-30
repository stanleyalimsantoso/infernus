#SLURM-specific code to find the memory usage of a job. This can be used to avoid exceeding memory limits.

# find slurm job memory via job cgroup

# cgroups v1
pre='/sys/fs/cgroup/memory'
names=( ( 'usage', 'memory.memsw.usage_in_bytes' ),
        ( 'max_usage', 'memory.memsw.max_usage_in_bytes' ),
        ( 'limit', 'memory.memsw.limit_in_bytes') )

class jobmem:
    def __init__(self):
        self.jobdir = None
        self.findJobDir()

    def findJobDir(self):
        p=''
        with open('/proc/self/cgroup', 'r') as f:
            for l in f:
                if l.split(':')[1] == 'memory':
                    p = l.split(':')[2]

        # slurm 23.11 uses /slurm/uid_X/job_Y/... 
        # remove elements after job_Y/
        s = p.split('/')[1:4]
        # check
        if len(s) != 3 or 'job' not in s[2]:
            return

        self.jobdir = pre + '/' + s[0] + '/' + s[1] + '/' + s[2]

    def read(self):
        if self.jobdir == None:
            return None
        # make a dict of mem usage
        o = {}
        for n,f in names:
            with open( self.jobdir + '/' + f) as m:
                for l in m:
                    o[n] = int(l)
        return o