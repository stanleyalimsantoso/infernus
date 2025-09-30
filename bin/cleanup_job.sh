#! /bin/bash
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

jsonfile=$1
#savedir=$2
gpu_job=$2
savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

ml python/3.10.4
echo $VIRTUAL_ENV
source $VIRTUAL_ENV/bin/activate
#source /fred/oz016/alistair/nt_310/bin/activate

python ${INFERNUS_DIR}/bin/cleanup.py --jsonfile=$jsonfile
pid=$!

wait $pid
status=$?
if [ $status -ne 0 ]; then
	echo "Cleanup script failed with status $status"
	echo "Retaining log files for debug purposes"

else
	echo "Cleanup script completed successfully"
	echo "deleting log files"
	#get this job's name
	jobname=${SLURM_JOB_NAME}
	#trim "_cleanup" from the end of the jobname
	jobname=${jobname%_cleanup}
	#echo "To remove: ${jobname}_*[1-9]*.log"
	#rm -f ${INFERNUS_DIR}/triton_logs/${jobname}_*[1-9]*.log
	echo "To remove: $savedir/../${jobname}_*[1-9]*.log"
	rm -f $savedir/../logs/${jobname}_*[1-9]*.log
fi
echo "Cleanup done"

if [ -z "$gpu_job" ]; then
	echo "No GPU job to cancel"
	exit 0
else
	echo "Cancelling GPU job $gpu_job"
	scancel $gpu_job
fi
#remove any .port files from the savedir. TODO: ensure this works with the old job format.
savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")
rm -vf $savedir/*.port
