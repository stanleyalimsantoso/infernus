#!/bin/bash

#argument 1 is the HTTP port. The other ports are allocated sequentially.
#echo the node name

echo $HOSTNAME
echo "nodename: $SLURMD_NODENAME"


#socket_finder.py finds a set of available consecutive ports.
# port=$(python /fred/oz016/alistair/infernus/infernus/socket_finder.py)

# echo found sockets
# echo $port

port=$1

http_port=$port
grpc_port=$((port+1))
metrics_port=$((port+2))

echo GRPC port is $grpc_port


savedir=${2}
echo "savedir: $savedir"

#get the slurm task ID of this job
echo "task PID: $SLURM_TASK_PID"
echo "step ID: $SLURM_ARRAY_TASK_ID"


#echo $grpc_port > $savedir/${SLURMD_NODENAME}_${SLURM_ARRAY_TASK_ID}.port

#old method
#modeldir=${savedir}/../model_repositories/repo_1

serverdir=${3}

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

apptainer run --nv -B ${savedir}:/models ${serverdir} tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=true --model-control-mode=explicit --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port &
#apptainer run --nv -B ${modeldir}:/models ${serverdir} tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=true --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port &

echo "waiting a bit for the server to start"
sleep 60
echo "done waiting"
#echo $grpc_port > $savedir/${SLURMD_NODENAME}_${SLURM_ARRAY_TASK_ID}.port
echo "done writing port"

wait

#apptainer run --nv -B /fred/oz016/alistair/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/triton_server/containers_22.11/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=true --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port