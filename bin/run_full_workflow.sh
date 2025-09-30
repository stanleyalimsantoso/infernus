#!/bin/bash

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0
set -e

dataset_file=$1

training_args=$(jq -r '.training_args' $dataset_file)
validation_args=$(jq -r '.validation_args' $dataset_file)
testing_args=$(jq -r '.testing_args' $dataset_file)
model_args=$(jq -r '.model_args' $dataset_file)
jobdir=$(jq -r '.jobdir' $model_args)

echo "Training args: $training_args"
echo "Validation args: $validation_args"
echo "Testing args: $testing_args"
echo "Model args: $model_args"
#create the training dataset


###################################################################
#------Generating training, validation, and testing datasets------#
###################################################################

#check if the numpy files already exist
train_dir=$(jq -r '.project_dir' $training_args)
val_dir=$(jq -r '.project_dir' $validation_args)
test_dir=$(jq -r '.project_dir' $testing_args)
#check if SNR_abs.npy exists in each directory

if [ -f "$train_dir/SNR_abs.npy" ]; then
	echo "Training dataset already exists, skipping generation"
else
	echo "Training dataset does not exist, generating"
	training=$(bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh $training_args) 
	echo "SUBMITTED TRAINING JOB"
	echo "Output: $training"
	training_id=$(echo $training | awk '{print $NF}')
	echo
	echo "Training dataset job ID: $training_id"
	echo
	echo
fi

if [ -f "$val_dir/SNR_abs.npy" ]; then
	echo "Validation dataset already exists, skipping generation"
else
	echo "Validation dataset does not exist, generating"
	validation=$(bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh $validation_args)
	echo "SUBMITTED VALIDATION JOB"
	echo "Output: $validation"
	validation_id=$(echo $validation | awk '{print $NF}')
	echo
	echo "Validation dataset job ID: $validation_id"
	echo
	echo
fi

if [ -f "$test_dir/SNR_abs.npy" ]; then
	echo "Testing dataset already exists, skipping generation"
else
	echo "Testing dataset does not exist, generating"
	testing=$(bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh $testing_args)
	echo "SUBMITTED TESTING JOB"
	echo "Output: $testing"

	testing_id=$(echo $testing | awk '{print $NF}')
	echo
	echo "Testing dataset job ID: $testing_id"
fi

sleep 5


##########################################################################
#--------------Running tuning (if needed) and then training--------------#
##########################################################################

savedir=$(jq -r '.save_dir' $model_args)
mkdir -p $savedir
echo "Save directory: $savedir"


num_models=$(jq -r '.metamodel_jobs' $model_args)

echo "Number of metamodel jobs: $num_models"

jobname=$(jq -r '.jobname' $model_args)

tune_name=${jobname}_tune
n_tasks=$(jq -r '.n_workers' $model_args)
echo "--dependency=afterok:${training_id}:${validation_id}:${testing_id}"

if [ -z "$training_id" ]; then
	echo "No training job ID found"
	dep=""
else
	echo "Training job ID found: $training_id"
	dep=--dependency=afterok:${training_id}:${validation_id}:${testing_id}
fi

# tuning=$(sbatch --job-name=${tune_name} --output=${jobdir}/logs/%x.log --ntasks=$n_tasks ${dep} --parsable /fred/oz016/alistair/NN_training/tune_distributed_better.sh $model_args)

echo "Saving logs to" $jobdir

#if we're doing tuning OR sample generation
if [ -n "$tuning" ] || [ -n "$training_id" ]; then
	dep="--dependency=afterok:"
	if [ -n "$tuning" ]; then
		echo "Tuning job ID found: $tuning"
		dep+="${tuning}"
	fi
	if [ -n "$training_id" ]; then
		echo "Training job ID found: $training_id"
		dep+="${training_id}:${validation_id}:${testing_id}"
	fi
else
	echo "No tuning or sample generaton job ID found, proceeding to training without dependencies"
	dep=""
fi


#alternatively, hardcode a dependency here
#dep="--dependency=afterok:"

#tuning, no samplegen
#no tuning, samplegen
echo "dependency: $dep"

#TODO: generalise to not be reliant on my code
#0-$((num_models - 1))
metamodel_name=${jobname}_metamodel
metamodel=$(sbatch --job-name=${metamodel_name} --array=0-$((num_models - 1)) ${dep} --output=${jobdir}/logs/%x_%a.log  --parsable /fred/oz016/alistair/NN_training/train_metamodel_args.sh $model_args)

#use jq to set the field "train_dir" in the metamodel file to the directory "project_dir" in $training_args

#use jq to set the field "val_dir" in the metamodel file to the directory "project_dir" in $validation_args
echo "TODO: set the train_dir and val_dir fields in the metamodel file automatically"
#jq '.train_dir = $dir' --arg dir $training_args.project_dir $model_file > $model_file 

#--dependency=afterok:${metamodel}
prep_repos=$(sbatch --job-name=${jobname}_prep_repos --dependency=afterok:${metamodel} --output=${jobdir}/logs/%x.log --parsable /fred/oz016/alistair/NN_training/prep_model_repos.sh $model_args)
#TODO: generalise to not be reliant on my code
python /fred/oz016/alistair/NN_training/make_bg_inj_files.py --jsonfile=${model_args}


BG=$(bash ${INFERNUS_DIR}/bin/filter_and_predict.sh $savedir/BG.json ${prep_repos})
inj=$(bash ${INFERNUS_DIR}/bin/filter_and_predict.sh $savedir/inj.json ${prep_repos})

BG_id=$(echo $BG | awk '{print $NF}')
inj_id=$(echo $inj | awk '{print $NF}')

echo "BG job ID: $BG_id"
echo "Injection job ID: $inj_id"


#real event job
ret=$(python3 ${INFERNUS_DIR}/dev/real_events_prep.py --configfile=${model_args})
array=$(echo $ret | awk '{print $NF}')

echo "array: $array"

tasks=$(jq -r '.ntasks' $savedir/inj.json)
mem=$(jq -r '.mem' $savedir/inj.json)
num_triggers=$(jq -r '.num_triggers' $savedir/inj.json)
temp_size=$(( ${num_triggers} * 3 ))
echo "real event cpus: $tasks"
real_events=$(sbatch -J ${jobname}_real --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --dependency=afterok:${prep_repos} \
			--tmp=${temp_size}GB --output=${jobdir}/logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $savedir/real_events.json)
echo "Real event job ID: $real_events"

echo "Trying plotting code"

logdir=$(jq -r '.jobdir' $model_args)/plotting.log
echo "Log directory: $jobdir"

#${BG_id}:${inj_id}:${real_events}
plotting=$(sbatch --job-name=${jobname}_plotting --output=${jobdir}/logs/${jobname}_plotting.log --time=02:00:00 --mem=30G --dependency=afterok:${BG_id}:${inj_id}:${real_events} \
	--parsable --wrap "python ${GWSAMPLEGEN_DIR}/share/plot_model_test_performance.py --configfile=${model_args}")