#!/bin/bash

#infernus_dir='/fred/oz016/alistair/infernus'


source ~/.bashrc
#infernus_dir=${INFERNUS_DIR}

jsonfile=$1

infernus_dir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['infernus_dir'])")
echo $infernus_dir
#get directory from json file

savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

#get tensorflow model file path from json file

#tf_model=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['tf_model'])")

#echo $tf_model
#get batch file from json file

batch_size=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['batch_size'])")

#metamodel_dir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['metamodel_dir'])")
jobdir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['jobdir'])")

n_models=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['final_model_candidates'])")

mkdir -p $savedir/model_repositories/repo_1
#mkdir -p $savedir/model_repositories/repo_2

cp $jsonfile $savedir

cd $savedir/model_repositories

new_tf_model='tf_test'

#new path should be cwd + new_tf_model
new_tf_model=$(pwd)/$new_tf_model
echo $new_tf_model

for i in $(seq 0 $(($n_models-1)));
do

	echo "loading and splitting tensorflow model"
	#python ${infernus_dir}/infernus/serving/convert_model/save_tf_model_2.py $new_tf_model $jobdir $metamodel_dir $i
	python ${infernus_dir}/infernus/serving/convert_model/save_tf_model_2.py $new_tf_model $savedir $i
	echo "finished saving tensorflow models"
	#convert models to onnx
	python -m tf2onnx.convert --saved-model ${new_tf_model}_full --output temp_full.onnx

	mkdir -p repo_1/model_full_${i}/1

	#modify onnx models
	#sleep 5

	python ${infernus_dir}/infernus/serving/convert_model/onnx_modify.py temp_full.onnx $batch_size repo_1/model_full_${i} ${i}

	rm temp_*.onnx
	rm -r tf_test_*

done
echo "done"

