# The following script assumes that you prepare the output.json and test_eval.json under ./challenge
# make sure you are under ./challenge
folder_name=$1
python evaluation.py \
--root_path1 ./output_$folder_name.json \
--root_path2 ./test_eval.json
