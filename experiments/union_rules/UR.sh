echo "Training on Union of Rules in $1."
path="$1"
length=${#path}
last_char=${path:length-1:1}
[[ $last_char == "/" ]] && path="${path:0:length-1}";

python trainer.py --data_dir "$path"/ \
                --model_arch roberta-large \
                --epochs 3 \
                --verbose

$model_name = $(ls models/| tail -1)

python tester.py --test_data_dir "$path"/ \
                --model_dir models/"$model_name" \
                --verbose

echo "Finished training and testing.\n"