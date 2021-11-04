# RuleBERT + bAbi

This file shows how to apply RuleBERT on [bAbi Task #15](https://research.fb.com/downloads/babi/).
## Recipe
1. __Download RuleBERT Model__
```bash
bash download_model.sh
```
2. __Download Data__
The dataset can be downloaded [here](https://research.fb.com/downloads/babi/). We are interested in Task #15. Copy the files from `en_valid/` to `data/external_datasets/bAbI/`.
3. __Fine-tune on bAbI__
```bash
#RuleBERT
!python experiments/external_datasets/bAbI/run_bAbI.py --babi_dir "data/external_datasets/bAbI/" \
                                                       --model_arch 'models/rulebert_161' \
                                                       --epochs 3 \
                                                       --reps 10
#Roberta
!python experiments/external_datasets/bAbI/run_bAbI.py --babi_dir "data/external_datasets/bAbI/" \
                                                       --model_arch 'roberta-large' \
                                                       --epochs 3 \
                                                       --reps 10
```
The results are stored in `--babi_dir`.

## License
[MIT](https://choosealicense.com/licenses/mit/)