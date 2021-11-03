# RuleBERT + bAbi (Under Construction)

This file shows how to apply RuleBERT on [bAbi Task #15](https://research.fb.com/downloads/babi/).
## Recipe
1. __Download RuleBERT Model__
```bash
bash download_model.sh
```
2. __Fine-tune on bAbI__
```bash
python trainer.py --data-dir data/external_datasets/bAbI \
                  -- model_arch models/rulebert_161 \
                  --epochs 3 \
                  --verbose \
```
3. __Fine-tune on QQP dataset__


## License
[MIT](https://choosealicense.com/licenses/mit/)