# RuleBERT + CheckList 

This file shows how to apply rulebert on [CheckList](https://github.com/marcotcr/checklist).
## Recipe
1. __Download RuleBERT Model__
```
bash download_model.sh
```
2. __Fine-tune on some CheckList Rules__
```
python trainer.py --data-dir data/external_datasets/CheckList
                  -- model_arch models/rulebert_161
                  --epochs 3
                  --verbose
```
3. __Fine-tune on QQP dataset__
    
    The code can be found [here]().
4. __Apply Checklist__

   The code can be found [here](https://github.com/marcotcr/checklist#qqp).
