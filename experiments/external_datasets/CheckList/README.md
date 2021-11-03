# RuleBERT + CheckList 

This file shows how to apply RuleBERT on [CheckList](https://github.com/marcotcr/checklist).
## Recipe
1. __Download RuleBERT Model__
```bash
bash download_model.sh
```
2. __Fine-tune on some CheckList Rules__
```bash
python trainer.py --data-dir data/external_datasets/CheckList \
                  -- model_arch models/rulebert_161 \
                  --epochs 3 \
                  --verbose \
                  --hard_rule
```
3. __Fine-tune on QQP dataset__
    
    We use a HuggingFace notebook for fine-tuning on QQP. The notebook can be found [here](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb).

    Make sure to load the RuleBERT encoder with an untrained CLS layer:

    ```python
    from transformers import RobertaForSequenceClassification

    #Load RuleBERT
    rulebert_cls = RobertaForSequenceClassification.from_pretrained("models/rulebert_161")

    #Load Roberta_MLM
    roberta_cls = RobertaForSequenceClassification.from_pretrained('roberta-large')

    # Transfer encoder
    roberta_cls.roberta = rulebert_cls.roberta
    ```
4. __Apply Checklist__

   The repository can be found [here](https://github.com/marcotcr/checklist#qqp).


## License
[MIT](https://choosealicense.com/licenses/mit/)