# RuleBERT + Negated LAMA 
This file shows how to apply rulebert on the [Negated LAMA Dataset](https://github.com/facebookresearch/LAMA).
## Recipe
1. __Download RuleBERT Model__

```bash
bash download_model.sh
```

2. __Replace CLS head with MLM head__
```python
from transformers import RobertaForMaskedLM, AutoModelForSequenceClassification,RobertaTokenizer

# Load Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

#Load RuleBERT
rulebert_cls = AutoModelForSequenceClassification.from_pretrained("models/rulebert_161")

#Load Roberta_MLM
roberta_mlm = RobertaForMaskedLM.from_pretrained('roberta-large')

# Transfer encoder
roberta_mlm.roberta = rulebert_cls.roberta

roberta_mlm.save_pretrained('models/rulebert_161_mlm')
tokenizer.save_pretrained('models/rulebert_161_mlm')

```
3. __Run Negated LAMA__

   The code can be found [here](https://github.com/facebookresearch/LAMA). At of the time of this paper, there was no hugging-face connector for the roberta model. We provide ```rulebert_connector.py``` as the file to be used for integration. The following table shows the results for the HuggingFace (HF) Roberta connector and FairSeq Roberta Connector. 

| Dataset | HF Roberta (ρ)| Fairseq Roberta (ρ)| HF Roberta (%) | Fairseq Roberta (%)|
| --- | :-----------: |  :-----------: |  :-----------: |  :-----------: |
| `Google-RE birth-place` |  90.99  |90.99   | 18.51 |18.51|
| `Google-RE birth-date `|  82.87  |82.87   | 1.4 |1.4|
| `Google-RE death-place` |  86.44  |86.44   | 0.31 |0.31|
| `T-REx 1-1` |  78.95  |78.95   | 61.38 |61.38|
| `T-REx N-1` |  87.56  |87.56   | 43.8 |43.8|
| `T-REx N-M` |  89.38  |89.38   | 50.78 |50.78|
| `ConceptNet` |  49.18  |42.61   | 12.93 |9|
| `SQuAD` |  90.88  |89.71   | 45.26 |44.76|


## License
[MIT](https://choosealicense.com/licenses/mit/)