get dataset here: https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied



Term was tested to be leaky using XGBoost
```
(.venv) benmajkut@Mac 4705_group_project % python3 term_leakiness.py sba_clean.csv

random 80/20
  Term only           ROC-AUC = 0.945
  other cols, no Term ROC-AUC = 0.746
  other cols + Term   ROC-AUC = 0.958

time (train FY<=2003, test FY2004+)
  Term only           ROC-AUC = 0.946
  other cols, no Term ROC-AUC = 0.633
  other cols + Term   ROC-AUC = 0.945

Default rate by term:
         loans  default_rate
term                        
<=1y     37259         0.440
1-3y     64293         0.547
3-5y    156493         0.318
5-7y    282038         0.111
7-10y   123222         0.114
10-15y   48217         0.083
15-20y   13966         0.221
20y+    150371         0.016
```

naics wasn't a thing until 2001, and at that point it stops being unlabeled, so we'll do FY2001-2010
