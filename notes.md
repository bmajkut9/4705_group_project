get dataset here: https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied



# Cleaning the data 

Assesing immediate leaks - anything that let you know info about the loan after it was given
- chgoffdate - leaks if it was defaulted
- mis_status - leaks final status of loan
- chgoffpringr - leaks missed payments
- disbursment date - leaks info that you'd have after approval time
- disbursment gross - leaks info you'd have after approval time
- balance gross - leaks amount outstanding
- ~~approvaldate - if approved it was deemed good~~
- ~~approvalfy - if approved it was deemed good~~
- ~~grappv - leaks that bank approved it~~
- ~~sba_appv - leaks SBA approval~~

Basic cleaning before further leak assesment 
- default: (CHGOFF -> 1, P I F -> 0, drop rows with no outcome)
- drop dupicates using row ID
- drop immediate leaks
- take first 4 digits (year) of ApprovalFY
- ApprovalDate: convert date strings to DateTime datatype 
- Taking everything from 2001 - 2014, cutoff is for valid NAICS vals and removing ~25% of data spread over 40 years vs. keeping ~75% over 13 years
- Drop leaky cols 

Running cleaning script 

((.venv-icl) ) benmajkut@Mac 4705_group_project % python3 code/clean_sba.py archive/SBAnational.csv ./sba_clean.csv
raw                                             899,164 rows
drop missing outcome                            897,167 rows
drop duplicate loan IDs                         897,167 rows
drop unparseable date / amount / term           897,167 rows
drop approvals after FY2010                     875,859 rows
drop approvals before FY2001                    507,571 rows

default rate: 25.52%
blank cells per column:
Bank             12
BankState        12
NAICS            68
NewExist        557
UrbanRural     6574
RevLineCr     13834
LowDoc         3565

Cols were testing as leaky using XGBoost Reciever Operating Characteristic - Area Under the Curve vals
If a majority of the combined rating is mostly made from the single col, then it is probably leaky

XGBoost config uses 200 trees with 4 question limit per tree, 

`python3 code/leakiness_testing.py sba_clean.csv`
random 80/20
  Term only           ROC-AUC = 0.953
  other cols, no Term ROC-AUC = 0.687
  other cols + Term   ROC-AUC = 0.963

time (train FY<=2003, test FY2004+)
  Term only           ROC-AUC = 0.942
  other cols, no Term ROC-AUC = 0.631
  other cols + Term   ROC-AUC = 0.945

Default rate by term:
         loans  default_rate
term                        
<=1y     23330         0.589
1-3y     50294         0.605
3-5y    100020         0.444
5-7y    199958         0.132
7-10y    60362         0.169
10-15y   13636         0.117
15-20y    4646         0.313
20y+     55325         0.022

Single-column ROC-AUC (0.5 = no signal, Term = 0.945):
               random   time
column                      
Term            0.953  0.942
Bank            0.704  0.645
SBA_Appv        0.659  0.620
GrAppv          0.640  0.611
LoanNr_ChkDgt   0.678  0.590
sba_share       0.587  0.583
BankState       0.632  0.577
NAICS           0.612  0.575
RetainedJob     0.593  0.567
NoEmp           0.560  0.552
CreateJob       0.548  0.542
Zip             0.604  0.540
UrbanRural      0.527  0.535
State           0.583  0.535
City            0.597  0.532
RevLineCr       0.543  0.530
FranchiseCode   0.611  0.505
NewExist        0.503  0.495
LowDoc          0.520  0.489

Term is 100% leaky and is mostly explained by shorter loans defaulting more. Nothing else appears to be even close to the 0.9 val that term has. So that will be removed and the rest kept in feature engineering
Maybe we could have referenced term against NAICS sectors? 
There is a weird point where loans from 2010-2014 with 15-20 year term are labeled PIF, which seems impossible to know, but it's for a very small minority of rows



# Feature Engineering

Now that data is cleaned we want it to be 'model ready' 

features.py - formats data into model features (bank specific features are added later in split_data.py)

Log transform - these are right skewed cols, so this is a good way to linearize them for logistic regression
- log_loan_amount
- log_employees
- log_jobs_created
- log_jobs_retained
Bucketing
- GrAppv in 5 bins at 50k, 150k, 350k, 1M+
Binary conversion 
- is_new_business
- is_urban
- is_revolving_line
- is_low_doc
Grouping
- NAICS to 2-digit sector (31-33, 44-45, 48-49 merged): these are the only sector groups in NAICS, everything else is seperate
Unchanged
- State
Added
- is_franchise (FranchiseCode > 1) 
- same_state_bank (State == BankState)
- sba_share = (SBA_Appv / GrAppv) added, represents gauranteed share of the loan

Blanks are left as is which is not a problem for the models
Scaling and one-hot encoding happen only inside the logistic regression pipeline. XGBoost and the tabular models use categories directly and don't need scaling



# Split + Bank Features

split_data.py - splits by ApprovalFY, then adds bank features computed from TRAIN only

Split
- train FY2001-2004: the model learns from these
- val FY2005: tuning (settings, features, thresholds)
- test FY2006-2010: final score, only run once at the end

Bank features (need other loans' outcomes, so they can't be built before the split)
- log_bank_loans = log(1 + number of train loans the bank made)
- bank_default_rate = bank's train default rate, shrunk toward the overall rate so small banks aren't extreme
- Train rows use out-of-fold values (5 folds) so a loan's own outcome never feeds its own feature
- Banks not seen in train are blank

Outputs: train.csv, val.csv, test.csv



# Baselines

baselines.py - logistic regression + XGBoost on the time-based split

- Logistic regression: median fill + blank indicator, StandardScaler, one-hot for categories
- XGBoost: native categories, early stopping on val
- Outputs: baseline_metrics.csv, test_predictions.csv



