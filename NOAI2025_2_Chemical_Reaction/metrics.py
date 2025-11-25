import numpy as np
import pandas as pd
import json

sub_file = pd.read_csv("submission_val.csv")
y_pred = sub_file.iloc[:, 1].values
ans_file = pd.read_csv("data/data_val/val_t12_target.dat")
y_true = ans_file.iloc[:, 1].values

exp_score_val = np.maximum(0, 1 - np.log(1+0.1*abs(y_pred-y_true))/5)
val_score = np.mean(exp_score_val)
exp_score_val = exp_score_val.tolist()
val_score = val_score.item()

sub_file = pd.read_csv("submission_test.csv")
y_pred = sub_file.iloc[:, 1].values
ans_file = pd.read_csv("data/data_test/test_t12_target.dat")
y_true = ans_file.iloc[:, 1].values

exp_score_test = np.maximum(0, 1 - np.log(1+0.1*abs(y_pred-y_true))/5)
test_score = np.mean(exp_score_test)
exp_score_test = exp_score_test.tolist()
test_score = test_score.item()

score = {
    "public_a": val_score,
    "public_detail": {
        "individual_score": exp_score_val,
    },
    "private_b": test_score,
    "private_detail":{
        "individual_score": exp_score_test,
    },
}
#print(score)
ret_json = {
    "status": True,
    "score": score,
    "msg": "Success!",
}
with open('score.json', 'w') as f:
    f.write(json.dumps(ret_json))  