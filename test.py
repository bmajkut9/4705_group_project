import pandas as pd
t = pd.read_csv("test.csv")
print(t.groupby("ApprovalFY")["default"].agg(loans="size", default_rate="mean").round(3))