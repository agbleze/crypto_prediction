#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import (precision_recall_curve, precision_recall_fscore_support,
                             f1_score
                             )

#%%
train_path = "/home/lin/codebase/crypto_prediction/crypto_competition/train.csv"
test_path = "crypto_prediction/crypto_competition/test.csv"
soln_path = "crypto_prediction/crypto_competition/solution_format.csv"

#%%
df = pd.read_csv(train_path)

# %%
df.info()
# %%
df.describe()
# %%
df.columns
# %%


# %%



