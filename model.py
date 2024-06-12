import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score as R2, max_error as ME

path = 'testdata'
data = pd.read_csv(f'{path}/data.csv')
x = data.loc[:, 'density':'CN_sum']
y = data.loc[:, 'Formation_energy']
xtrain, xtest, ytrain, ytest = TTS(x, y, test_size=0.3, random_state=420)

# Train model
model = XGBR(n_estimators=200)
model.fit(xtrain, ytrain)
ytrain_preds = model.predict(xtrain)
ytest_preds = model.predict(xtest)

# Test model
r2=R2(ytest, ytest_preds)
mse=MSE(ytest, ytest_preds)
mae=MAE(ytest, ytest_preds)
me=ME(ytest, ytest_preds)
print(f'R2: {r2:.4f}')
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'ME: {me:.4f}')

# Export model
pickle.dump(model, open(f'{path}/TestModel.dat','wb'))


