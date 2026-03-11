import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dataset.csv")

X = data[['c','phi','gamma','H','slope']]
y = data['fos']

model = RandomForestRegressor()

model.fit(X,y)

def predict_fos(c,phi,gamma,H,slope):

    return model.predict([[c,phi,gamma,H,slope]])[0]