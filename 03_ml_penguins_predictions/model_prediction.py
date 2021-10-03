import joblib
import pandas as pd

df = pd.read_csv('datasets/penguins_example.csv')

model_artefacts = joblib.load('models/model.pkl')
features = model_artefacts['features']
model = model_artefacts['model']

X = df.filter(features)

prediction_label = model.predict(X)

df.assign(prediction = prediction_label).to_csv('datasets/penguins_example_with_prediction.csv', index=False)
