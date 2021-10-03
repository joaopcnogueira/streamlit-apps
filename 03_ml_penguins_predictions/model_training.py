import joblib
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv('datasets/penguins_cleaned.csv')

num_vars = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_vars = ['island', 'sex']
features = cat_vars + num_vars
target   = 'species'

X = df.filter(features)
y = df[target]

model = Pipeline(steps=[
        ('OHE', OneHotEncoder(variables=cat_vars, drop_last=True)),
        ('ALGO', RandomForestClassifier(random_state=30))
    ]
)

model.fit(X, y)

model_artefacts = {
    'num_vars': num_vars, 
    'cat_vars': cat_vars, 
    'features': features,
    'target': target, 
    'model': model
}

joblib.dump(model_artefacts, 'models/model.pkl')
