from sklearn.compose import make_column_selector
from fastapi import FastAPI
import dill
from datetime import datetime
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


app = FastAPI()
@app.get('/status')
def status():
    return "I'm OK"

with open('pipe.pkl', 'rb') as file:
    model = dill.load(file)
@app.get('/version')
def version():
    return model['metadata']

class Form(BaseModel):
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object

class Prediction(BaseModel):
    pred: int

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'pred': y
    }


def main():
    df= pd.read_csv('df.csv', sep=',')
    X = df.drop(columns = ['target'], axis = 1)
    y = df['target']

    cat_features = make_column_selector(dtype_include=[object, int, float])

    cat_transformer = Pipeline(steps = [
        ('encoder', OneHotEncoder(min_frequency = 0.001, sparse_output= False, handle_unknown= 'ignore', dtype = 'int64'))
    ])
    processor = ColumnTransformer(transformers = [
        ('cat', cat_transformer, cat_features)
    ])

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier()
    ]
    best_score = .0
    best_pipe = None
    for m in models:
        pipe = Pipeline(steps = [
            ('pro', processor),
            ('mods', m)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(m).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    print(f'best model: {type(best_pipe.named_steps["mods"]).__name__}, roc_auc: {best_score:.4f}')

    best_pipe.fit(X,y)
    with open('pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'prediction model',
                'author': 'Kolosov D',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["mods"]).__name__,
                'accuracy': best_score
            }
        }, file)

#if __name__ == '__main__':
#   main()