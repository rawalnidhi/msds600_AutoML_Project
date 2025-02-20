import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
   
    df = pd.read_csv(filepath)

    return df


def make_predictions(df):
  
    model = load_model('gbc')
    predictions = predict_model(model, data=df)
    predictions.rename({'prediction_label': 'churn_prediction'}, axis=1, inplace=True)
    predictions['churn_prediction'].replace({1: 'Churn', 0: 'No churn'},
                                            inplace=True)
    return predictions['churn_prediction']

if __name__ == "__main__":
    df = load_data('/Users/nidhi/Documents/Intro to DS/classes/Week5/new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
