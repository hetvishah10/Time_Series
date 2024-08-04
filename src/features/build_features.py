import pandas as pd

def create_target_variable(data):
    data['Next_day'] = data['Close'].shift(-1)
    data['Target'] = (data['Next_day'] > data['Close']).astype(int)
    return data
