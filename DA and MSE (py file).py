import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')



def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    #Converting data into dataframe
    df = pd.DataFrame(data)
    df = df[['Close']]
    
    #Cubic Interpolation for NAN values(we compared all interpolations on the stock index data and 
    #concluded that cubic interpolation fits the best)
    df = df.interpolate(method="cubic")
    
    #Loading our trained model
    model = tf.keras.models.load_model('Model.h5')
    
    #Taking closing price values for prediction of 1st day
    ds = df.values
    
    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    
    #Defining test values for 1st day prediction
    X_test1 = []
    X_test1.append(ds_scaled)
    X_test1 = np.array(X_test1)
    
    #Reshaping data to fit into LSTM model
    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1],1))
    
    #Predicting for 1st day using our model
    pred_price1=model.predict(X_test1)
    #Inverse transform to get actual value
    pred_price1 = normalizer.inverse_transform(pred_price1)
    
    #Taking closing price values for prediction of 2nd day
    ds=np.array(ds)
    i=0
    while (i<49):
        ds[i]=ds[i+1]
        i=i+1
    ds[49]=pred_price1[0][0]
    #Normalizing data between 0 & 1
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    
    #Defining test values for 2nd day prediction
    X_test2 = []
    X_test2.append(ds_scaled)
    X_test2 = np.array(X_test2)
    
    #Reshaping data to fit into LSTM model
    X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1],1))
    
    #Predicting for 2nd day using our model
    pred_price2=model.predict(X_test2)
    #Inverse transform to get actual value
    pred_price2 = normalizer.inverse_transform(pred_price2)
    
    #Making the final list with prediction of next two days
    pred_prices_list = []
    pred_prices_list.append(pred_price1[0][0])
    pred_prices_list.append(pred_price2[0][0])
    
    return pred_prices_list


if __name__== "__main__":
    evaluate()
    
