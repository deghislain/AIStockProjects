import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt



stock_data = pd.read_csv('csv_weekly_data.csv')
print(stock_data.head())
print(stock_data.shape)
print(stock_data.describe())

stock_data.isnull().sum()

stock_data_columns = stock_data.columns

predictors = stock_data['week_number']
target = stock_data['volume']

print("predictors*******************************************************")
print(predictors.head())
print(target.head())

print("Normalized predictors*******************************************************")
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors_norm.head())


# Create a traditional RNN network
def create_RNN_volume_model(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mse', optimizer='adam')
    return model


RNN_volume_model = ""
isExistingTrainingModel = True
try:
    RNN_volume_model = keras.models.load_model('stock_RNN_volume_model.h5')
except FileNotFoundError as err:
    isExistingTrainingModel = False
    print("FileNotFoundError:", err)

if isExistingTrainingModel == False:
    RNN_volume_model = create_RNN_volume_model(hidden_units=2, dense_units=1, input_shape=(3, 1),
                                               activation=['tanh', 'tanh'])
    RNN_volume_model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
    RNN_volume_model.save("stock_RNN_volume_model.h5")

RNN_volume_model.summary()

prediction_df = pd.read_csv('csv_evaluation_data.csv')

prediction_data = prediction_df['week_number']

print(prediction_df.head())

predicted_volume = RNN_volume_model.predict(prediction_data)

print("******************************", predicted_volume)
def plot_predictions(predicttors, predicted_volume):
    plt.title("Stock volume prediction")
    plt.xlabel("Week")
    plt.ylabel("Stock Volume")
    plt.plot(predicttors, predicted_volume)
    plt.show()


plot_predictions(prediction_data, predicted_volume)
