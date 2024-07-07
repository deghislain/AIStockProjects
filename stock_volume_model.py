import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

stock_data = pd.read_csv('csv_weekly_data.csv')
print(stock_data.head())
print(stock_data.shape)
print(stock_data.describe())

stock_data.isnull().sum()

stock_data_columns = stock_data.columns

predictors = stock_data['week_number']
target = stock_data['volume']

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3)

print("predictors*******************************************************")
print("X_train_head",X_train.head())
print("y_train_head", y_train.head())

print("Normalized predictors*******************************************************")
X_train_norm = (X_train - X_train.mean()) / X_train.std()
print("X_train_norm ",X_train_norm.head())


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
    RNN_volume_model = keras.models.load_model('stock_RNN_volume_model_old.h5')
except FileNotFoundError as err:
    isExistingTrainingModel = False
    print("FileNotFoundError:", err)

if isExistingTrainingModel == False:
    RNN_volume_model = create_RNN_volume_model(hidden_units=100, dense_units=1, input_shape=(2, 1),
                                               activation=['tanh', 'tanh'])
    RNN_volume_model.fit(X_train_norm, y_train, validation_split=0.3, epochs=400, verbose=2)
    RNN_volume_model.save("stock_RNN_volume_model.h5")
    trainScore = RNN_volume_model.evaluate(X_train_norm, y_train, verbose=0)
    print("trainScore ", trainScore)


RNN_volume_model.summary()

prediction_df = pd.read_csv('csv_evaluation_data.csv')

prediction_data = prediction_df['week_number']

print(prediction_df.head())

predicted_volume = RNN_volume_model.predict(prediction_data)

print("*******************************", predicted_volume)
def plot_predictions(predicttors, predicted_volume):
    plt.title("Stock volume prediction")
    plt.xlabel("Week")
    plt.ylabel("Stock Volume")
    plt.plot(predicttors, predicted_volume)
    plt.show()


plot_predictions(prediction_data, predicted_volume)
