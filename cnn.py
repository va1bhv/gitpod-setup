# Importing the libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow
# gpus = tensorflow.config.experimental.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(gpus[0], True)

"""# **Data Preprocessing**"""

# Train-Test Split
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)

# Reshape the data
# Our data has shape (6000, 28, 28) i.e. 6000 rows and 28 X 28 image
# Our model needs another dimension, so we reshape the data to (6000, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Encode categorical data and one-hot-encode them
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Correct the datatype of input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Scaling the data
X_train /= 255
X_test /= 255

# Shape of data now
print(f"X_train's shape: {X_train.shape}.")
print(f"{X_train.shape[0]} training samples.")
print(f"{X_test.shape[0]} testing samples.")

"""# **Creating the model**"""

# Setting the parameters
batch_size = 10
num_classes = 10
epochs = 10
input_shape = (28, 28, 1)

# Initializing the model
classifier = Sequential()

# Adding the layers
# 1. Convolutional layer with a kernel size of (3*3) and 32 filters with ReLU activation.
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      activation='relu', input_shape=input_shape))

# 2. Convolutional layer with a kernel size of (3*3) and 64 filters with ReLU activation
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 3. MaxPooling layer with a pool size of (2*2) and stride of 2 in both directions.
classifier.add(MaxPooling2D(pool_size=(2*2)))

# 4. DropOut of 25% (Randomly sets 25% of inputs to 0. Used to reduce degree of OverFitting)
classifier.add(Dropout(0.25))

# 5. Flatten the image into a stream of data.
classifier.add(Flatten())

# 6. Add a hidden layer with 256 neurons (Fully Connected Hidden Layer).
classifier.add(Dense(units=256, activation='relu'))

# 7. Another DropOut for this Hidden Layer: 25%
classifier.add(Dropout(0.25))

# 8. Output Layer with 10 neurons and Softmax activation function (With softmax
#     all values will lie in [0, 1] and summation of all values is 1. This makes
#     it a probabilistic model.)
classifier.add(Dense(num_classes, activation='softmax'))

# Compiling the Neural Network with loss function as Categorical CrossEntropy, optimizer as adadelta and choosing the accuracy as the metric.
classifier.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

"""# **Training the model**"""

# Run the fit method on the Classifier Object and store the training history in the training_history object.
training_history = classifier.fit(
    X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
print('The model has been trained successfully.')

# Saving the trained Classifier.
# classifier.save('mnist.h5')
# print('Saving the model as mnist.h5')

"""# **Evaluating the Classifier**"""

score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')
