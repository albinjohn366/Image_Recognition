import tensorflow as tf
import sys

# Getting the data set
data_set = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data_set.load_data()
x_train, x_test = x_train / 255, x_test / 255
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[
    2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[
    2], 1)
y_test = tf.keras.utils.to_categorical(y_test)
y_train = tf.keras.utils.to_categorical(y_train)

# Initializing the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
    28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(
    28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Flatten())

# Adding some hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.7))

# Adding output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling and fitting the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
    'accuracy'])
model.fit(x_train, y_train, epochs=10)

# Test the model
model.evaluate(x_test, y_test, verbose=1)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")