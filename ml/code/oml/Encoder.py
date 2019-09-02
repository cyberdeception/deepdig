import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers


def autoencode(x_train, x_test):
    input_dim = x_train.shape[1]
    encoding_dim = 200

    compression_factor = float(input_dim) / encoding_dim
    print("Compression factor: %s" % compression_factor)

    autoencoder = Sequential()
    autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
    )
    autoencoder.add(
    Dense(input_dim, activation='sigmoid')
    )

    autoencoder.summary()

    input_img = Input(shape=(input_dim,))
    encoder_layer = autoencoder.layers[0]
    encoder = Model(input_img, encoder_layer(input_img))

    encoder.summary()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    return encoded_train, encoded_test

        

