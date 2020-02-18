from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Dense
from keras.layers import Flatten


def classifier(train_generator, validation_generator):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(196, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        steps_per_epoch=256,
        validation_steps=8041)
