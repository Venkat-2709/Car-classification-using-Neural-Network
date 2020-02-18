from keras_preprocessing.image import ImageDataGenerator


def image_processing():
    train_data = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=8,
        zoom_range=0.2,
        horizontal_flip=True)

    test_data = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data.flow_from_directory(
        'car_data/car_data/train',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

    test_generator = test_data.flow_from_directory(
        'car_data/car_data/test',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

    return train_generator, test_generator
