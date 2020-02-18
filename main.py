from preprocessing import image_processing
from classifier import classifier


def main():
    train_gen, test_gen = image_processing()
    classifier(train_gen, test_gen)


if __name__ == '__main__':
    main()
