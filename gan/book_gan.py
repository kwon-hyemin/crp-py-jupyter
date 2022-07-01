if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import matplotlib.pyplot as plt

    dataset, info = tfds.load("fashion_mnist", split="train", with_info=True)


    def convert(row):
        image = tf.image.convert_image_dtype(row["image"], tf.float32)
        label = tf.cast(row["label"], tf.float32)
        return image, label


    batch_size = 32
    dataset = dataset.map(convert).batch(batch_size).prefetch(1)