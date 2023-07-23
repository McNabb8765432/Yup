import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

# Загрузка данных Fashion-MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Предварительная обработка данных
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Определение архитектуры модели
model = keras.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="elu"),
        layers.Dense(128, activation="elu"),
        layers.Dense(10),
    ]
)

# Компиляция модели
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Оценка производительности модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
