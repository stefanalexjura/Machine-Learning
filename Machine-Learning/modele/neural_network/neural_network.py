import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#am luat un model de imagini din biblioteca, unul dintre cele mai folosite si am facut matricea de pixeli
train_images, test_images = train_images / 255.0, test_images / 255.0

"""class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']"""

"""plt.figure(figsize=(10,10))
for i in range (0,10):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()"""

model = models.Sequential()
#fac o convolutie, iau 32 de layere/straturi/neuroni pe care le inmultesc cu o matrice de 3x3 pt convolutie cu functia de activare ReLu si la in
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

model.save('model_salvat.h5')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_loss)

img_path = "poza.jpg"
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0 
predictions = model.predict(img_array)
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
probabilities = probability_model.predict(img_array)

print(probabilities)



