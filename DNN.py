
# A very simple deep neural netword to recognize digits using tensorflow and keras


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist=tf.keras.datasets.mnist # 28*28 images of hand-written digits(0-9)

(x_train,y_train),(x_test,y_test)=mnist.load_data() #unpacking the images data into train and test

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


#we will create the model now
model=tf.keras.models.Sequential() #Secuential model is normal feed forwarding model
#Adding layer using model.add()
model.add(tf.keras.layers.Flatten()) #first layer or the input layer , we are not taking 28*28 input
#instead we are flatenning it 
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #second layer which is hidden layer with 128 nodes and activation function is rectified linear
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #3rd layer
# model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #3rd layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) #output layer , 10 because we have to classify 10 digit and softmax for probability distribution

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) #'adam' optomizer and loss function = sparse_categorical_crossentropy
model.fit(x_train,y_train,epochs=3)

val_loss,val_acr=model.evaluate(x_test,y_test)
print(val_loss,val_acr)

#testing the prediction
prediction=model.predict([x_test])
print(np.argmax(prediction[1]))
plt.imshow(x_test[1],cmap=plt.cm.binary)
plt.show()



# plt.imshow(x_train[0]) 
# print(x_train[0])
# plt.imshow(x_train[0],cmap=plt.cm.binary) #cmap=plt.cm.binary will color map the image into binary 
# plt.show()


