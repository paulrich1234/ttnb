import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt






data=pd.read_table('combine_local_net.txt',sep='\t',encoding='utf-8',engine='python')
X=data.drop(['Unnamed: 0','label'],axis=1)
y=data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)







# print(X_train)

model = models.Sequential()
model.add(layers.Dense(28, activation='sigmoid', input_shape=(14,)))
model.add(layers.Dense(14, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))





model.compile(optimizer=optimizers.RMSprop(lr=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(np.array(X_train),
                    y_train,
                    epochs=300,
                    batch_size=20,
                    validation_data=(np.array(X_test), y_test))

history_dict = history.history
history_dict.keys()


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()



plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()