
import pandas as pd
import os
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy import misc
from keras.callbacks import ModelCheckpoint
from random import shuffle
import glob
import os


image_size = (224, 224, 3)

print("Suffle, Decrease Learning Rate, Checkpoint")
print("Reading csv file ...")


csvfile = pd.read_csv('./all_products_3m_with_images.csv')




cmap = {}
pmap = {}
j = 0
for i in range(len(csvfile.ProductID)):
    pid = csvfile.ProductID[i]
    cid = csvfile.CategoryID[i]
    if cid not in cmap:
        cmap.update({cid : j})
        j += 1
    if pid not in pmap:
        pmap.update({pid : cmap[cid]})

print("Spliting data into train, validation, test")



shuffle_data = True  # shuffle the addresses before saving
path = './imgs_3m/'  # address to where you want to save

# read addresses and labels from the 'train' folder
addrs = os.listdir(path)
labels = [pmap[int(filename.split('_')[0])] for filename in addrs]

if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.7*len(addrs))]
train_labels = labels[0:int(0.7*len(labels))]
val_addrs = addrs[int(0.7*len(addrs)):]
val_labels = labels[int(0.7*len(addrs)):]


print("Done")


import numpy as np
import keras
print("Creating DataGenerator ...")

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=20, dim=(224,224), n_channels=3,
                 n_classes=126, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation__(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = Image.open(path + ID)
            img.load()
            img = misc.imresize(img, image_size)
            data = np.asarray(img, dtype='int32')
            X[i,] = data / 255

            # Store class
            y[i] = self.labels[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# In[6]:


training_generator = DataGenerator(train_addrs, train_labels)
validation_generator = DataGenerator(val_addrs, val_labels)


# In[12]:


from keras.applications import resnet50

model = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=None, pooling='max', classes=126)

filepath="/warehouse/weights_resnet50_126.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.compile(loss=keras.losses.categorical_crossentropy,S
              optimizer=keras.optimizers.Adam(lr=0.0005),
              metrics=['accuracy'])



model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 60, callbacks=callbacks_list, verbose=1)
