import tensorflow as tf 
import keras 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from keras.applications import VGG19
from keras.layers import Dense , Dropout,Convolution2D
vgg_model =  VGG19(include_top=True , weights='imagenet')
for models in vgg_model.layers:
  models.trainable= False
#converting from functionally model to sequential model
#removing the last 2 alyer to get rid of output layer in VGG16
vgg_model = keras.Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)
model = keras.Sequential()
for layer in vgg_model.layers:
  model.add(layer)
#add trianbles layers
model.add(Dense(4056, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
train_data = defect_tree = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/ASUS/Downloads/defect_detection/images', #path to dataset folder images
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/ASUS/Downloads/defect_detection/images',#path to dataset folder images
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
)
batch_size =32

model.fit(train_data,
    validation_data = test_data,callbacks=[early],
    epochs = 50)


model.evaluate(test_data)
#predict model
y_pred = np.array([])
y_true = np.array([])
i = 0

for image,label in test_data : 
  i+=1
  y = model.predict(image)
  y = np.argmax(y,axis=1)
  y_true = np.append(y_true,label)
  y_pred = np.append(y_pred,y)
  if i == 176 // 32 + 1:
    break
from  sklearn .metrics import classification_report,confusion_matrix
report=classification_report(y_true,y_pred)
print(report)
cm= confusion_matrix(y_true,y_pred)
print(cm)
import pandas as pd
import seaborn 
df_cm = pd.DataFrame(cm, index = [i for i in [0,1]],
                  columns = [i for i in [0,1]])
seaborn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
plt.title('confusion matrix')
plt.xlabel('prediction')
plt.ylabel('Actual')
model.save("textile.h5")