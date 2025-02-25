import cv2,os

data_path = "C:/Users/MRUH/Desktop/Face mask detection/Dataset"
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))
print(label_dict)
print(categories)
print(labels)

#222222222
img_size=100
data=[]
target=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
    
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img= cv2.imread(img_path)
        
        try:
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            resized=cv2.resize(gray,(img_size,img_size))
            
            data.append(resized)
            
            target.append(label_dict[category])
        except Exception as e:
            print("Exception:",e)
            
#33333333333
import numpy as np
    
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
    
target=np.array(target)
    
from keras.utils import np_utils
new_target=np_utils.to_categorical(target)
#4444
np.save('data',data)
np.save('target',new_target)
#555555
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
model = Sequential()

model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

