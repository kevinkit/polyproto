# polyproto
A generator based on geometric shapes to test a keras CNN. Often you want to test your CNN but you do not have any data, or the data you have is not quite fitting. With this repo you can change the width, height and amount of classes you want to differ with simple settings. Then polygons will be produced. The generators are working out of the box

# Install
```
pip install polyproto
```
# Usage

Simple example for showing the usage
```python
import polyproto as pp
from matplotlib import pyplot as plt
import numpy as np

# intialize a random Generator
gen = pp.generators.GeometricNGenerator()

cnt = 0
#check that it all works
x,y = gen.__getitem__(0)
for image,label in zip(x,y):
    
    plt.figure(cnt)
    plt.imshow(image)
    plt.title("Label: " + str(np.argmax(label)))
    cnt += 1
```

Advanced example with usage in Keras Environment

```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

import polyproto as pp 

gen = pp.generators.GeometricNGenerator(batch_mul=3,
                                        forms=5,
                                        seed=123,
                                        epoch_length=20)
val_gen = pp.generators.GeometricNGenerator(batch_mul=3,
                                        forms=5,
                                        seed=123,
                                        epoch_length=20)

base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# use the gens.forms member variable to adapt it accordingly 
predictions = Dense(gen.forms, activation='softmax')(x)
odel = Model(inputs=base_model.input, outputs=predictions)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["acc"])
hist = model.fit_generator(gen,validation_data=val_gen,epochs=20)            
```