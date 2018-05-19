
# building the CNN
from keras.models import Sequential # initialize NN
from keras.layers import Convolution2D # used for conv step and to add conv layer
from keras.layers import MaxPooling2D # add pooling layers
from keras.layers import Flatten # to make 2D vector
from keras.layers import Dense # add fully conn layers

# initialize the CNN
# 'classifier' is an obj of Seq class
classifier = Sequential()

# conv step1-choose no. of feature detectors
# 32 feature detec with kernel dim 3x3
# 3 RBG channels
# 64x64 is dim of i/p imgs
# act fn removes -ve pixels coz we want non-linearity
classifier.add(Convolution2D(32,3,3, input_shape=(128,128,3), activation = 'relu'))

# end of layer 1

# pooling reduces size of feature map
# pool size will be 2x2. it still keeps precision
classifier.add(MaxPooling2D(pool_size=(2,2)))

# to improve performance
classifier.add(Convolution2D(16,3,3, activation = 'relu'))    #########
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

# now add ann part
# adding hidden layer (fully conn)
# 128 nodes in hidden layer
# relu aka rectifier fn
classifier.add(Dense(output_dim = 128, activation = 'relu'))   #########

# sigmoid fn is for binary classi
# softmax for multi-class
classifier.add(Dense(output_dim = 24, activation = 'softmax'))

# binary_crossentropy for binary classi
classifier.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# now, we do img pre-process
# imgs are distorted on purpose before fitting to avoid over-fitting
# keras pre-processing
# image augmentation allows us to prevent over-fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,  # pixel value will be btwn 0&1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('https://drive.google.com/drive/folders/1wgXtF6QHKBuXRx3qxuf-o6aOmN87t8G-',
                                                 target_size = (128, 128), ###########
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (128, 128),     ###########
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fits model in train while testing performance in test
classifier.fit_generator(training_set,
                         steps_per_epoch = 2258, # no. of imgs in train
                         epochs = 10,
                         validation_data = test_set,  # test set on which u wanna evaluate performance
                         validation_steps = 240) # no of img in test set

# increase target_size or add more conv/dense layers to improve accuracy

