# Import Libraries
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping


'''
#### DATA AUGMENTATION FUNCTIONS ####
'''
# Set Parameters
'''
Parameters for data manipulation
'''
centerAngles = 0.015
lowerBound = 0.2
lowerBound2 = 0.3
upperBound = 0.98
ANGLE_SHIFT = 0.006


def filterSamples(labels, threshold, delRate):
    '''
    Function to create mask to filter dataset with based upon the threshold
    and the percentage to remove
    '''
    zero_idx = np.where(abs(labels)<=threshold)
    zero_idx = zero_idx[0]
    sizeDel = int(len(zero_idx) * delRate)
    return np.random.choice(zero_idx, sizeDel, replace=False)


def flipImages(images, angles):
    newImages = []
    newAngles = []
    for image, angle in zip(images, angles):
        if lowerBound2 < abs(angle) < upperBound:
            newImages.append(cv2.flip(image, 1))
            newAngles.append(angle * -1)

    if len(newAngles) > 0:  # To protect against no image transformation
        return np.append(images,np.array(newImages),axis=0), \
            np.append(angles,np.array(newAngles),axis=0)
    else:
        return images, angles


def imageShift(images, angles, maxShift=10):
    shiftedImages = []
    newAngles = []
    for image, angle in zip(images, angles):
        if lowerBound2 < abs(angle) < upperBound:
            shiftX = np.random.uniform(-maxShift, maxShift)
            shiftY = np.random.uniform(-maxShift, maxShift)
            M = np.float32([[1,0,shiftX],[0,1,shiftY]])
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            shiftedImages.append(image)

            angle = angle + ANGLE_SHIFT * shiftX
            newAngles.append(angle)

    if len(newAngles) > 0:
        return np.append(images,np.array(shiftedImages),axis=0),\
            np.append(angles,np.array(newAngles),axis=0)
    else:
        return images, angles


def brightnessAugmentation(images, angles):
    newImages = []
    newAngles = []
    for image, angle in zip(images, angles):
        newImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightnessAdjustment = np.random.uniform(0.1,1.1)
        newImage[:,:,2] = newImage[:,:,2] * brightnessAdjustment
        np.clip(newImage[:,:,2], 0, 255)
        newImages.append(cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR))
        newAngles.append(angle)

    if len(newAngles) > 0:
        return np.array(newImages), np.array(newAngles)
    else:
        return images, angles


def randomShadow(images, angles):
    shadowedImages = []
    newAngles = []
    for image, angle in zip(images,angles):
        # Get indicies of mask
        top_y = 320*np.random.uniform()
        top_x = 0
        bot_x = 160
        bot_y = 320*np.random.uniform()
        # convert image to hls color space
        image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        # create mask
        shadow_mask = 0*image_hls[:,:,1]
        X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
        shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
        if np.random.randint(2)==1:
            random_bright = .5
            cond1 = shadow_mask==1
            cond0 = shadow_mask==0
            if np.random.randint(2)==1:
                image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
            else:
                image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
        image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
        shadowedImages.append(image)
        newAngles.append(angle)

    if len(newAngles) > 0:
        return np.array(shadowedImages), np.array(newAngles)
    else:
        return images, angles


'''
#### DATA PREPROCESSING FUNCTIONS ####
'''

# Parameters
carWidth = 1.0
recoverydist = 4.0
horizon = 60
bonnet = 140


def getLabels(samples):
    y_labels = []
    for sample in samples:
        y_labels.append(float(sample[3]))
    return np.array(y_labels)


def loadImages(sample):
    image_center = cv2.imread(sample[0])
    image_left = cv2.imread(sample[1].replace(" ",""))
    image_right = cv2.imread(sample[2].replace(" ",""))
    angle_center = float(sample[3])
    angle_left = angle_center + (carWidth/recoverydist) / 25
    angle_right = angle_center - (carWidth/recoverydist) / 25
    return [image_center, image_left, image_right], [angle_center, angle_left, angle_right]


def resizeImages(images):
    images = [cv2.resize(image,(200,66), interpolation=cv2.INTER_AREA) for image in images]
    return images


def ChangeColourSpace(images):
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2YUV) for image in images]
    return images


def cropImages(images, maxShift=10):
    croppedImages = [image[horizon:bonnet,:] for image in images]
    return croppedImages


def generator(samples, batch_size=32):
    while 1:
        samples = shuffle(samples)

        # This augmentation is done here to make it random for each epoch
        angles = getLabels(samples)
        delIdx = filterSamples(np.array(angles), threshold=0.015, delRate=0.98)
        samplesRemoved = np.delete(np.array(samples), delIdx, axis=0)

        angles = getLabels(samplesRemoved)
        delIdx = filterSamples(np.array(angles), threshold=0.1, delRate=0.60)
        samplesRemoved = np.delete(np.array(samplesRemoved), delIdx, axis=0)

        num_samples = len(samplesRemoved)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samplesRemoved[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                allImages, allAngles = loadImages(batch_sample)
                images.extend(allImages)
                angles.extend(allAngles)

            # Augmentations
            images, angles = brightnessAugmentation(images, angles)
            images, angles = randomShadow(images, angles)
            images, angles = flipImages(images, angles)
            images, angles = imageShift(images, angles)

            # Pre Processing
            images = cropImages(np.array(images))
            images = resizeImages(images)
            images = ChangeColourSpace(images)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def generator_valid(samples, batch_size=32):
    while 1:
        samples = shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0].replace(" ",""))
                images.append(image)
                angle = float(batch_sample[3])
                angles.append(angle)

            images = cropImages(np.array(images))
            images = resizeImages(images)
            images = ChangeColourSpace(images)
            X_valid = np.array(images)
            y_valid = np.array(angles)
            yield shuffle(X_valid, y_valid)


'''

#### MODEL BUILDING AND TRAINING ####

'''

csv_directory = './data_udacity/driving_log.csv'
samples = []
with open(csv_directory) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Load the recovery data
csv_directory2 = './data_recovery/driving_log.csv'
with open(csv_directory2) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = np.array(samples[1:])


train_samples, valid_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
valid_generator = generator_valid(valid_samples, batch_size=32)

# Model Selection, Hyperparameters
dropoutRate = 0.5
num_epochs = 20
num_samples_train = 26142

print('Number of samples to train on: {}'.format(num_samples_train))

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(66,200,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), init='he_normal',
                        activation='elu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), init='he_normal',
                        activation='elu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), init='he_normal',
                        activation='elu'))
model.add(Convolution2D(64,3,3, subsample=(1,1), init='he_normal',
                        activation='elu'))
model.add(Convolution2D(64,3,3, subsample=(1,1), init='he_normal',
                        activation='elu'))
model.add(Flatten())
model.add(Dropout(dropoutRate))
model.add(Dense(100, init='he_normal', activation='elu'))
model.add(Dropout(dropoutRate))
model.add(Dense(50, init='he_normal', activation='elu'))
model.add(Dense(10, init='he_normal', activation='elu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mse', optimizer='adam')

# Set up training
filepath = './model-NVIDIA-{epoch:02d}-{val_loss:.5f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=.001, patience=5)
callbacks = [checkpoint, stopper]

history = model.fit_generator(train_generator, samples_per_epoch=num_samples_train,
                                    validation_data=valid_generator,
                                    nb_val_samples=len(valid_samples), nb_epoch=num_epochs,
                                    callbacks=callbacks)

print('Done! Im Out...')
