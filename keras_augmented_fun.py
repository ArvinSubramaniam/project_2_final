import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def keras_augmentation(img,i, augmentation_rate = 2):
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    aug_img = []
    
    j = 0;
    for batch in datagen.flow(x, batch_size=1, seed = i):
        img_new = batch.reshape((400,400,-1))
        aug_img.append(img_new)
        
        j += 1
        if j > augmentation_rate:
            break  # otherwise the generator would loop indefinitely
    return aug_img