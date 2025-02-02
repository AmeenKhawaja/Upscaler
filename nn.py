import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, UpSampling2D
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import array_to_img
from constants import *
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import datetime
import h5py


def process_high_and_low_quality_image(high_res_images_path, low_res_images_path):
    """
    This method loads and processes a single pair of images, which is one high resolution image and its corresponding
    low resolution image. It then resizes the high_res image into IMAGE_SIZE (from constants.py) which in this case is
    (128x128), and the low_res image is resized into half the dimensions of the high_res, so (64x64)

    For example, if we take a look at the DIV2K dataset and X2 dataset, we have the following:
    Image 1 from DIV2K_train_HR: 0001.png
    Image 1 from X2 Dataset: 0001x2.png
    These are two corresponding images that are the same, the first being a high quality version of the image, and the
    second being a low quality version of the image.
    """
    high_res = Image.open(high_res_images_path)
    low_res = Image.open(low_res_images_path)
    # over here the images are being resized. In the original dataset, DIV2K, high res images are exactly twice the
    # width and height of low_res images.
    # so, the high_res image is resized to 128x128, whereas the low_res image is resized to exactly half that. (64x64)
    # This maintains the exact same pattern as the DIV2K dataset, just at lower resolutions so our computers can
    # train this large network.
    high_res = resize_image(high_res)
    new_lr_height = IMAGE_SIZE[0] // 2
    new_lr_width = IMAGE_SIZE[1] // 2
    low_res = resize_image(low_res, (new_lr_height, new_lr_width))

    # lastly, using tensorflow's img_to_array method, we convert the images to arrays and normalize it.
    high_res_to_normalized_array = img_to_array(high_res) / 255.0
    low_res_to_normalized_array = img_to_array(low_res) / 255.0
    return high_res_to_normalized_array, low_res_to_normalized_array,


def load_dataset(high_res_images_path, low_res_images_path):
    """
    This method goes through the entire DIV2K high resolution image directory and the low resolution image directory X2.
    It computes every low_res and high_res image pair and returns an entire dataset of high resolution and low
    resolution images as numpy arrays, which is what is used in training/testing the model with TensorFlow.
    """
    all_high_res_images = []
    all_low_res_images = []

    # this retrieves the entire 800 image file names from the dataset. It had to be sorted because without sorting it,
    # the image pairs would mix and match with one another incorrectly.
    high_res = sorted(os.listdir(high_res_images_path))
    low_res = sorted(os.listdir(low_res_images_path))

    # over here, we combine the high_res and low_res lists into pairs of the image names
    # so, if we had: '0001.png' and '0001x2.png', it would combine it into a pair like this: [(00001.png, 0001x2.png)]
    for high_res_images, low_res_images in zip(high_res, low_res):
        # combining the directory path and image file names
        high_res_path = os.path.join(high_res_images_path, high_res_images)
        low_res_path = os.path.join(low_res_images_path, low_res_images)
        high_res_image, low_res_image = (process_high_and_low_quality_image(high_res_path, low_res_path))
        all_high_res_images.append(high_res_image)
        all_low_res_images.append(low_res_image)
    np_high_res_images = np.array(all_high_res_images)
    np_low_res_images = np.array(all_low_res_images)

    # returns a list of processes high res and low res images that are numpy arrays,
    # which will be used in training/testing
    return np_high_res_images, np_low_res_images


def resize_image(image, image_width_and_height=None):
    """
    This method resizes the images within the dataset to IMAGE_SIZE (from constants.py), while maintaining the aspect
    ratio. For this project, the dimensions of the image have remained 128x128 because we don't have the hardware
    capability of running any larger of an image dimension.
    We have tried on our laptops/computers running it at 256x256, and even at full scale of
    the image, but, it would take hours to train the model for just 50 epochs which produced poor results still.

    To successfully implement this, the Pillow package was used, along with two different online resources to help
    with the coding of this, which are linked below:
    https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/ (helpful source for padding)

    How the method works:
    It first creates a copy of the original image from the dataset and then it resizes the image using the pillow
    method, "thumbnail". By using thumbnail, the image maintains its aspect ratio. We then create a new image that is
    initially a black background. So, at this point, there is a 128x128 black background image, and now, the original
    image needs to be placed within the center of this. This is done so by calculating its left/right/top paddings
    and then pasting the image with those new calculated dimensions.
    """
    if image_width_and_height is None:
        image_width_and_height = IMAGE_SIZE
    height_of_image = image_width_and_height[0]
    width_of_image = image_width_and_height[1]
    copy_of_original_image = image.copy()
    # initially, the results of the images were very blurry when being compressed to 128x128, so the Pillow library has
    # a built in high quality image resampling method LANCZOS. By using this, it significantly reduced the blurring.
    copy_of_original_image.thumbnail((width_of_image, height_of_image), Image.Resampling.LANCZOS)
    resized_image = Image.new('RGB', (width_of_image, height_of_image), (0, 0, 0))
    first_dimension = width_of_image - copy_of_original_image.size[0]
    second_dimension = width_of_image - copy_of_original_image.size[1]
    left_side_padding = first_dimension // 2
    padding_top = second_dimension // 2
    resized_image.paste(copy_of_original_image, (left_side_padding, padding_top))
    return resized_image


def upscale_neural_network():
    """
    I created this model with less layers to train faster and gather results, and we also tried out the PreLu
    activation function that is built into tensorflow. I also looked into the PreLu function from other sources, and
    apparently the consensus is that it is better than ReLu as it prevents overfitting. Our testing shows nothing
    noticable.
    https://www.sciencedirect.com/topics/computer-science/parametric-rectified-linear-unit#:~:text=A%20Parametric%20Rectified%20Linear%20Unit,parameters%20through%20the%20backpropagation%20algorithm.
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/PReLU
    """
    model = Sequential([
        Conv2D(32, kernel_size=3, padding='same', input_shape=(64, 64, 3)),
        PReLU(),
        BatchNormalization(),
        Conv2D(16, kernel_size=1, padding='same'),
        PReLU(),
        Conv2D(32, kernel_size=3, padding='same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=3, padding='same'),
        Conv2D(24, kernel_size=3, padding='same'),
        PReLU(),
        UpSampling2D(size=(2, 2)),
        PReLU(),
        Conv2D(16, kernel_size=1, padding='same'),
        PReLU(),
        Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')
    ])
    return model


def upscale_neural_network_2():
    """
    This is described in depth in the report, but a TLDR is it has more features than model 1 per layer
    and performs better than model 1 and 3.
    """
    model = Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.Conv2D(3, 3, padding='same', activation='sigmoid')
    ])
    return model


def upscale_neural_network_3():
    """
    This was another model I built to test the performance of the model with even more layers.
    It's described in depth in the report.
    """
    model = Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(96, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(96, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.UpSampling2D(size=(2, 2)),
        layers.Conv2D(96, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(3, 3, padding='same', activation='sigmoid')
    ])
    return model


def train_upscale_nn(high_res_images, low_res_images):
    """
    This method just trains our models. Three total models you can run.
    """
    model = upscale_neural_network() # model 1
    # model = upscale_neural_network_2() # model 2
    # model = upscale_neural_network_3() # model 3

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mean_squared_error', lambda y_true, y_pred: tf.image.psnr(y_true, y_pred, max_val=1.0)]
    )
    model.summary()
    # using tensorboard to gather data: https://www.tensorflow.org/tensorboard/get_started
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        low_res_images, high_res_images,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[tensorboard_callback]
    )

    return model, history


def display_and_save_images(model, low_res_image, high_res_image):
    """
    This method outputs an image from the dataset (of the users choice from constant.py), where it shows the
    low resolution image, the super resolution image generated by the network, and the high resolution image from
    the dataset.
    """
    super_res_image = model.predict(low_res_image[SAVE_IMAGE_NUMBER:SAVE_IMAGE_NUMBER + 1])[0]

    # using tensorflows resize function, we resize the low_res image to 128x128 which is the same resolution as the
    # high_res image. This way, we can calculate the PSNR which will indicate just how well the network learned.
    resize_low_and_super_to_128x128 = tf.image.resize(low_res_image[SAVE_IMAGE_NUMBER],
                                                      (high_res_image[SAVE_IMAGE_NUMBER].shape[0],
                                                       high_res_image[SAVE_IMAGE_NUMBER].shape[1]),
                                                      method='bicubic')

    # comparing the two PSNR values to see the performance
    # code taken straight from tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/image/psnr
    print(
        f"PSNR -> (low_res vs high_res ): {tf.image.psnr(resize_low_and_super_to_128x128, high_res_image[SAVE_IMAGE_NUMBER], max_val=1.0).numpy()} dB")
    print(
        f"PSNR -> (super_res vs high_res ): {tf.image.psnr(super_res_image, high_res_image[SAVE_IMAGE_NUMBER], max_val=1.0).numpy()} dB")
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(low_res_image[SAVE_IMAGE_NUMBER])
    plt.title(f"Low Res {low_res_image[SAVE_IMAGE_NUMBER].shape}")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(super_res_image)
    plt.title(f"Super Res {super_res_image.shape}")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(high_res_image[SAVE_IMAGE_NUMBER])
    plt.title(f"High Res {high_res_image[SAVE_IMAGE_NUMBER].shape}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_IMAGES_PATH, bbox_inches='tight', dpi=300)
    print(f"image saved at: {SAVE_IMAGES_PATH}")
    plt.close()


def reuse_image_data(high_res_images, low_res_images, filepath='image_data.h5'):
    """
    This method creates a .h5 file and saves the high_res and low_res image array contents in it.
    The reason this is done is to save lots of time when re-running the program, it doesn't need to calculate all the
    array pixel values over and over again, instead, it reloads it.
    """
    with h5py.File(filepath, 'w') as hf:
        hf.create_dataset('high_res_images', data=high_res_images)
        hf.create_dataset('low_res_images', data=low_res_images)
    print(f"data saved for reusing later in {filepath} directory")


def load_reused_image_data(filepath='image_data.h5'):
    """
    This method loads the image data from the h5 file that initially saves all the image data. This way, every time
    the program is ran, it doesn't need to compute all the image pairs over again, and instead, I can reuse them.
    https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python

    """
    with h5py.File(filepath, 'r') as hf:
        high_res = np.array(hf['high_res_images'])
        low_res = np.array(hf['low_res_images'])
    print(f"reused image data has been loaded")
    return high_res, low_res


def create_single_super_res_image():
    loaded_model = load_model(LOAD_MODEL, compile=False)
    low_img_resized = resize_image(Image.open(SINGLE_PHOTO_TEST_PATH), (64, 64))
    # since conv2d from tensorflow takes a 4d input, we insert a new dimension to make its shape (1,64,64,3)
    # https://stackoverflow.com/questions/66426381/what-is-the-use-of-expand-dims-in-image-processing
    low_img_array = np.expand_dims(img_to_array(low_img_resized) / 255.0, axis=0)

    super_res_result = loaded_model.predict(low_img_array)[0]
    super_res_array_to_image = array_to_img(super_res_result)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(low_img_resized)
    plt.title("Low Resolution")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(super_res_array_to_image)
    plt.title("Super Resolution")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(SINGLE_PHOTO_OUTPUT_PATH, dpi=300)
    print(f"Super-resolved image saved as {SINGLE_PHOTO_OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    if MODEL_TRAINED is False:
        reuse_images_to_save_time = 'DIV2K_TO_ARRAY.h5'

        if os.path.exists(reuse_images_to_save_time):
            print("loading reused image data")
            hr_images, lr_images = load_reused_image_data(reuse_images_to_save_time)
        else:
            print("no reused image data found. starting image dataset processing.")
            hr_images, lr_images = load_dataset(HR_PHOTOS_PATH, LR_PHOTOS_PATH)
            reuse_image_data(hr_images, lr_images, filepath=reuse_images_to_save_time)
        model, history = train_upscale_nn(hr_images, lr_images)
        model.save('RecentlyRanModel.h5')
        display_and_save_images(model, lr_images, hr_images)
    elif MODEL_TRAINED is True:
        create_single_super_res_image()
