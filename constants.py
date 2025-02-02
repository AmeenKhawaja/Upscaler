EPOCHS = 30  # for best results, 30 - 100 epochs.
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.0001
IMAGE_SIZE = (128, 128) # no need to change this, it just saves every high-res image as 128x128

SAVE_IMAGE_NUMBER = 2  # this will output number x image in the div2k dataset. So, change this to any number between 1-800
MODEL_TRAINED = False  # set this to true if ONLY if u want to test a single image
SINGLE_PHOTO_TEST_PATH = 'blurry.png'  # this is where u feed ur own custom low qual image for the model to test. (low.png and blurry.png) You can add your own test image too!
SINGLE_PHOTO_OUTPUT_PATH = 'ImagesNN/SuperResPhoto.png' # This is the result of the single image you input. Only works if MODEL_TRAINED is True
LOAD_MODEL = 'TrainedModels/model_3_trained.h5'  # put the pre-trained model name you want to load here, e.g, model_3_trained.h5 and also set MODEL_TRAINED to False
HR_PHOTOS_PATH = 'DIV2K_train_HR' # high-resolution div2k dataset path
LR_PHOTOS_PATH = 'X2' # low-resolution downscale div2k dataset path
SAVE_IMAGES_PATH = 'ImagesNN/Photos' # After the model finishes training its epochs, it saves the image to this path

"""
run in terminal while nn.py is running to access visual graphs: tensorboard --logdir logs/fit
"""
