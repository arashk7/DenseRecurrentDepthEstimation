from DRDE_AK.DRDE import DRDE
from PIL import Image
import numpy as np
import os
import time

_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


# Load the input stereo images from dataset directory
def load_input_dataset(path) -> object:
    files = os.listdir(path)
    x = np.zeros((len(files), int(100), int(300)))
    # x = np.zeros((len(files), int(128), int(512)))
    for i in range(len(files)):
        # img = load_image(path + "/" + files[i])
        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x


# Load the output disparity map from dataset directory
def load_output_dataset(path) -> object:
    files = os.listdir(path)
    x = np.zeros((len(files), int(100), int(150)))
    # x = np.zeros((len(files), int(128), int(256)))
    for i in range(len(files)):
        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x



# Initialize Dataset (network input)
path = "../../Dataset/DepthMap_dataset-master/Stereo"
x = load_input_dataset(path)
XL = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] / 2), 1))
XR = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] / 2), 1))
XL[:, :, :, 0] = x[:, :x.shape[1], :int(x.shape[2] / 2)]
XR[:, :, :, 0] = x[:, :x.shape[1], int(x.shape[2] / 2):]
min=np.min(np.min(XL))
max=np.max(np.max(XL))
XL= (XL-min)/max
min=np.min(np.min(XR))
max=np.max(np.max(XR))
XR= (XR-min)/max

# Initialize Dataset (network output)
path = "../../Dataset/DepthMap_dataset-master/Depth_map"
y = load_output_dataset(path)
Y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 1))
Y[:, :, :, 0] = y[:, :, :]
min = np.min(np.min(Y))
max = np.max(np.max(Y))
Y = (Y - min) / max

# Initialize the DispNet network
model_name = 'model_DRDE_1'
first_run = False
min_err = 1000
last_epoch = 0
last_best_epoch=0
drde = DRDE()
if first_run:
    # create model
    drde.init_model()
    drde.compile()
else:
    # load model
    drde.load_model_and_weight('model/' + model_name)
    drde.compile()
    min_err=drde.get_error_rate([XR, XL], Y, 4)
    last_epoch=drde.last_epoch
    last_best_epoch = drde.last_best_epoch

tic()
# Train loop
for i in range(last_epoch, 3000):
    # Train the model for one epoch
    drde.train([XR, XL], Y, n_epoch=1, batch_size=4)
    # Check the error
    err =drde.get_error_rate([XR, XL],Y, 4)

    # If the network improved then save the network weights
    if err < min_err:
        drde.save_model_and_weight('model/' + model_name,i,last_best_epoch)
        min_err = err
        last_best_epoch=i
        print("epoch: " + str(i) + "   error:" + str(err))
    else:
        print("epoch: " + str(i) + "   error:" + str(err))
        drde.save_model_and_weight('model/' + model_name + "_last_epoch", i, last_best_epoch)


tac()
