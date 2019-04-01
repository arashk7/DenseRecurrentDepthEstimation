from DRDE_AK.DRDE import DRDE
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt

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
    # x = np.zeros((len(files), int(128), int(512)))
    x = np.zeros((len(files), int(100), int(300)))
    for i in range(len(files)):
        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x


# Load the output disparity map from dataset directory
def load_output_dataset(path) -> object:
    files = os.listdir(path)
    # x = np.zeros((len(files), int(128), int(256)))
    x = np.zeros((len(files), int(100), int(150)))
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
model_name = 'model/model_DRDE_1'
dsn = DRDE()
dsn.load_model_and_weight(model_name)
dsn.compile()

# set a timer
tic()
id=212
# Predicting the disparity map
# x=[X[id:id+1,:,:,:],X[id:id+1,:,:,:]]
dsn.get_error_rate([XR, XL],Y)
p = dsn.model.predict([XR[id:id+1,:,:,:],XL[id:id+1,:,:,:]], batch_size=1, verbose=1)
img = Y[id, :, :, 0]
depth = p[0,:,:,0]
# Displaying the predicted disparity map
plt.imshow(img, cmap="hot")
plt.show()
plt.imshow(depth, cmap="hot")
plt.show()
# print the process time
tac()
