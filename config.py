# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5
lr = 10 ** log10_lr
epochs = 1500
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 1
lamda_guide = 10
lamda_low_frequency = 1
lambda_m = 0.5
device_ids = [0]

# Train:
batch_size = 2
cropsize = 512
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 500
batchsize_val = 1
shuffle_val = False
val_freq = 10


# Dataset
TRAIN_PATH = '/gdata/cold1/zhangxuanyu/StegFormer/DIV2K/DIV2K_train_HR/'
VAL_PATH = '/gdata/cold1/zhangxuanyu/HiNet/demo1/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = 'checkpoint/'

checkpoint_on_error = True
SAVE_freq = 10

IMAGE_PATH = '/data03/zxy/OmniGuard/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_temp = IMAGE_PATH + 'temp/'
IMAGE_PATH_fuse = IMAGE_PATH + 'fuse/'
IMAGE_PATH_demo = IMAGE_PATH + 'demo/'

# 512
suffix = 'model_checkpoint_01500.pt'
tain_next = True
trained_epoch = 0