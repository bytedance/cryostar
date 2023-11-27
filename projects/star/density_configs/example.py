# The name of the experiment. This affects the name of output directory.
exp_name = ""
# Whether you want to do evaluation only, usually False.
eval_mode = False

# ==========================================
# Where is your dataset and how it looks like?
# ------------------------------------------
#
# Parameters:
#   dataset_dir: the path of your dataset.
#       This parameter specifies the directory location where the top-level directory of
#       the .mrcs file in your starfile is located.
#   starfile_path: the path of your starfile.
#   apix: original pixel size of the dataset.
#       If None, this value will be inferred from starfile
#   side_shape: original shape of the dataset.
#       If None, this value will be inferred from starfile.
#   f_mu/f_std: the mean and standard devisation for images in the fourier space.
#       We will calculate them automatically before training.
dataset_attr = dict(
    # >>> important parameters you may change <<<
    dataset_dir="xxx/xxx/xxx",
    starfile_path="xxx/xxx/xxx/xxx.star",
    # >>> default ones work well <<<
    apix=None,
    side_shape=None,
    f_mu=None,
    f_std=None,
)

# ==========================================
# How to preprocess the data?
# ------------------------------------------
#
# Parameters:
#   down_side_shape: the downsampled side length.
#       If None, it will be set to the min(256, dataset_attr.side_shape)
#       Tip: Considering lowering it (to 128) if your computation resource is limited.
#   mask_rad: the radius in ratio (0.0~1.0) that the particles will be masked
data_process = dict(
    # >>> important parameters you may change <<<
    down_side_shape=None,
    # >>> default ones work well <<<
    mask_rad=1.0,
)

# ==========================================
# Some extra information of the input
# ------------------------------------------
#
# Parameters:
#   given_z: a file storing the latent code of each particle.
#       Tip-1: You can set it to None and we can induce it automatically.
#       Tip-2: You can set it to z.npy generated from step 1.
#   ckpt_path: checkpoint path if you want to resume from other experiment
extra_input_data_attr = dict(
    # >>> important parameters you may change <<<
    given_z=None,
    ckpt_path=None,
)

# ==========================================
# How to load the data?
# ------------------------------------------
#
# Parameters:
#   train_batch_per_gpu: training mini-batch size per GPU.
#       Tip-1: Too small batch size may cause the model to be affected by noises.
#       Tip-2: In this example config, we control the training process through `max_epochs`.
#           In this case, a larger batch size may not always work well, since it may cause
#           a smaller number of total updating steps. The number of total steps is crucial to
#           the reconstruction quality.
#       In a word, you can increase the batch size (since deep learning prefers large batch size)
#       but if you find the result becomes worse, consider increase the `trainer.max_epochs` simutaneously!
#   val_batch_per_gpu: validation mini-batch size per GPU.
#       It can be slightly larger than training.
data_loader = dict(
    # >>> important parameters you may change <<<
    train_batch_per_gpu=16,
    # >>> default ones work well <<<
    val_batch_per_gpu=64,
    workers_per_gpu=4,
)

# ==========================================
# How to modify the model architecture?
# ------------------------------------------
#
# Parameters:
#   hidden: the hidden size of the model.
#       Tip: Change it to 256 may make training faster.
model = dict(
    # >>> default ones work well <<<
    shift_data=False,
    enc_space="fourier",
    hidden=1024,
    z_dim=8,
    pe_type="gau2",
    net_type="cryodrgn",
)

# ==========================================
# What is the loss function?
# ------------------------------------------
loss = dict(
    # >>> default ones work well <<<
    loss_fn="fmsf",
    mask_rad_for_image_loss=0.85,
    free_bits=3.0
)

# ==========================================
# How do you want to train the model?
# ------------------------------------------
#
# Parameters:
#   max_epochs: how many epochs you would like to train cryoSTAR.
#       Tip-1: We suggest use 20 by default.
#       Tip-2: Increasing it may always help.
#
#   devices: how many GPU you want to use.
#       It must not exceed the total number of GPUs of your machine.
trainer = dict(
    # >>> important parameters you may change <<<
    max_epochs=20,
    devices=4,
    # >>> default ones work well <<<
    precision="16-mixed",
    num_sanity_val_steps=0,
    val_check_interval=None,
    check_val_every_n_epoch=5)
