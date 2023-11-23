# You need to modify the dataset related information here
dataset_attr = dict(
    # where your dataset located, specify the directory location where the first-level directory of
    # the .mrcs file in your starfile is located.
    dataset_dir="xxx/xxx/xxx",
    # specify the path of your starfile
    starfile_path="xxx/xxx/xxx/xxx.star",
    # original pixel size of the dataset, if None, this value will be inferred from starfile
    apix=None,
    # original shape of the dataset, if None, this value will be inferred from starfile
    side_shape=None,
    # mu and std for images in fourier space, will calculate automatically before training
    f_mu=None,
    f_std=None,
)

# You need to alter the given_z value here
extra_input_data_attr = dict(
    # Here does the magic that using the latent space in step 1 as input!
    # change the value to where the latest z.npy locate from step 1.
    given_z=None,
    # checkpoint path if you want to resume from other experiment
    ckpt_path=None,
)

data_process = dict(
    # down-sampled shape, also can be the original data shape, this should be a tradeoff between output
    # resolution and computation resource. if None, it will be the dataset_attr.side_shape
    # if dataset_attr.side_shape < 256, else it will be set to 256 automatically
    down_side_shape=None,
    # control if gt images will be masked, defined by the radius in ratio
    mask_rad=1.0,
)

data_loader = dict(
    # training mini-batch size per GPU
    train_batch_per_gpu=16,
    # validation mini-batch size per GPU, can be slightly larger than training
    val_batch_per_gpu=64,
    workers_per_gpu=4,
)

# same as the atom config above
exp_name = ""
# do evaluation only, usually False
eval_mode = False

# model related parameters
model = dict(shift_data=False,
             enc_space="fourier",
             hidden=1024,
             z_dim=8,
             pe_type="gau2",
             net_type="cryodrgn", )

# loss type
loss = dict(
    loss_fn="fmsf",
    mask_rad_for_image_loss=1,
)

# lightning trainer setup, you may need to change devices number to the available number of GPUs you can use
trainer = dict(
    # max_epochs, usually 5 is enough for large protein, some dataset may need 20 epochs to get better result
    max_epochs=20,
    devices=4,
    precision="16-mixed",
    num_sanity_val_steps=0,
    val_check_interval=None,
    check_val_every_n_epoch=5
)
