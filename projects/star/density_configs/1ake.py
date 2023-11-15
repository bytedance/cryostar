dataset_attr = dict(
    dataset_dir="tutorial_data_1ake/uniform_snr0-0001_ctf",
    starfile_path="tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star",
    apix=1.0,
    side_shape=128,
    f_mu=0.0,
    f_std=330.53455,
)

extra_input_data_attr = dict(
    given_z=None,
    ckpt_path=None,
)

data_process = dict(
    down_side_shape=128,
    mask_rad=1.0,
)

data_loader = dict(
    train_batch_per_gpu=16,
    val_batch_per_gpu=64,
    workers_per_gpu=4,
)

exp_name = ""
eval_mode = False

model = dict(shift_data=False,
             enc_space="fourier",
             hidden=1024,
             z_dim=8,
             pe_type="gau2",
             net_type="cryodrgn",)

# control volume supervised region
mask = dict(mask_rad=1)

loss = dict(loss_fn="fmsf")

trainer = dict(
    max_epochs=5,
    devices=4,
    precision="16-mixed",
    eval_every_step=0,
    eval_every_epoch=5,
)
