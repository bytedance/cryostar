dataset_attr = dict(
    dataset_dir="empiar_link/10180/data",
    starfile_path="empiar_link/10180/data/Example/consensus_data.star",
    apix=1.7,
    side_shape=320,
    f_mu=0.0,
    f_std=330.53455,
)

extra_input_data_attr = dict(
    given_z=None,
    ckpt_path=None,
)

data_process = dict(
    down_side_shape=256,
    mask_rad=1.0,
)

data_loader = dict(
    train_batch_per_gpu=12,
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
