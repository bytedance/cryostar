exp_name = ""
eval_mode = False
ckpt_path = None

data = dict(
    dataset_dir="empiar_link/10180/data",
    starfile_name="Example/consensus_data.star",
    starfile_apix=1.7,
    side_shape=256,
    voxel_size=320 * 1.7 / 256,
    train_batch_per_gpu=16,
    val_batch_per_gpu=64,
    workers_per_gpu=4,
    f_mu=0.0,
    f_std=330.53455,
)

model = dict(shift_data=False,
             enc_space="fourier",
             hidden=1024,
             z_dim=8,
             pe_type="gau2",
             net_type="cryodrgn",
             given_z=None)

ctf = dict(
    size=data["side_shape"],
    resolution=data["voxel_size"],  # equal to voxel_size
    kV=300,
    valueNyquist=1.,
    cs=2.7,
    amplitudeContrast=0.1,
)

mask = dict(mask_rad=1)

loss = dict(loss_fn="fmsf")

trainer = dict(
    max_epochs=20,
    max_steps=999999,
    devices=4,
    precision="16-mixed",
    eval_every_step=0,
    eval_every_epoch=5,
)
