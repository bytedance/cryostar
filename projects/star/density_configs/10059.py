exp_name = ""
eval_mode = False
ckpt_path = None

data = dict(
    dataset_dir="empiar_link/10059/data",
    starfile_name="J3_particles_exported.star",
    starfile_apix=1.2156,
    side_shape=192,
    voxel_size=1.2156,
    train_batch_per_gpu=16,
    val_batch_per_gpu=64,
    workers_per_gpu=4,
    f_mu=0.0,
    f_std=169.59818,
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
    cs=2.0,
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
