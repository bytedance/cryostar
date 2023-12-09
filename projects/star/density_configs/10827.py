dataset_attr = dict(
    dataset_dir="empiar_link/10827/rln_job009",
    starfile_path="empiar_link/10827/rln_job009/J40_particles.star",
    apix=0.9,
    side_shape=224,
    f_mu=0.0,
    f_std=197.95383,
)

extra_input_data_attr = dict(
    given_z=None,
    ckpt_path=None,
)

data_process = dict(
    down_side_shape=224,
    down_method="interp",
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
             shift_method="interp",
             enc_space="fourier",
             ctf="v1",
             hidden=1024,
             z_dim=8,
             pe_type="gau2",
             net_type="cryodrgn", )

# loss type
loss = dict(
    loss_fn="fmsf",
    mask_rad_for_image_loss=1,
    free_bits=3.0
)

trainer = dict(
    max_epochs=5,
    devices=4,
    precision="16-mixed",
    num_sanity_val_steps=0,
    val_check_interval=None,
    check_val_every_n_epoch=5
)
