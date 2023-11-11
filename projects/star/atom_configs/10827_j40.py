data = dict(
    dataset_dir="empiar_link/10827/rln_job009",
    starfile_name="J40_particles.star",
    starfile_apix=0.9,
    # ref structure
    ref_path="empiar_link/10827/res/7ptx_af2_J40_s16e1240.pdb",
    nma_path="",
    side_shape=224,
    voxel_size=0.9,
    lp_bandwidth=16.9,  # low-pass bandwidth
    # control if gt images will be masked, almost always None
    mask_rad=1.0,
    train_batch_per_gpu=64,
    val_batch_per_gpu=128,
    workers_per_gpu=4,
)

ckpt_path = None
use_domain = False
seed = 1
exp_name = ""
eval_mode = False
do_ref_init = True

mask = dict(
    # control the mask during training
    mask_rad=0.9375)

ctf = dict(
    size=data["side_shape"],
    resolution=data["voxel_size"],  # equal to voxel_size
    kV=300,
    valueNyquist=1.,
    cs=2.7,
    amplitudeContrast=0.1,
)

gmm = dict(tunable=False)

model = dict(model_type="VAE",
             input_space="real",
             model_cfg=dict(
                 encoder_cls='MLP',
                 decoder_cls='MLP',
                 e_hidden_dim=(512, 256, 128, 64, 32),
                 latent_dim=8,
                 d_hidden_dim=(512, 256, 128, 64, 32)[::-1],
                 e_hidden_layers=5,
                 d_hidden_layers=5,
             ))

loss = dict(
    intra_chain_cutoff=12.,
    inter_chain_cutoff=0.,
    intra_chain_res_bound=None,
    clash_min_cutoff=4.0,
    gmm_cryoem_weight=1.0,
    connect_weight=1.0,
    sse_weight=0.0,
    dist_weight=1.0,
    # dist_penalty_weight = 1.0,
    dist_keep_ratio=0.99,
    clash_weight=1.0,
    warmup_step=10000,
    kl_beta_upper=0.5,
    free_bits=3.0)

optimizer = dict(lr=1e-4, )

analyze = dict(cluster_k=10, skip_umap=True, downsample_shape=112)

runner = dict(log_every_n_step=50, )

trainer = dict(max_steps=24000,
               devices=4,
               precision="16-mixed",
               num_sanity_val_steps=0,
               val_check_interval=12000,
               check_val_every_n_epoch=None)
