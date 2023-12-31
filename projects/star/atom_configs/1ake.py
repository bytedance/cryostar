dataset_attr = dict(
    dataset_dir="tutorial_data_1ake/uniform_snr0-0001_ctf",
    starfile_path="tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star",
    apix=1.0,
    side_shape=128,
    ref_pdb_path="tutorial_data_1ake/pdbs/1akeA_50.pdb",
)

extra_input_data_attr = dict(
    nma_path="",
    use_domain=False,
    domain_path=None,
    ckpt_path=None
)

data_process = dict(
    down_side_shape=None,
    mask_rad=1.0,
    # optional
    low_pass_bandwidth=4.,
)

data_loader = dict(
    train_batch_per_gpu=64,
    val_batch_per_gpu=128,
    workers_per_gpu=4,
)

seed = 1
exp_name = ""
eval_mode = False
do_ref_init = True

gmm = dict(tunable=False)

model = dict(model_type="VAE",
             input_space="real",
             ctf="v1",
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
    mask_rad_for_image_loss=0.9375,
    gmm_cryoem_weight=1.0,
    connect_weight=1.0,
    sse_weight=0.0,
    dist_weight=1.0,
    # dist_penalty_weight = 1.0,
    dist_keep_ratio=0.8,
    clash_weight=1.0,
    warmup_step=10000,
    kl_beta_upper=0.5,
    free_bits=3.0)

optimizer = dict(lr=1e-4, )

analyze = dict(cluster_k=10, skip_umap=True)

runner = dict(log_every_n_step=50, )

trainer = dict(max_steps=24000,
               devices=4,
               precision="16-mixed",
               num_sanity_val_steps=0,
               val_check_interval=6000,
               check_val_every_n_epoch=None)
