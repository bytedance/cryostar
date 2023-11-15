# Users essentially only need to modify the parameters here to adapt to their own dataset.
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
    # ref PDB structure file path
    ref_pdb_path="xxx/xxx.pdb",
)

extra_input_data_attr = dict(
    # NMA meta data path
    nma_path="",
    # use protein domain or not, this is now not included in the released version
    use_domain=False,
    # domain meta data path
    domain_path=None,
    # checkpoint path if you want to resume from other experiment
    ckpt_path=None
)

data_process = dict(
    # down-sampled shape. if it is None, it will be the dataset_attr.side_shape
    # if dataset_attr.side_shape < 256, else it will be set to 128 automatically
    down_side_shape=None,
    # control if gt images will be masked, defined by the radius in ratio
    mask_rad=1.0,
    # optional: low-pass filter bandwidth determined by the FSC between Gaussian
    # density and consensus map, typically use 10
    low_pass_bandwidth=10.,
)

data_loader = dict(
    # training mini-batch size per GPU
    train_batch_per_gpu=64,
    # validation mini-batch size per GPU, can be slightly larger than training
    val_batch_per_gpu=128,
    workers_per_gpu=4,
)

# random seed
seed = 1
# you can customize experiment name here, by default it is named by the python script and config file name
exp_name = ""
# do evaluation only, usually False
eval_mode = False
# init the decoder outputs to be 0 or not
do_ref_init = True

# use trainable Gaussian density parameter or not, keep it False otherwise the gmm parameter will fit noise
gmm = dict(tunable=False)

# model related parameters
model = dict(model_type="VAE",
             input_space="real",
             model_cfg=dict(
                 encoder_cls='MLP',
                 decoder_cls='MLP',
                 # number of neurons in encoder (e_) every layer
                 e_hidden_dim=(512, 256, 128, 64, 32),
                 # the latent space dimension
                 latent_dim=8,
                 # number of neurons in decoder (d_) every layer
                 d_hidden_dim=(32, 64, 128, 256, 512),
                 # number of layers in encoder, should be equal to len(e_hidden_dim)
                 e_hidden_layers=5,
                 # number of layers in decoder, should be equal to len(d_hidden_dim)
                 d_hidden_layers=5,
             ))

# loss and regularization related parameters
loss = dict(
    # intra-chain cutoff in protein chains, related to the cutoff defines k_{EN} in the paper
    intra_chain_cutoff=12.,
    # always 0
    inter_chain_cutoff=0.,
    # always None
    intra_chain_res_bound=None,
    # intra-chain cutoff in DNA/RNA chains
    nt_intra_chain_cutoff=15.,
    # inter-chain cutoff in DNA/RNA chains
    nt_inter_chain_cutoff=15.,
    # always None
    nt_intra_chain_res_bound=None,
    # related to cutoff k_{clash} in the paper
    clash_min_cutoff=4.0,
    # mask radius ratio used for image loss, only regions in the mask will be calculated
    mask_rad_for_image_loss=0.9375,
    # L_{image} loss weights
    gmm_cryoem_weight=1.0,
    # L_{cont} loss weights
    connect_weight=1.0,
    # deprecated
    sse_weight=0.0,
    # L_{EN} loss weights
    dist_weight=1.0,
    # dist_penalty_weight=1.0,
    # related to the p-th percentile in the paper
    dist_keep_ratio=0.99,
    # L_{clash} loss weights
    clash_weight=1.0,
    # learning rate warmup steps
    warmup_step=10000,
    # L_{KL} loss weight upper bound
    kl_beta_upper=0.5,
    free_bits=3.0)

# optimizer related parameters, define learning rate here
optimizer = dict(lr=1e-4, )

# evaluation settings
analyze = dict(cluster_k=10, skip_umap=True, downsample_shape=112)

# print log on console every 50 steps
runner = dict(log_every_n_step=50, )

# lightning trainer setup, you may need to change devices number to the available number of GPUs you can use
trainer = dict(max_steps=96000,
               devices=4,
               precision="16-mixed",
               num_sanity_val_steps=0,
               val_check_interval=12000,
               check_val_every_n_epoch=None)
