# random seed
seed = 1
# The name of the experiment. This affects the name of output directory.
exp_name = ""
# Whether you want to do evaluation only, usually False.
eval_mode = False
# init the decoder outputs to be 0 or not
do_ref_init = True

# ==========================================
# Where is your dataset and how it looks like?
# ------------------------------------------
#
# Parameters:
#   dataset_dir: the path of your dataset.
#       This parameter specifies the directory location where the top-level directory of
#       the .mrcs file in your starfile is located.
#   starfile_path: the path of your starfile.
#   ref_pdb_path: the path of a reference pdb structure.
#   apix: original pixel size of the dataset.
#       If None, this value will be inferred from starfile
#   side_shape: original shape of the dataset.
#       If None, this value will be inferred from starfile.
dataset_attr = dict(
    # >>> important parameters you may change <<<
    dataset_dir="xxx/xxx/xxx",
    starfile_path="xxx/xxx/xxx/xxx.star",
    ref_pdb_path="xxx/xxx.pdb",
    # >>> default ones work well <<<
    apix=None,
    side_shape=None,
)

# ==========================================
# How to preprocess the data?
# ------------------------------------------
#
# Parameters:
#   down_side_shape: the downsampled side length.
#       If None, it will be set to the min(128, dataset_attr.side_shape)
#       Tip: You can set it to the original side length without pains if the
#           result looks odd.
#   mask_rad: the radius in ratio (0.0~1.0) that the particles will be masked
#   low_pass_bandwidth: low-pass filter bandwidth of the Gaussian density.
#       Oftern determined by the FSC between Gaussian density and consensus map.
#       We set it to 10 by default. This parameter is not sensitive.
data_process = dict(
    # >>> important parameters you may change <<<
    down_side_shape=None,
    # >>> default ones work well <<<
    mask_rad=1.0,
    low_pass_bandwidth=10.,
)

# ==========================================
# Some extra information of the input
# ------------------------------------------
#
# Parameters:
#   nma_path: NMA meta data path.
#   use_domain: use protein domain or not, this is now not included in the released version
#   domain_path: domain meta data path
#   ckpt_path: checkpoint path if you want to resume from other experiment
extra_input_data_attr = dict(
    # >>> default ones work well <<<
    nma_path="",
    use_domain=False,
    domain_path=None,
    ckpt_path=None)

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
    train_batch_per_gpu=64,
    # >>> default ones work well <<<
    val_batch_per_gpu=128,
    workers_per_gpu=4,
)

# ==========================================
# How does the Gaussian density look like?
# ------------------------------------------
#
# Parameters:
#   tunable: use trainable Gaussian density parameter or not.
#       Tip: just keep it to be False otherwise the gmm parameter will fit noise.
gmm = dict(tunable=False)

# ==========================================
# How to modify the model architecture?
# ------------------------------------------
#
# Parameters:
#   e_hidden_dim: number of neurons in each layer of the encoder.
#   latent_dim: the latent space dimension.
#   d_hidden_dim: number of neurons in each layer of the decoder.
#   e_hidden_layers: number of layers in encoder, should be equal to len(e_hidden_dim)
#   d_hidden_layers: number of layers in decoder, should be equal to len(d_hidden_dim)
#
model = dict(
    # >>> default ones work well <<<
    model_type="VAE",
    input_space="real",
    model_cfg=dict(
        encoder_cls='MLP',
        decoder_cls='MLP',
        e_hidden_dim=(512, 256, 128, 64, 32),
        latent_dim=8,
        d_hidden_dim=(32, 64, 128, 256, 512),
        e_hidden_layers=5,
        d_hidden_layers=5,
    ))

# ==========================================
# What is the loss function?
# ------------------------------------------
loss = dict(
    # >>> default ones work well <<<
    # related to cutoff k_{EN} in the paper
    intra_chain_cutoff=12.,
    inter_chain_cutoff=0.,
    intra_chain_res_bound=None,
    nt_intra_chain_cutoff=15.,
    nt_inter_chain_cutoff=15.,
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

# ==========================================
# How do you want to train the model?
# ------------------------------------------
#
# Parameters:
#   max_steps: how many steps you would like to train cryoSTAR.
#       Tip-1: We suggest use 96000 by default.
#       Tip-2: Increasing it may always help.
#
#   devices: how many GPU you want to use.
#       It must not exceed the total number of GPUs of your machine.
trainer = dict(
    # >>> important parameters you may change <<<
    max_steps=96000,
    devices=4,
    # >>> default ones work well <<<
    precision="16-mixed",
    num_sanity_val_steps=0,
    val_check_interval=12000,
    check_val_every_n_epoch=None)
