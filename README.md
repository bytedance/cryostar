# CryoSTAR

[🏠 Homepage](https://bytedance.github.io/cryostar/) |
[📚 User Guide](https://byte-research.gitbook.io/cryostar/)

`CryoSTAR` is a neural network based framework for recovering conformational heterogenity of protein complexes. By leveraging the structural prior and constraints from a reference `pdb` model, `cryoSTAR` can output both the protein structure and density map.

![main figure](./assets/main_fig.png)

## User Guide
The detailed user guide can be found at [here](https://byte-research.gitbook.io/cryostar/). This comprehensive guide provides in-depth information about the topic at hand. Feel free to visit the link if you're seeking more knowledge or need extensive instructions regarding the topic. 

## Installation

- Create a conda enviroment: `conda create -n cryostar python=3.9 -y`
- Install the package: `pip install .`

## Quick start

### Preliminary

You may need to prepare the resources below before running `cryoSTAR`:

- a concensus map (along with each particle's pose)
- a pdb file (which has been docked into the concensus map)

### Training
CryoSTAR operates through a two-stage approach where it independently trains an atom generator and a density generator. Here's an illustration of its process:

#### S1: Training the atom generator
In this step, we generate an ensemble of coarse-grained protein structures from the particles. Note that the `pdb` file is used in this step and it should be docked into the concensus map!

```shell
cd projects/star
python train_atom.py atom_configs/1ake.py
```

The outputs will be stored in the `work_dirs/atom_xxxxx` directory, and we perform evaluations every 12,000 steps. Within this directory, you'll observe sub-directories with the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
atom_xxxxx/
├── 0000_0000000/
├── ...
├── 0112_0096000/        # evaluation results
│  ├── ckpt.pt           # model parameters
│  ├── input_image.png   # visualization of input cryo-EM images
│  ├── pca-1.pdb         # sampled coarse-grained atomic structures along 1st PCA axis
│  ├── pca-2.pdb
│  ├── pca-3.pdb
│  ├── pred.pdb          # sampled structures at Kmeans cluster centers
│  ├── pred_gmm_image.png
│  └── z.npy             # the latent code of each particle
|                        # a matrix whose shape is num_of_particle x 8
├── yyyymmdd_hhmmss.log  # running logs
├── config.py            # a backup of the config file
└── train_atom.py        # a backup of the training script
```

#### S2: Training the density generator

In step 1, the atom generator assigns a latent code `z` to each particle image. In this step, we will drop the encoder and directly use the latent code as a representation of a partcile. You can execute the subsequent command to initiate the training of a density generator.

```shell
# change the xxx/z.npy path to the output of the above command
python train_density.py density_configs/1ake.py --cfg-options extra_input_data_attr.given_z=xxx/z.npy
```

Results will be saved to `work_dirs/density_xxxxx`, and each subdirectory has the name `epoch-number_step-number`. We choose the most recent directory as the final results.

```text
density_xxxxx/
├── 0004_0014470/          # evaluation results
│  ├── ckpt.pt             # model parameters
│  ├── vol_pca_1_000.mrc   # density sampled along the PCA axis, named by vol_pca_pca-axis_serial-number.mrc
│  ├── ...
│  ├── vol_pca_3_009.mrc
│  ├── z.npy
│  ├── z_pca_1.txt         # sampled z values along the 1st PCA axis
│  ├── z_pca_2.txt
│  └── z_pca_3.txt
├── yyyymmdd_hhmmss.log    # running logs
├── config.py              # a backup of the config file
└── train_density.py       # a backup of the training script
```


## Reference
You may cite this software by:
```bibtex
@article {Li2023cryostar,
    author = {Yilai Li and Yi Zhou and Jing Yuan and Fei Ye and Quanquan Gu},
    title = {CryoSTAR: Leveraging Structural Prior and Constraints for Cryo-EM Heterogeneous Reconstruction},
    elocation-id = {2023.10.31.564872},
    year = {2023},
    doi = {10.1101/2023.10.31.564872},
    URL = {https://www.biorxiv.org/content/early/2023/11/02/2023.10.31.564872},
    eprint = {https://www.biorxiv.org/content/early/2023/11/02/2023.10.31.564872.full.pdf},
    journal = {bioRxiv}
}
```
