# Discriminator Ensemble

A library for creating and managing ensembles of discriminators for multi-adversarial learning. This plug-and-play component can replace the discriminator in existing single-adversary frameworks like GANs, improving robustness and performance across various domains including image, tabular, and spectral data synthesis [1]. 

## Installation

### Via Conda

```bash
conda env create -f environment.yml
conda activate discr_ensemble
```

### Adding to an Existing Project

Add the following to your `environment.yml`:

```yaml
dependencies:
  # ... your other dependencies
  - pip:
    # ... your other pip packages
    - git+https://github.com/ben-schindler/discr_ensemble@master
```

Or install directly with pip:

```bash
pip install git+https://github.com/ben-schindler/discr_ensemble@master
```

## Quick Start

A comprehensive Quick Start Notebook will be added soon.

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{schindler2025discriminator,
  title={When to Use Discriminator Ensembles? Cross-Domain Evidence of Multi-Adversarial Learning},
  author={Schindler, Benjamin and Mendikowski, Melle and Schmid, Thomas and Verboven, Sam},
  booktitle={IDEAL 2025},
  year={2025},
  doi={10.1007/978-3-032-10486-1_32}
}
```

**Reference:** [1] Schindler, B., Mendikowski, M., Schmid, T., & Verboven, S. (2025). When to Use Discriminator Ensembles? Cross-Domain Evidence of Multi-Adversarial Learning. *IDEAL 2025*. https://doi.org/10.1007/978-3-032-10486-1_32

This paper provides reliable benchmarks for image, tabular, and spectral data synthesis using discriminator ensembles.
