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

A Quick Start Notebook will be added soon.

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{schindler2025discriminator,
author="Schindler, Benjamin
and Mendikowski, Melle
and Schmid, Thomas
and Verboven, Sam",
title="When to Use Discriminator Ensembles? Cross-Domain Evidence of Multi-Adversarial Learning",
booktitle="Intelligent Data Engineering and Automated Learning -- IDEAL 2025",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="345--357",
abstract="Multiple adversary networks represent a recurring extension in generative adversarial learning, leveraging discriminator ensembles for improved model performance and training stability. However, determining when discriminator ensembles provide substantial benefits remains unclear. We systematically apply this approach across image, tabular, and spectral data, demonstrating significant improvements with minimal implementation complexity - notably reducing Frechet Inception Distance scores by up to 40.6{\%} for image generation. Our comprehensive study, spanning 11 distinct use-cases, pioneers the underexplored realm of multi-adversarial techniques for tabular and spectral data synthesis. We identify gradient orthogonality within discriminator ensembles as the primary driver of performance gains. Our findings provide practical guidance on when to implement multi-adversarial approaches, complemented by gradient-based measures for monitoring ensemble dynamics and quantifiable performance expectations across various architectures.",
isbn="978-3-032-10486-1"
}

```

**Reference:** [1] Schindler, B., Mendikowski, M., Schmid, T., & Verboven, S. (2026). When to Use Discriminator Ensembles? Cross-Domain Evidence of Multi-Adversarial Learning. *IDEAL 2025*. https://doi.org/10.1007/978-3-032-10486-1_32

This paper provides reliable benchmarks for image, tabular, and spectral data synthesis using discriminator ensembles.
