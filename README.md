# Induced Surface and Curvature Tensions Equation of State of hadron matter

Source code for the results published in the following papers:

* N. S. Yakovenko, K. A. Bugaev, L. V. Bravina and E. E. Zabrodin\
**The concept of induced surface and curvature tensions for EoS of hard discs and hard spheres**\
Published in: Eur. Phys. J. ST 229 (2020) 22-23, 3445-3467\
e-Print: [1910.04889](https://arxiv.org/abs/1910.04889) [nucl-th]\
DOI: [10.1140/epjst/e2020-000036-3](https://doi.org/10.1140/epjst/e2020-000036-3)

* N. S. Yakovenko, K. A. Bugaev, L. V. Bravina and E. E. Zabrodin\
**Induced surface and curvature tensions equation of state of hard spheres and its virial coefficients**\
Published in: Int. J. Mod. Phys. E 29 (2020) 11, 2040010\
e-Print: [2004.08693](https://arxiv.org/abs/2004.08693) [nucl-th]\
DOI: [10.1142/S0218301320400108](https://doi.org/10.1142/S0218301320400108)

* K. A. Bugaev, N. S. Yakovenko, P. V. Oliinyk, E. G. Nikonov, D. B. Blaschke, L. V. Bravina and E. E. Zabrodin\
**Induced Surface and Curvature Tensions Equation of State of Hadrons with Relativistic Excluded Volumes and its Relation to Morphological Thermodynamics**\
Published in: Phys. Scr. 96 125302 (2021)\
e-Print: [2104.06528](https://arxiv.org/abs/2104.06528) [nucl-th]\
DOI: [10.1088/1402-4896/ac183e](https://doi.org/10.1088/1402-4896/ac183e)

## Overview

The repository contains an implementation of the Induced Surface and Curvature Tensions (ISCT) equation of state (EoS) for classical and relativistic cases.

You can find inside:

* analytical expressions for virial expansions of pressure, surface, and curvature tensions and their comparison with exact calculations. Comparison with Carnahan-Starling and Mansoori-Carnahan-Starling-Leland equations of state included.\
*N. F. Carnahan and K. E. Starling, J. Chem. Phys. 51, 635 (1969)*\
*G. A. Mansoori, N. F. Carnahan, K. E. Starling, T. Leland, J. Chem. Phys. 54, 1523, (1971)*

* calculations of exact classical and approximate relativistic excluded volumes of pairs of hadrons. Effective excluded volume for the relativistic case is also present.

* calculations of the thermodynamic speed of sound by definition.
  
* calculations of cumulants and their comparison with\
  *Sorensen et al. Speed of sound and baryon cumulants in heavy-ion collisions* [arXiv:2103.07365](https://arxiv.org/abs/2103.07365v1) [nucl-th]

* comparison of the ISCT EoS with morphological thermodynamics properties according to\
*Hendrik Hansen-Goos and Roland Roth 2006 J. Phys.: Condens. Matter 18 8413* [arXiv:0606658](https://arxiv.org/abs/cond-mat/0606658v2) [cond-mat.soft]

* search of the mixed phase of light hadrons according to\
*Bugaev, K.A., Emaus, R., Sagun, V.V. et al. Phys. Part. Nuclei Lett. 15, 210–224 (2018)* [arXiv:1709.05419](https://arxiv.org/abs/1709.05419) [hep-ph]\
*Bugaev, K.A., Ivanytskyi, A.I., Oliinychenko, D.R. et al. Phys. Part. Nuclei Lett. 12, 238–245 (2015)* [	arXiv:1405.3575](https://arxiv.org/abs/1405.3575) [hep-ph]


## Description

* [`main.py`](main.py) contains a base class for the thermodynamical equations of state and provides core functionality for calculating general properties such as particle and entropy densities, energy, speed of sound, etc.

* [`eos`](eos) folder contains various equations of state of hadron matter including basic van der Waals EoS, MIT Bag model, ideal gas for both classical and relativistic cases, etc. Also contains an implementation of the ISCT EoS.

* [`scripts`](scripts) folder contains calculation scripts and their `SLURM` schedulers for various tasks. Can serve as an example of the usage.

* various notebooks contain non-computationally demanding calculations and visualizations of some obtained results.\
*(note: images might not be available for preview in the notebook directly on `github` but should be visible locally in the cloned repository)*


## Citation
Please cite if using in your work

```bibtex
@article{Yakovenko:2019dby,
    author = "Yakovenko, Nazar S. and Bugaev, Kyrill A. and Bravina, Larissa V. and Zabrodin, Eugene E.",
    title = "{The concept of induced surface and curvature tensions for EoS of hard discs and hard spheres}",
    eprint = "1910.04889",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    doi = "10.1140/epjst/e2020-000036-3",
    journal = "Eur. Phys. J. ST",
    volume = "229",
    number = "22-23",
    pages = "3445--3467",
    year = "2020"
}

@article{Yakovenko:2020swu,
    author = "Yakovenko, Nazar S. and Bugaev, Kyrill A. and Bravina, Larissa V. and Zabrodin, Evgeny E.",
    title = "{Induced surface and curvature tensions equation of state of hard spheres and its virial coefficients}",
    eprint = "2004.08693",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    doi = "10.1142/S0218301320400108",
    journal = "Int. J. Mod. Phys. E",
    volume = "29",
    number = "11",
    pages = "2040010",
    year = "2020"
}

@article{Bugaev:2021mof,
    author = "Bugaev, K. A. and Yakovenko, N. S. and Oliinyk, P. V. and Nikonov, E. G. and Blaschke, D. B. and Bravina, L. V. and Zabrodin, E. E.",
    title = "{Induced Surface and Curvature Tensions Equation of State of Hadrons with Relativistic Excluded Volumes and its Relation to Morphological Thermodynamics}",
    eprint = "2104.06528",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    doi = "10.1088/1402-4896/ac183e",
    month = "4",
    year = "2021"
}
```
