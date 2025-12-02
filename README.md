# Bayesian inference on Calabi-Yau moduli spaces and the axiverse
 
This repository accompanies the paper [ArXiv:2512.00144](https://arxiv.org/abs/2512.00144) and provides the computational framework developed therein. It contains code and data to explore Bayesian inference on Calabi-Yau (CY) Kähler moduli spaces and to study axion phenomenology (“the axiverse”) using both classical MCMC and normalising flows. It includes scripts to sample the Weil-Petersson (WP) prior on moduli space under stretched Kähler cone constraints, build theory-informed priors for axion parameters $(m_a, f_a)$, and reproduce key diagnostic plots.





## Summary of results

In [ArXiv:2512.00144](https://arxiv.org/abs/2512.00144), we introduce a suite of tools for performing Bayesian inference on the moduli space of Calabi-Yau (CY) manifolds. The code implements efficient sampling from the invariant Weil-Petersson (WP) measure using both Markov Chain Monte Carlo (MCMC) and normalising flows on Kähler moduli spaces with dimensions up to Hodge number $h^{1,1} = 30$.

Our analysis explores the spectrum of Calabi-Yau volumes and the behaviour of divisors when the WP measure is restricted by physically motivated constraints such as the stretched Kähler cone conditions. Building on this foundation, we construct a theory-informed prior on axion masses and decay constants $(m_a, f_a)$, marginalised over the WP measure for all Calabi-Yau threefolds obtainable from the Kreuzer-Skarke database with $h^{1,1} \leq 5$ in the orientifold-even sector ($h^{1,1} = h^{1,1}_+$).

The framework further demonstrates how to impose likelihoods informed by axion physics. In particular, the detection of a relatively heavy QCD axion—such as might be observed by ADMX—can rule out Calabi–Yau geometries with large $h^{1,1}$, while providing detailed geometric and topological information at smaller $h^{1,1} $.

Finally, we implement a forward model combining cosmological likelihoods from the CMB and the Lyman-α forest. The resulting posterior distribution identifies the regions of moduli space most favoured by a resolution of the CMB–Lyα tension through the presence of an ultralight axion making up a percent-level fraction of the dark matter density.



<br>


<p align="center">
  <img src="/images/plotex2.png" width="800">
</p>


<br>

<br>


## Working with this repository


### Installation

```
# From the repository root
conda env create -f environment.yml
conda activate cy-axiverse
```

> [!IMPORTANT]
> The code makes use of basic functions from [CYTools](https://cy.tools). For help with the installation, please check out the [documentation](https://cy.tools/docs/getting-started/) or reach out to us.



### Reading the data

We summarize the data structure for the files in [`data`](./data/) and the python scripts that can be found in [`code`](./code/). We also provide a small demo on how to run MCMC by interfacing with [CYTools](https://cy.tools) in the notebook [`mcmc.ipynb`](/notebooks/mcmc.ipynb).


## Contact 

For questions or feedback, please get in touch: <as3475@cornell.edu> or <a.schachner@lmu.de>.


## Reference

If you use this database for future publications, please cite

```
@article{Jain:2025vfh,
    author = "Jain, Mudit and Sheridan, Elijah and Marsh, David J. E. and Heyes, Elli and Rogers, Keir K. and Schachner, Andreas",
    title = "{Bayesian inference on Calabi--Yau moduli spaces and the axiverse: experimental data meets string theory}",
    eprint = "2512.00144",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    month = "11",
    year = "2025"
}

```

[![DOI](https://zenodo.org/badge/???.svg)](https://doi.org/???)
