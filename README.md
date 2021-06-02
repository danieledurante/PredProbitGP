# PredProbitGP: Scalable computation of predictive probabilities in probit models with GP priors

This repository is associated with the article [**scalable computation of predictive probabilities in probit models with Gaussian process priors**](https://arxiv.org/abs/2009.01471) and aims at providing detailed materials and codes to implement the methods presented in the article.

The documentation is organized in two main parts described below.  

- [`Source Codes`](https://github.com/danieledurante/PredProbitGP/tree/main/Data%20and%20Codes).  It contains commented source `R` functions to implement the methods presented in the article. More specifically, the library `tlrmvnratio.tar.gz` comprises all the routines related to **Algorithm 1** the article (see Section 2.1), whereas `functionsVariational.R` contains the source codes to implement **Algorithm 2** in the article (see Section 2.2).

- [`Tutorial.md`](https://github.com/danieledurante/PredProbitGP/blob/main/Tutorial.md). It contains a detailed tutorial on how to implement the methods and algorithms presented in the article. To accomplish this goal, we mainly focus on reproducing step-by-step the scenario `n=250` of the simulation study in the article (see Section 3).

The analyses are performed with an **iMac (macOS Sierra, version 10.12.6),**, using a `R` version **3.6.1**. 
