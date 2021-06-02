Description
================
This tutorial contains guidelines and code to perform the analyses for the scenario `n = 250` in the simulation study of the article **scalable computation of predictive probabilities in probit models with Gaussian process priors**. In particular, you will find a detailed step-by-step guide and `R` code to **implement the methods under analysis** and to **reproduce the results in Table 1** for the scenario `n = 250` (Section 3). For implementation purposes, please **execute the code below considering the same order in which is presented**. 

Simulate the data 
================
To start the analysis open `R` and **set the working directory** where the files `tlrmvnratio.tar.gz` and `functionsVariational.R` are located. Once this has been done, **clean the workspace, and load the data along with useful** `R` **packages**.

``` r
library(mvtnorm)
library(fields)
library(RColorBrewer)
library(tlrmvnmvt)
library(TruncatedNormal)
library(truncnorm)
library(mnormt)
library(pROC)
library(rstan)
library(devtools)
```

As described in Section 3, to generate the Gaussian process we first create `10000` **known locations** on a 100 × 100 grid in the unit square.

``` r
m <- 100
n <- m * m
geom <- cbind(
  kronecker(seq(0, 1, length.out = m), rep(1, m)),
  kronecker(rep(1, m), seq(0, 1, length.out = m))
)
```

Once this has been done, we produce `100` locations on a 10 × 10 **grid in the unit square**, and another `100` **randomly distributed in the unit square**. These locations are required to assess **out-of-sample test performance** of the different methods under analysis.

```r
set.seed(123)
nUnknown <- 100
geomTmp <- geom[(geom[, 1] < 1 - 0.9 / (m - 1)) &
  (geom[, 2] < 1 - 0.9 / (m - 1)), ]

# random locations
geomUnknownRnd <- geomTmp[sample(1:(n - 2 * m + 1), nUnknown, F), ] +
  matrix(runif(nUnknown, 0.2 / m, 0.8 / m), nUnknown, 2)

# grid locations
geomUnknownGrid <- cbind(
  kronecker(seq(0.21, 0.39, length.out = 10), rep(1, 10)),
  kronecker(rep(1, 10), seq(0.41, 0.59, length.out = 10))
)
```

To generate the binary responses at the locations defined above, we first need to **simulate the realizations of the Gaussian process (GP) with mean function and covariance kernel defined in Section 3 of the article**. Let us first build a function (`grf_gen`) to perform this task.

``` r
grf_gen <- function(geom, alpha) {
  if (ncol(geom) != 2) {
    stop("grf_gen: only works under 2D geometry")
  }
  if (min(geom) < 0 || max(geom) > 1) {
    warning("grf_gen: geom is not in the unit square")
  }
  n <- nrow(geom)
  geom[, 1] <- geom[, 1] * alpha[1]
  geom[, 2] <- geom[, 2] * alpha[2]
  distM <- as.matrix(dist(geom))
  covM <- exp(-distM^2)
  return(as.vector(mvtnorm::rmvnorm(1, sigma = covM)))
}
```

Leveraging the above function, we can **simulate the GP realizations at both training and test locations**, and then **transform theses values into the probabilities using the probit link**, as described in Section 3 of the article.

``` r
set.seed(123)
alpha1 <- sqrt(30)
alpha2 <- sqrt(30)
alpha <- c(alpha1, alpha2)
prTtl <- pnorm(grf_gen(rbind(geom, geomUnknownRnd, geomUnknownGrid), alpha))
prUnknownRnd <- prTtl[(n + 1):(n + nUnknown)]
prUnknownGrid <- prTtl[(n + nUnknown + 1):length(prTtl)]
z <- matrix(prTtl, m, m)
```

To **visualize the output** as in Figure 2 of the article, execute the code below.

``` r
# random locations
image.plot(seq(0, 1, length.out = m), seq(0, 1, length.out = m), z, col = colorRampPalette(brewer.pal(11, "RdBu")[11:1])(30), xlab = expression(x[1]), ylab = expression(x[2]), cex.lab = 1.3, cex.axis = 1.3, legend.shrink = 0.8, legend.cex = 2.5, legend.width = 2, mgp = c(2, 1, 0))
points(x = geomUnknownRnd[, 1], y = geomUnknownRnd[, 2], col = "white", cex = 0.6, pch = 21, bg = "white")

# grid locations
image.plot(seq(0, 1, length.out = m), seq(0, 1, length.out = m), z, col = colorRampPalette(brewer.pal(11, "RdBu")[11:1])(30), xlab = expression(x[1]), ylab = expression(x[2]), cex.lab = 1.3, cex.axis = 1.3, legend.shrink = 0.8, legend.cex = 2.5, legend.width = 2, mgp = c(2, 1, 0))
points(x = geomUnknownGrid[, 1], y = geomUnknownGrid[, 2], col = "white", cex = 0.6, pch = 21, bg = "white")
```

We conclude the data simulation part, by **generating the binary responses `y` at the different training and test locations** from Bernoulli variables with probabilities simulated in the previous steps.

``` r
set.seed(123)
yTtl <- rbinom(n = n + 200, size = 1, prob = prTtl)
y <- yTtl[1:n]
```

Estimation of GP parameters
================
As discussed in the article, Gaussian processes are commonly indexed by a (generally low) number of unknown parameters **`α`**, which typically enter the covariance kernel function. A simple and practically feasible strategy to **estimate such parameters** is to **maximize the marginal likelihood via a grid search**. As discussed in Section 2, this requires to evaluate cumulative distribution functions of multivariate Gaussian. Here, we address this goal by using both the proposed **tile-low-rank strategy** (`TLR`) and also an alternative solution relying on the **minimax-tilting method** (`TN`) by [Botev (2017)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12162). To accomplish this task, let us first **define the two functions for computing the GP marginal likelihood** at multiple **`α`** values, under the two methods.

``` r
# TLR (leverages the `tlrmvnmvt` package)
mle_func_TLR <- function(alpha, geom, y) {
  n <- nrow(geom)
  xi <- rep(0, n)
  geom[, 1] <- geom[, 1] * alpha[1]
  geom[, 2] <- geom[, 2] * alpha[2]
  distM <- as.matrix(dist(geom))
  covM <- exp(-distM^2)
  xi <- (2 * y - 1) * xi
  covM <- outer(2 * y - 1, 2 * y - 1) * covM
  diag(covM) <- diag(covM) + 1
  ret <- tlrmvnmvt::pmvn(
    lower = rep(-Inf, n), upper = xi, sigma = covM, uselog2 = T, N = 998,
    algorithm = TLRQMC(m = round(sqrt(nrow(covM))), epsl = 1e-4)
  )
  cat("TLR: ", alpha, " -- ", ret, "\n")
  return(ret)
}

# TN (leverages the `TruncatedNormal` package)
mle_func_TN <- function(alpha, geom, y) {
  n <- nrow(geom)
  xi <- rep(0, n)
  geom[, 1] <- geom[, 1] * alpha[1]
  geom[, 2] <- geom[, 2] * alpha[2]
  distM <- as.matrix(dist(geom))
  covM <- exp(-distM^2)
  xi <- (2 * y - 1) * xi
  covM <- outer(2 * y - 1, 2 * y - 1) * covM
  diag(covM) <- diag(covM) + 1
  ret <- TruncatedNormal::pmvnorm(
    mu = rep(0, length(xi)), sigma = covM,
    log = TRUE, ub = xi
  )[[1]]
  cat("TN: ", alpha, " -- ", ret, "\n")
  return(ret)
}
```

We now **define the grid of **`α`** values** for point-search, and **create the training sample for the scenario `n = 250`** by selecting a 15 × 15 sub-grid of equally-spaced configurations (along with their associated probability parameters and simulated responses) from the `10000` known locations previously simulated; see Section 3 in the article for additional details.

``` r
# alpha grid
alphaVec <- sqrt(seq(15, 45, length.out = 10))
alphaPool <- cbind(kronecker(alphaVec, rep(1, length(alphaVec))), kronecker(rep(1, length(alphaVec)), alphaVec))

# indexes of the sub-grid for estimation and prediction in scenario n=250
# (change mSub to 25, 50 and 100 for testing the other scenarios in the simulations. 
# NOTE: the runtime for these additional scenarios will be much higher)
mSub <- 15
nSub <- mSub^2
idx1D <- round(seq(1, m, length.out = mSub))
idx2D <- c(kronecker(idx1D - 1, rep(m, mSub)) + idx1D)
```

Let us now **estimate `α`** under the two grid search strategies.

``` r
# TLR
set.seed(123)
lkVecTLR <- apply(alphaPool, 1, mle_func_TLR, geom = geom[idx2D, ], y = yTtl[idx2D])
alphaTLR <- alphaPool[which.max(lkVecTLR), ]

# TN
set.seed(123)
lkVecTN <- apply(alphaPool, 1, mle_func_TN, geom = geom[idx2D, ], y = yTtl[idx2D])
alphaTN <- alphaPool[which.max(lkVecTN), ]
```

Computation of predictive probabilities
================
Here, we **compute the predictive probabilities at the grid and random test locations** under the four methods discussed in Section 3 for the scenario with `n = 250` (i.e., the training data are those observed at the 15 × 15 grid specified previously). The methods evaluated, are:
  - The **TLR** strategy proposed in Section 2.1 of the article (requires `tlrmvnratio`)
  - The variational (**VB**) strategy proposed in Section 2.2 of the article (requires `functionsVariational.R`)
  - The **TN** strategy based on the calculation of the numerator and the denominator in eq (4) via [Botev (2017)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12162) (requires `TruncatedNormal`)
  - The Monte Carlo strategy which evaluates predictive probabilities via **STAN** samples from the exact GP posterior (requires `rstan`)

The **step-by-step code** to implement the above methods and produce the output in **Table 1** for the scenario `n = 250` is provided below.

**TLR**

```r
# install the tlrmvnratio package. *IMPORTANT*: installation requires "gfortran" for a successful execution
install.packages("tlrmvnratio.tar.gz", repos = NULL, method = "source")

# define the n = 250 training responses and produce the covariance matrix Sigma defined in Algorithm 1
ySub <- yTtl[idx2D]
geomSub <- geom[idx2D, , drop = F]
geomSub[, 1] <- geomSub[, 1] * alphaTLR[1]
geomSub[, 2] <- geomSub[, 2] * alphaTLR[2]
covM <- matrix(0, nSub + 1, nSub + 1)
covM[1:nSub, 1:nSub] <- exp(-(as.matrix(dist(geomSub)))^2)
covM[1:nSub, 1:nSub] <- outer(2 * ySub - 1, 2 * ySub - 1) * covM[1:nSub, 1:nSub]
diag(covM[1:nSub, 1:nSub]) <- diag(covM[1:nSub, 1:nSub]) + 1
covM[nSub + 1, nSub + 1] <- 2

# create the empty vectors of predictive probabilities at random and grid test locations  
set.seed(123)
predRnd <- rep(NA, nrow(geomUnknownRnd))
predGrid <- rep(NA, nrow(geomUnknownGrid))

# compute predictive probabilities at the 100 random locations via TLR and save runtime for first prediction
startTime <- Sys.time()
for (i in 1:nrow(geomUnknownRnd))
{
  geomTmp <- geomSub
  geomTmp[, 1] <- geomTmp[, 1] - geomUnknownRnd[i, 1] * alphaTLR[1]
  geomTmp[, 2] <- geomTmp[, 2] - geomUnknownRnd[i, 2] * alphaTLR[2]
  covM[nSub + 1, 1:nSub] <- (2 * ySub - 1) * exp(-rowSums(geomTmp^2))
  covM[1:nSub, nSub + 1] <- covM[nSub + 1, 1:nSub]
  predRnd[i] <- tlrmvnratio::mvn.ratio.tlr(
    upper = 0, sigma = covM, m = mSub,
    N = 998, epsl = 1e-4
  )[[1]]
  if(i == 1)
    endTime <- Sys.time()
}

# compute predictive probabilities at the 100 grid locations via TLR
for (i in 1:nrow(geomUnknownGrid))
{
  geomTmp <- geomSub
  geomTmp[, 1] <- geomTmp[, 1] - geomUnknownGrid[i, 1] * alphaTLR[1]
  geomTmp[, 2] <- geomTmp[, 2] - geomUnknownGrid[i, 2] * alphaTLR[2]
  covM[nSub + 1, 1:nSub] <- (2 * ySub - 1) * exp(-rowSums(geomTmp^2))
  covM[1:nSub, nSub + 1] <- covM[nSub + 1, 1:nSub]
  predGrid[i] <- tlrmvnratio::mvn.ratio.tlr(
    upper = 0, sigma = covM, m = mSub,
    N = 998, epsl = 1e-4
  )[[1]]
}

# compute and display the runtime and MSEs shown in Table 1 for TLR
timeCost <- as.numeric(difftime(endTime, startTime, units = "secs"))
MSERnd <- sum((prTtl[(n + 1):(n + 100)] - predRnd)^2) / 100
MSEGrid <- sum((prTtl[(n + 101):(n + 200)] - predGrid)^2) / 100
cat(
  "Time[seconds]:", timeCost, "\n",
  "MSE[random]", MSERnd, "\n",
  "MSE[grid]", MSEGrid, "\n"
)
```

**VB**

``` r
# load the source code for implementing the VB method
source("functionsVariational.R")

# define the GP covariance matrix Omega with alpha as estimated via TLR
geomSub <- geom[idx2D, , drop = F]
geomSub[, 1] <- geomSub[, 1] * alphaTLR[1]
geomSub[, 2] <- geomSub[, 2] * alphaTLR[2]
Omega <- exp(-(as.matrix(dist(geomSub,diag=TRUE,upper=TRUE)))^2)
# NOTE: we add a small nugget effect only for numerical reasons to avoid singularity
diag(Omega) <- diag(Omega)+1e-10
n_sub <- nrow(geomSub)

# pre-compute the inverse of Sigma_z and run the CAVI part in Algorithm 2
startTime <- Sys.time()
set.seed(123)
invOmZ <- pd.solve(Omega+diag(1,n_sub,n_sub))
paramsVB <- getParamsVB(n=n_sub,y=ySub,invOmZ=invOmZ,tolerance=1e-3,maxIter=1e4)

# sample the values from the optimal approximating univariate truncated normals required to compute eq. 13
nSample <- 20000
muTN <- paramsVB$mu
muTN[ySub==0] <- -muTN[ySub==0]
sigmaTN <- paramsVB$sigma
set.seed(123)
sampleTruncNorm <- matrix(rtruncnorm(n_sub*nSample, a = 0, b = Inf, mean = muTN, sd = sigmaTN), nrow = n_sub, ncol = nSample, byrow = F ) 
sampleTruncNorm[ySub==0,] = -sampleTruncNorm[ySub==0,] 

# pre-compute the inverse of Omega required to compute predictive probabilities at generic locations
invOmega <- pd.solve(Omega)
omega_new <- 1

# compute predictive probabilities at the 100 random locations via VB and save runtime for first prediction
predRnd <- rep(NA, nrow(geomUnknownRnd))
geomTmpRnd <- matrix(NA,nrow(geomUnknownRnd),2)
for (i in 1:nrow(geomUnknownRnd))
{
	geomTmpRnd[i, 1] <- geomUnknownRnd[i, 1] * alphaTLR[1]	
	geomTmpRnd[i, 2] <- geomUnknownRnd[i, 2] * alphaTLR[2]	
	omega_new_marg <- matrix(exp(-apply((t(geomSub)-geomTmpRnd[i,])^2,2,sum)),n_sub,1)
	H_x <- crossprod(omega_new_marg,invOmega)
	H_Sigma <- crossprod(omega_new_marg,invOmZ)
	pred_variance <- 1+omega_new-H_x%*%omega_new_marg+H_Sigma%*%t(H_x)
	pred_mean <- H_Sigma%*%sampleTruncNorm	
	predRnd[i] <- mean(pnorm(c(pred_mean),0,sd=sqrt(pred_variance)))
	if(i == 1)
    	endTime <- Sys.time()
}

# compute predictive probabilities at the 100 grid locations via VB
predGrid <- rep(NA, nrow(geomUnknownGrid))
geomTmpGrid <- matrix(NA,nrow(geomUnknownGrid),2)
for (i in 1:nrow(geomUnknownGrid))
{
	geomTmpGrid[i, 1] <- geomUnknownGrid[i, 1] * alphaTLR[1]	
	geomTmpGrid[i, 2] <- geomUnknownGrid[i, 2] * alphaTLR[2]	
	omega_new_marg <- matrix(exp(-apply((t(geomSub)-geomTmpGrid[i,])^2,2,sum)),n_sub,1)
	H_x <- crossprod(omega_new_marg,invOmega)
	H_Sigma <- crossprod(omega_new_marg,invOmZ)
	pred_variance <- 1+omega_new-H_x%*%omega_new_marg+H_Sigma%*%t(H_x)
	pred_mean <- H_Sigma%*%sampleTruncNorm	
	predGrid[i] <- mean(pnorm(c(pred_mean),0,sd=sqrt(pred_variance)))
}

# compute and display the runtime and MSEs shown in Table 1 for VB
timeCost <- as.numeric(difftime(endTime, startTime, units = "secs"))
MSERnd <- sum((prTtl[(n + 1):(n + 100)] - predRnd)^2) / 100
MSEGrid <- sum((prTtl[(n + 101):(n + 200)] - predGrid)^2) / 100
cat(
  "Time[seconds]:", timeCost, "\n",
  "MSE[random]", MSERnd, "\n",
  "MSE[grid]", MSEGrid, "\n"
)
```

**TN**

``` r
# produce the covariance matrix Sigma at the numerator of (3) using the alpha values estimated under TN
set.seed(123)
geomSub <- geom[idx2D, , drop = F]
geomSub[, 1] <- geomSub[, 1] * alphaTN[1]
geomSub[, 2] <- geomSub[, 2] * alphaTN[2]
covM <- matrix(0, nSub + 1, nSub + 1)
covM[1:nSub, 1:nSub] <- exp(-(as.matrix(dist(geomSub)))^2)
covM[1:nSub, 1:nSub] <- outer(2 * ySub - 1, 2 * ySub - 1) * covM[1:nSub, 1:nSub]
diag(covM[1:nSub, 1:nSub]) <- diag(covM[1:nSub, 1:nSub]) + 1
covM[nSub + 1, nSub + 1] <- 2

# create the empty vectors of predictive probabilities at random and grid test locations  
predRnd <- rep(NA, nrow(geomUnknownRnd))
predGrid <- rep(NA, nrow(geomUnknownGrid))

# compute the denominator of (3), which is common to all predictions, via TN  
startTime <- Sys.time()
denormTN <- TruncatedNormal::pmvnorm(
  mu = rep(0, nSub),
  sigma = covM[1:nSub, 1:nSub],
  lb = rep(-Inf, nSub),
  ub = rep(0, nSub)
)[[1]]

# compute predictive probabilities at the 100 random locations via TN and save runtime for first prediction
for (i in 1:nrow(geomUnknownRnd))
{
  geomTmp <- geomSub
  geomTmp[, 1] <- geomTmp[, 1] - geomUnknownRnd[i, 1] * alphaTN[1]
  geomTmp[, 2] <- geomTmp[, 2] - geomUnknownRnd[i, 2] * alphaTN[2]
  covM[nSub + 1, 1:nSub] <- (2 * ySub - 1) * exp(-rowSums(geomTmp^2))
  covM[1:nSub, nSub + 1] <- covM[nSub + 1, 1:nSub]
  predRnd[i] <- TruncatedNormal::pmvnorm(
    mu = rep(0, nSub + 1),
    sigma = covM,
    lb = rep(-Inf, nSub + 1),
    ub = rep(0, nSub + 1)
  )[[1]] / denormTN
  if(i == 1)
    endTime <- Sys.time()
}

# compute predictive probabilities at the 100 grid locations via TN
for (i in 1:nrow(geomUnknownGrid))
{
  geomTmp <- geomSub
  geomTmp[, 1] <- geomTmp[, 1] - geomUnknownGrid[i, 1] * alphaTN[1]
  geomTmp[, 2] <- geomTmp[, 2] - geomUnknownGrid[i, 2] * alphaTN[2]
  covM[nSub + 1, 1:nSub] <- (2 * ySub - 1) * exp(-rowSums(geomTmp^2))
  covM[1:nSub, nSub + 1] <- covM[nSub + 1, 1:nSub]
  predGrid[i] <- TruncatedNormal::pmvnorm(
    mu = rep(0, nSub + 1),
    sigma = covM,
    lb = rep(-Inf, nSub + 1),
    ub = rep(0, nSub + 1)
  )[[1]] / denormTN
}

# compute and display the runtime and MSEs shown in Table 1 for TN
timeCost <- as.numeric(difftime(endTime, startTime, units = "secs"))
MSERnd <- sum((prTtl[(n + 1):(n + 100)] - predRnd)^2) / 100
MSEGrid <- sum((prTtl[(n + 101):(n + 200)] - predGrid)^2) / 100
cat(
  "Time[seconds]:", timeCost, "\n",
  "MSE[random]", MSERnd, "\n",
  "MSE[grid]", MSEGrid, "\n"
)
```

**STAN**

``` r
# define the GP probit model using STAN 
probmodel <- "
data {
  int<lower=0> N;
  int Y[N];
  vector[N] mu;
  matrix[N, N] Omega;
}
parameters {
  vector[N] g;
}
model {
  g ~ multi_normal(mu, Omega);
  for(n in 1:N)
    Y[n] ~ bernoulli(Phi(g[n]));
}
"

# define the mean and covariance matrix of the GP prior using the true values of alpha 
mu <- rep(0, nSub)
geomSub <- geom[idx2D, , drop = F]
geomSub[, 1] <- geomSub[, 1] * alpha[1]
geomSub[, 2] <- geomSub[, 2] * alpha[2]
covM <- exp(-(as.matrix(dist(geomSub)))^2)
diag(covM) <- diag(covM) + 1e-2

# run MCMC algorithm based on STAN implementation to sample from the GP posterior 
startTime <- Sys.time()
parmsSTAN <- list(N = length(ySub), Y = ySub, mu = mu, Omega = covM)
HMCSamples <- stan(
  model_code = probmodel, data = parmsSTAN,
  iter = 20000, warmup = 10000, chains = 1, init = "0",
  algorithm = "NUTS", seed = 123
)
timeHMC <- get_elapsed_time(HMCSamples)[1] + get_elapsed_time(HMCSamples)[2]
gHMC <- t(extract(HMCSamples)$g)

# load the "geoR" library to compute predictive probabilities from STAN samples via ordinary kriging
# *IMPORTANT*: This library requires XCode, which conflicts with other libraries used for TLR, VB and TN. 
#              Hence, load this library only after TLR, VB and TN have been implementeed.
library(geoR)

geomUnknown <- rbind(geomUnknownRnd, geomUnknownGrid)
# find the time for prediction at one location
gUnknown <- apply(
  X = gHMC, MARGIN = 2,
  FUN = function(v) {
    krige.conv(
      data = v,
      coords = geom[idx2D, , drop = F],
      locations = geomUnknown[1, ],
      krige = krige.control(
        cov.model = "powered.exponential",
        cov.pars = c(1, 1 / alpha[1]),
        kappa = 2, nugget = 0
      )
    )$predict
  }
)
endTime <- Sys.time()
gUnknown <- apply(
  X = gHMC, MARGIN = 2,
  FUN = function(v) {
    krige.conv(
      data = v,
      coords = geom[idx2D, , drop = F],
      locations = geomUnknown,
      krige = krige.control(
        cov.model = "powered.exponential",
        cov.pars = c(1, 1 / alpha[1]),
        kappa = 2, nugget = 0
      )
    )$predict
  }
)
# endTime <- Sys.time()
timeCost <- as.numeric(difftime(endTime, startTime, units = "secs"))
pred <- rowMeans(pnorm(gUnknown))
predRnd <- pred[1:100]
predGrid <- pred[101:200]
MSERnd <- sum((prTtl[(n + 1):(n + 100)] - predRnd)^2) / 100
MSEGrid <- sum((prTtl[(n + 101):(n + 200)] - predGrid)^2) / 100
cat(
  "Time[seconds]:", timeCost, "\n",
  "MSE[random]", MSERnd, "\n",
  "MSE[grid]", MSEGrid, "\n"
  )
```
