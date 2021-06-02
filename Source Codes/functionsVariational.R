getParamsVB = function(n,y,invOmZ,tolerance = 1e-2, maxIter = 1e4) {

  ######################################################
  # PRECOMPUTATION
  ######################################################

  # compute optimal sigma2

  sigma2 <- matrix(1/(diag(invOmZ)), ncol = 1)
  sigma <- sqrt(sigma2)

  # compute matrix to write the CAVI update in a vectorized form

  A <- -c(sigma2)*invOmZ
  diag(A) <- 0

  # other useful quantities needed for ELBO

  diagInvOmZ = diag(invOmZ)
  coeffMean_Z2 = diagInvOmZ-1/sigma2

  # initialization of variables

  meanZ = matrix(0,n,1)
  mean_Z2 = matrix(0,n,1)
  mu = matrix(0,n,1)
  elbo = -1
  diff = 1
  nIter=0

  ######################################################
  # CAVI ALGORITHM
  ######################################################

  

  while(diff > tolerance & nIter < maxIter) {

    elboOld = elbo
    sumLogPhi = 0

 
    for(i in 1:n) {

      mu[i] = A[i,]%*%meanZ

      # compute first (needed for algorithm) and second (needed for ELBO) moments

      musiRatio = mu[i]/sigma[i]
      phiPhiRatio = dnorm(musiRatio)/pnorm((2*y[i]-1)*musiRatio)
      meanZ[i] = mu[i] + (2*y[i]-1)*sigma[i]*phiPhiRatio
      mean_Z2[i] = mu[i]^2+sigma2[i]+(2*y[i]-1)*mu[i]*sigma[i]*phiPhiRatio # needed for ELBO
      sumLogPhi = sumLogPhi + log(pnorm((2*y[i]-1)*musiRatio))

    }

    # computation of ELBO (up to an additive constant not depending on mu)

    elbo = -((t(meanZ)%*%invOmZ)%*%meanZ -
               sum((meanZ^2)*diagInvOmZ) +
               sum(mean_Z2*coeffMean_Z2))/2 -
          sum(meanZ*mu/sigma2) + sum((mu^2)/sigma2)/2 + sumLogPhi

    diff = abs(elbo-elboOld)
    nIter = nIter+1

    if(nIter%%100==0) {print(paste0("iter: ", nIter, ", ELBO: ", elbo))}

  }
  
  # get the optimal parameters of the normals before truncation, now that convergence has been reached

  mu = A%*%meanZ
 
  results = list(mu = mu, sigma = sigma, nIter = nIter)
  
  return(results)

}