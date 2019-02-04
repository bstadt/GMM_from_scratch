import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal

class GMM:

    def __init__(self, k, min_var=1e-2, iterations=1000):
        self._k = k
        self._min_var = min_var
        self._iterations = iterations


    def _e_step(self, x, assignments):
        likelihoods = []
        for cur_cluster_idx in range(self._k):
            #get members of the cluster
            members = [elem\
                       for i, elem in enumerate(x)\
                       if assignments[i] == cur_cluster_idx]

            #calculate distribution params
            members = np.stack(members)

            #calculate distribution stats
            mean = np.mean(members, axis=0)
            cov = np.cov(members.T)

            #fix collapsing diagonals
            for i in range(cov.shape[0]):
              if cov[i, i] < self._min_var:
                cov[i, i] = min_var

            #initialize likelihood function
            f = multivariate_normal(mean, cov)

            likelihoods.append(f)
        return likelihoods


    def _m_step(self, x, likelihoods):

      #preallocate for speed
      assignments = [0 for elem in x]

      #maximize likelihood of each data point
      for i, elem in enumerate(x):
        cur_likelihoods = [f.pdf(elem)\
                          for f in likelihoods]

        cur_likelihoods = np.array(cur_likelihoods)
        clust_max_likelihood = np.argmax(cur_likelihoods)

        assignments[i] = clust_max_likelihood
      return assignments


    def fit_transform(self, x):

       #get random initial assignment
       assignments = [np.random.randint(0, self._k)\
                      for row in x]

       #training loop
       for iteration in tqdm(range(self._iterations)):
           likelihoods = self._e_step(x, assignments)
           assignments = self._m_step(x, likelihoods)

       return assignments
