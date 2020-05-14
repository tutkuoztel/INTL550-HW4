import numpy as np

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)




import pandas as pd
import os
os.chdir('C:/Users/toztel17/Desktop/hw4')
tt = pd.read_csv('immSurvey.csv')
tt.head()

alphas = tt.stanMeansNewSysPooled
sample = tt.textToSend

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
X

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

#down-weighting frequent words; term frequency–inverse document frequency (TF–IDF), which weights the word counts by a measure of how often they appear in the documents
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, alphas,
random_state=1)
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

# Compute posterior predictive mean and covariance
mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

#test correlation between test and mus
np.corrcoef(ytest, mu_s) # 0.6832 

#how might we improve this?

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),
                                  token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()


X_2 = bigram_vectorizer.fit_transform(sample)#.toarray()
#pd.DataFrame(X_2.toarray(), columns=vec.get_feature_names())

X2train, X2test, y2train, y2test = train_test_split(X_2, alphas,
random_state=1)


gpr.fit(X2train.toarray(), ytrain)
mu_s2, cov_s2 = gpr.predict(X2test.toarray(), return_cov=True)

#test correlation between test and mus
np.corrcoef(y2test, mu_s2) # 0.4554



