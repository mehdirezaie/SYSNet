# SYSNet
## Neural Network modeling of imaging systematics in Galaxy Surveys
The repository hosts the pipeline that is presented in https://arxiv.org/abs/1907.11355. In this project we develop, validate, and present a Feed Forward Neural Network to model the relationship between the imaging maps eg. Stellar density and the observed galaxy density field, in order to mitigate the systematic effects and to make a robust galaxy clustering measurements. The cost function is Mean Squared Error and a L2 regularization term, and the optimization algorithm is Adaptive Moment (ADAM). We employ the 5-fold corss validation to perform feature selection, and the hyper parameter tuning. We find that the feature selection process is essential to protect the cosmological information from the regression. 

The strengths and weaknesses of the method is investigated using two sets of simulated datasets:

* Mocks without any contamination: to assess how much bias and variance is introduced in the absence of any systematic effects
* Mocks with contamination : a multivariate function of 10 maps to evaluate the performance of the feature elimination and the neural network.

If you use this code in research that results in publications, please cite the following paper:
```
Rezaie, Mehdi and Seo, Hee-Jong and Ross, Ashley J. and Bunescu, Razvan C., 2019, MNRAS, submitted
"Improving Galaxy Clustering Measurements with Deep Learning: analysis of the DECaLS DR7 data"
```
