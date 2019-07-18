# SYSNet
## Neural Network modeling of imaging systematics in Galaxy Surveys

In this project we develop a Feed Forward Neural Network to model the relationship between the imaging maps eg. Stellar density and the observed galaxy density field, in order to mitigate for the systematic effects and to make a robust galaxy clustering measurements. The hyper parameters of the neural network is trained with 5-fold Validation technique. The cost function is Mean Squared Error and a L2 regularization term, and the optimization algorithm is Adaptive Moment (ADAM). We perform backward feature elimination to remove redundant information from the input layer.

The strengths and weaknesses of the method is investigated using three versions of simulated datasets:
* Mocks without any contamination: to assess how much bias and variance is introduced in absence of any systematic effects
* Mocks with contamination : a multivariate function of 10 map


We use the Ablation technique to reduce the dimensionality of the problem, and minimize the likelihood of regressing out the cosmological clustering signal.
