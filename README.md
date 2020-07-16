# SYSNet
## Neural Network modeling of imaging systematics in Galaxy Surveys
The repository hosts the pipeline that is presented in https://arxiv.org/abs/1907.11355. In this project we develop, validate, and present a Feed Forward Neural Network to model the relationship between the imaging maps eg. Stellar density and the observed galaxy density field, in order to mitigate the systematic effects and to make a robust galaxy clustering measurements. The cost function is Mean Squared Error and a L2 regularization term, and the optimization algorithm is Adaptive Moment (ADAM). We employ the 5-fold corss validation to perform feature selection, and the hyper parameter tuning. We find that the feature selection process is essential to protect the cosmological information from the regression. 

The strengths and weaknesses of the method is investigated using two sets of simulated datasets:

* Mocks without any contamination: to assess how much bias and variance is introduced in the absence of any systematic effects
* Mocks with contamination : a multivariate function of 10 maps to evaluate the performance of the feature elimination and the neural network.

The software is still under development and documentation. If you need to use this code in research please contact me at mr095415@ohio.edu:
```
@ARTICLE{2020MNRAS.495.1613R,
       author = {{Rezaie}, Mehdi and {Seo}, Hee-Jong and {Ross}, Ashley J. and
         {Bunescu}, Razvan C.},
        title = "{Improving galaxy clustering measurements with deep learning: analysis of the DECaLS DR7 data}",
      journal = {\mnras},
     keywords = {editorials, notices, miscellaneous, catalogues, surveys, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Data Analysis, Statistics and Probability},
         year = 2020,
        month = may,
       volume = {495},
       number = {2},
        pages = {1613-1640},
          doi = {10.1093/mnras/staa1231},
archivePrefix = {arXiv},
       eprint = {1907.11355},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.1613R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"
```

April 2, 2020: new systematics mitigation and clustering analysis codes are now in https://github.com/mehdirezaie/LSSutils/           
July 16, 2020: new systematics mitigation pipeline based on Pytorch in https://github.com/mehdirezaie/sysnetdev/
