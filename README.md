# FairEdu: Balancing Biased Data for Fair Modeling in Education
An python repository to perform educational dataset balancing applied in submitted paper in @todo. 

Download this repository with `git clone` or equivalent.

```bash
git clone @repo
```

## Requirements  
* Python 3.8  
* Tensorflow > 1.5
* tensorflow-estimator 2.7.0
* tensorflow-macos 2.7.0
* tensorflow-metal 0.3.0
* Sklearn > 0.19.0  


## Dataset Balancing, Hardness Bias and Fairness implementation 
We detail below how to implement dataset balancing, hard-bias and fairness evaluation. See example code in ```data_balancing/DbtExample.py```

### Dataset Balancing
The dataset balancing is implemented by ```DbtExample.py```, function ```cbt(self, X, Y, G):```, where X is input features, Y is input label and G is input demographics. Then to re-sample with the traditional Class Balancing Strategy, simply apply SMOTE from ```imblearn.over_sampling ```  package: ``` SMOTE().fit_resample(X, Y) ```, or any other class balancing techqniues such as BorderlineSMOTE, NearMiss. To re-sample with demographics, simply add demographic into class parameter and treating the balancing as a multi-class balancing problem, i.e.,  ``` GY = G.astype(str) + Y.astype(str) ``` and then ``` X, GY = SMOTE().fit_resample(X, GY) ```.

### Hardness Bias Evaluation
After generating samples, we evaluate the kDN distribution by ``` distance.jensenshannon ```
The H-bias can be calculated by ``` calKDN ``` function in the ``` DbtExample.py ```.

<!-- and selected the lowest H-bias samples.  -->


### Fairness Evaluation 
We applied ``` abroca ``` package in [ABROCA](https://pypi.org/project/abroca/). A sample calculation of ABROCA: 
```
slice = compute_abroca(abrocaDf, 
        pred_col = 'prob_1' , 
        label_col = 'label', 
        protected_attr_col = 'gender',
        majority_protected_attr_val = '2',
        compare_type = 'binary', # binary, overall, etc...
        n_grid = 10000,
        plot_slices = False)
```



## Datasets & Model implementation detail
We describe below the detailed dataset and model implementation.

### Forum Post
The forum dataset is included as [Moodle Forum de-identified Embeddings](https://github.com/). The CNN-LSTM model was implemented using a tensorflow repo modified from [here](https://github.com/zackhy/TextClassification), and the BERT embeddings of input text were generated using [Bert-as-service](https://github.com/hanxiao/bert-as-service). We set the input layer dimension as 768 with a sigmoid output layer. The L2 regularizer lambda set to 0.001. For the CNN network, 128 convolution filters with a width of 5 was used. For the LSTM network, 128 hidden states and 128 cell states was used. During training, one cycle policy was used with the batch size of 32, training epochs of 50, maximum learning rate of 2e-05 and dropout probability of 0.5. The shuffling was performed at the end of each epoch and with an early stopping mechanism. After each epoch, 10% of the training data was chosen at random as validation data and the best model was selected based on the validation error. 


<!-- ### Legal Assignment
The Caseote is included as [Legal Casenotes de-identified Embeddings](https://github.com/). we adopted the open-sourced huggingface[Legal BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) model. We fine-tune BERT Legal by a broader hyper-parameter search space procedure proposed in Legal BERT, with batch size set to 8 without a fixed maximum epochs to avoid model under-fitting and apply early stopping based on validation loss. We adopted the same setting of low learning and high drop-out rate of 1e-5 and 0.2 respectively as it was shown to improve regularization. -->

### KDDCUP Student Dropout 
The KDDCUP dataset is linked in [KDDCUP](https://data-mining.philippe-fournier-viger.com/the-kddcup-2015-dataset-download-link/). We implemeneted [this](https://github.com/wzfhaha/dropout_prediction) model by Feng. In line with the original implementation, the model had 32*32 deep layers with dropout ratio set to 0.5 and 32 convolution filters with a width of 8. During training, the learning rate was set to 0.001 with Adam optimizer, 10 epoch, 256 batch and logloss type. After each epoch, 10% of the training data was chosen at random to serve as validation data and the best model was selected based on validation error.

### Open University Student Performance
The Open University dataste is linked in [Open University](https://analyse.kmi.open.ac.uk/open_dataset#description). The source code is provided in [OUA](https://github.com/gogoladzetedo/Open_University_Analytics). For the traditional ML model, we applied the GridSearchCV package to find the best model hyperparameters to optimise the performance. 
```
A sample GridSearched model: 
rfc=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    max_depth=None, max_features=0.5, max_leaf_nodes=None,
    min_impurity_split=1e-07, min_samples_leaf=1,
    min_samples_split=4, min_weight_fraction_leaf=0.0,
    n_estimators=250, n_jobs=1, oob_score=False, random_state=None,
    verbose=0, warm_start=False)
```

<!-- ### ADM STEM career
The ADM dataset is linked in [ADM STEM career prediction](https://sites.google.com/view/assistmentsdatamining/data-mining-competition-2017?authuser=0). We implemented this [model](https://github.com/ckyeungac/ADM2017). For the traditionla ML model, we applied the GridSearchCV package to find the best model hyperparameters to optimise the performance. 
```
A sample GridSearched model: 
lrc = LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
    fit_intercept=True, intercept_scaling=1, max_iter=100,
    n_jobs=1, penalty='l1', random_state=None,
    solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
``` -->

