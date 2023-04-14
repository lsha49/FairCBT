
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import math
import imblearn
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import OneSidedSelection
from sklearn import metrics
from deslib.util.instance_hardness import kdn_score
from numpy import where
import random
from math import log2, log, sqrt
from abroca import *
from scipy.spatial import distance
from statistics import stdev, mean

class DbtExample(object):
    def preprocessing(self, filename = "dataFile", useLabel = "Label"): 
        """import and split dataset for data balancing experiment.
        Parameters
        ----------
        filename: string, optional (default='dataFile')
            applied to import your data file
        useLabel: string, optional (default='Label')
            label column name in the csv file 
            e.g. is_content_related
        Returns
        -------
        """
        # @todo import data
        Corpus = pd.read_csv(filename, encoding='') 

        # @todo train test split
        # Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus.to_numpy(), Corpus[useLabel], test_size=, random_state=)        

        # @todo generate samples by Data Balancing Techniques
        # self.cbt()
        
        # @todo calculate KDN and hardness bias (hardness-bias)
        # self.calKDN()
                

    def cbt(self, X, Y, G):            
        """perform data balancing experiment.
        Parameters
        ----------
        X: array-like,
            Feature matrix with [n_samples, n_features].
        Y: array-like,
            Labels of given data [n_samples]
        G: array-like,
            Protected attributes of given data [n_samples]
        
        Returns
        -------
        X: array-like,
            Re-sampled feature matrix with [n_samples, n_features].
        Y: array-like,
            Re-sampled labels of given data [n_samples]
        G: array-like,
            Re-sampled protected attributes of given data [n_samples]
        """
        balanceMode = ''

        
        cbTech = SMOTE()
        # cbTech = ADASYN()
        # cbTech = BorderlineSMOTE()
        # cbTech = KMeansSMOTE()
        
        # cbTech = NearMiss()
        # cbTech = EditedNearestNeighbours()
        # cbTech = CondensedNearestNeighbour()
        # cbTech = TomekLinks()

        # cbTech = SMOTETomek()
        # cbTech = OneSidedSelection()
        # cbTech = Pipeline(steps=[('o', overSamplingTech), ('u', underSamplingTech)])
            
        if balanceMode == 'C': # class oversample
            X, Y = cbTech.fit_resample(X, Y)
            
        if balanceMode == 'CU': # class undersample
            X, Y = cbTech.fit_resample(X, Y) 
            
        if balanceMode == 'OV': # class + demo over sample
            GY = G.astype(str) + Y.astype(str)
            X, GY = cbTech.fit_resample(X, GY)  
            G = GY.str[0].astype(int); Y = GY.str[1].astype(int)

        if balanceMode == 'UN': # class + demo under sample
            GY = G.astype(str) + Y.astype(str)
            X, GY = cbTech.fit_resample(X, GY)  
            G = self.GY.str[0].astype(int); Y = self.GY.str[1].astype(int)

        return X, Y, G


    def calKDN(self, features, labels,  List00Index, List01Index, List10Index, List11Index): 
        """calculate Hardness-bias by KDN distribution.
        Parameters
        ----------
        features: array-like,
            Feature matrix with [n_samples, n_features].
        labels: array-like,
            Labels of given data [n_samples]
        List00Index: array-like,
            indices of (protected attribute=0, label=0) in the feature and label set
        List01Index: array-like,
            indices of (protected attribute=0, label=1) in the feature and label set
        List10Index: array-like,
            indices of (protected attribute=1, label=0) in the feature and label set
        List11Index: array-like,
            indices of (protected attribute=1, label=1) in the feature and label set
        Returns
        -------
        avg_kl_pq_01: float,
            Hardness bias score.
        """
        kdnResult = kdn_score(features, labels, 5)

        kdn_counts_00 = dict(Counter(kdnResult[0][List00Index]))
        kdn_counts_01 = dict(Counter(kdnResult[0][List01Index]))
        kdn_counts_10 = dict(Counter(kdnResult[0][List10Index]))
        kdn_counts_11 = dict(Counter(kdnResult[0][List11Index]))
        kdn_distribution_00 = []
        kdn_distribution_01 = []
        kdn_distribution_10 = []
        kdn_distribution_11 = []

        for i in range(6):
            if i*0.2 in kdn_counts_00.keys():
                kdn_distribution_00.append(kdn_counts_00[i*0.2]/kdnResult[0][List00Index].shape[0])
            else:
                kdn_distribution_00.append(0)
            
            if i*0.2 in kdn_counts_01.keys():
                kdn_distribution_01.append(kdn_counts_01[i*0.2]/kdnResult[0][List01Index].shape[0])
            else:
                kdn_distribution_01.append(0)
            
            if i*0.2 in kdn_counts_10.keys():
                kdn_distribution_10.append(kdn_counts_10[i*0.2]/kdnResult[0][List10Index].shape[0])
            else:
                kdn_distribution_10.append(0)
            
            if i*0.2 in kdn_counts_11.keys():
                kdn_distribution_11.append(kdn_counts_11[i*0.2]/kdnResult[0][List11Index].shape[0])
            else:
                kdn_distribution_11.append(0)
 
        kl_pq0 = distance.jensenshannon(kdn_distribution_00, kdn_distribution_10)
        kl_pq1 = distance.jensenshannon(kdn_distribution_01, kdn_distribution_11)

        return (kl_pq0 + kl_pq1)/2

    def getDfToComputeAbroca(self, predicted, predictionProb, Test_Y, Test_G):
        """prepare dataframe for ABROCA related calculations.
        Parameters
        ----------
        predicted: array-like,
            model predictions of given test data [n_samples]
        predictionProb: array-like,
            model prediction probability of given test data [n_samples, n_labels] or [n_samples]
        Test_Y: array-like,
            test set label of given test data [n_samples]
        Test_G: array-like,
            test set protected attribute of given test data [n_samples]
        Returns
        -------
        predictionDataframe: array-like,
            ABROCA dataframe containing prediction label and demographic info.
        """
        predictionDataframe = pd.DataFrame(predicted, columns = ['predicted'])
        predictionDataframe['prob_1'] = pd.DataFrame(predictionProb)[1]
        predictionDataframe['label'] = Test_Y.tolist()
        predictionDataframe['gender'] = Test_G.astype(str)
        return predictionDataframe

    def computeAbroca(self, abrocaDf):
        """perform ABROCA calculation.
        Parameters
        ----------
        abrocaDf: array-like,
            prediction dataframe built by getDfToComputeAbroca().
        Returns
        -------
        slice: array-like,
            contains abroca calculation result 
            see spec and parameter values of compute_abroca in https://pypi.org/project/abroca/
        """
        slice = compute_abroca(abrocaDf, 
                            pred_col = '' , 
                            label_col = '', 
                            protected_attr_col = '',
                            majority_protected_attr_val = '',
                            compare_type = '',
                            n_grid = 10000,
                            plot_slices = False)
        return slice
