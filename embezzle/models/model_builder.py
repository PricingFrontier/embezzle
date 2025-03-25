"""
GLM model builder module for Embezzle

This module provides the core functionality for building and analyzing
Generalized Linear Models (GLMs) using statsmodels backend.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import os
import logging
from statsmodels.genmod.families import (
    Gaussian, Binomial, Poisson, Gamma, InverseGaussian, 
    NegativeBinomial, Tweedie
)
from statsmodels.genmod.families.links import (
    Identity, Log, Logit, Power, Probit, CLogLog, NegativeBinomial as NBLink
)

# Configure logging
logger = logging.getLogger(__name__)

class GLMBuilder:
    """
    A class for building and analyzing Generalized Linear Models (GLMs).
    
    This class provides methods for creating, fitting, and analyzing GLMs
    using the statsmodels backend.
    """
    
    def __init__(self):
        """Initialize a new GLM Builder."""
        self.model = None
        self.results = None
        self.data = None
        self.formula = None
        self.family = None
        self.link = None
        self.weights = None
        self.tweedie_var_power = 1.5  # Default value
        self.alpha = 1.0  # Default for power link
        self.exposure = None
        self.training_data = None  # For filtered training data
        self.training_weights = None  # For filtered weightsself.predictor_data
        self.predictor_data = None # For filtered prediction data
        self.predictor_weights = None # For filtered prediction weights
        
        # Available families and links
        self.available_families = {
            'gaussian': Gaussian,
            'binomial': Binomial,
            'poisson': Poisson,
            'gamma': Gamma,
            'inverse_gaussian': InverseGaussian,
            'negative_binomial': NegativeBinomial,
            'tweedie': Tweedie
        }
        
        self.available_links = {
            'identity': Identity,
            'log': Log,
            'logit': Logit,
            'inverse': Power,  # Power(power=-1)
            'power': Power,
            'probit': Probit,
            'cloglog': CLogLog,
            'nbinom': NBLink
        }
    
    def load_data(self, data):
        """
        Load data for the model.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data to use for modeling
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
            return True
        else:
            raise ValueError("Data must be a pandas DataFrame")
    
    def set_formula(self, formula):
        """
        Set the model formula using R-style formula notation.
        
        Parameters
        ----------
        formula : str
            R-style formula notation (e.g., 'y ~ x1 + x2')
        """
        self.formula = formula
    
    def set_weights(self, weights):
        """
        Set observation weights for the model.
        
        Parameters
        ----------
        weights : array-like or pandas.Series
            Weights for each observation. Must be the same length as the data.
        """
        if hasattr(weights, 'values'):
            # If it's a pandas Series, extract the values
            weights = weights.values
        
        if len(weights) != len(self.data):
            raise ValueError("Weights must have the same length as the data")
        
        self.weights = weights
        return self
    
    def set_family(self, family_name, link_name=None, tweedie_var_power=None, alpha=None):
        """
        Set the error distribution family and link function.
        
        Parameters
        ----------
        family_name : str
            Name of the error distribution family
            ('gaussian', 'binomial', 'poisson', 'gamma', 'inverse_gaussian', 
             'negative_binomial', 'tweedie')
        link_name : str, optional
            Name of the link function
            ('identity', 'log', 'logit', 'inverse', 'power', 'probit', 'cloglog', 'nbinom')
        tweedie_var_power : float, optional
            Variance power for Tweedie distribution (typically 1-2)
        alpha : float, optional
            Parameter for power link function
        """
        if family_name not in self.available_families:
            raise ValueError(f"Family {family_name} not available. Choose from: {list(self.available_families.keys())}")
        
        if link_name and link_name not in self.available_links:
            raise ValueError(f"Link function {link_name} not available. Choose from: {list(self.available_links.keys())}")
        
        # Store additional parameters
        if tweedie_var_power is not None:
            self.tweedie_var_power = tweedie_var_power
        if alpha is not None:
            self.alpha = alpha
        
        # Create appropriate link function
        link_func = None
        if link_name:
            if link_name == 'inverse':
                link_func = self.available_links[link_name](power=-1)
            elif link_name == 'power' and alpha is not None:
                link_func = self.available_links[link_name](power=alpha)
            else:
                link_func = self.available_links[link_name]()
        
        # Create appropriate family
        if family_name == 'tweedie' and tweedie_var_power is not None:
            if link_func:
                self.family = self.available_families[family_name](var_power=self.tweedie_var_power, link=link_func)
            else:
                self.family = self.available_families[family_name](var_power=self.tweedie_var_power)
        else:
            if link_func:
                self.family = self.available_families[family_name](link=link_func)
            else:
                self.family = self.available_families[family_name]()
    
    def build_model(self):
        """
        Build the GLM model using the specified formula, data, and family.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before building model")
        
        if self.formula is None:
            raise ValueError("Formula must be set before building model")
        
        if self.family is None:
            # Default to Gaussian with identity link if not specified
            self.family = Gaussian(Identity())
        
        # Build the model
        if self.weights is not None:
            # Handle different weight interpretations based on model type
            
            # Case 1: Poisson models with log link (insurance frequency)
            # Use weights as exposure - log(weights) becomes an offset
            if isinstance(self.family, Poisson) and isinstance(self.family.link, Log):
                model = sm.GLM.from_formula(
                    formula=self.formula, 
                    data=self.data, 
                    family=self.family, 
                    exposure=self.weights
                )
            
            # Case 2: Gamma models with log link (insurance severity)
            # Often use weights as claim counts
            elif isinstance(self.family, Gamma) and isinstance(self.family.link, Log):
                model = sm.GLM.from_formula(
                    formula=self.formula, 
                    data=self.data, 
                    family=self.family, 
                    freq_weights=self.weights
                )
            
            # Case 3: Tweedie models (often used for pure premium modeling)
            # For var_power between 1-2, weights often represent exposure
            elif isinstance(self.family, Tweedie) and 1 < self.tweedie_var_power < 2:
                if isinstance(self.family.link, Log):
                    # With log link, use as exposure
                    model = sm.GLM.from_formula(
                        formula=self.formula, 
                        data=self.data, 
                        family=self.family, 
                        exposure=self.weights
                    )
                else:
                    # Otherwise as frequency weights
                    model = sm.GLM.from_formula(
                        formula=self.formula, 
                        data=self.data, 
                        family=self.family, 
                        freq_weights=self.weights
                    )
            
            # Case 4: All other models 
            # Use standard frequency weights approach
            else:
                model = sm.GLM.from_formula(
                    formula=self.formula, 
                    data=self.data, 
                    family=self.family, 
                    freq_weights=self.weights
                )
        else:
            model = sm.formula.glm(formula=self.formula, data=self.data, family=self.family)
        
        self.model = model
        return model
    
    def fit_model(self):
        """
        Fit the GLM model and store the results.
        
        Returns
        -------
        statsmodels.genmod.generalized_linear_model.GLMResults
            The fitted model results
        """
        if self.model is None:
            self.build_model()
        
        self.results = self.model.fit()
        return self.results
    
    def get_summary(self):
        """
        Get a summary of the fitted model.
        
        Returns
        -------
        statsmodels.iolib.summary.Summary
            Summary object containing model statistics
        """
        if self.results is None:
            raise ValueError("Model must be fit before getting summary")
        
        return self.results.summary()
    
    def predict(self, new_data=None):
        """
        Generate predictions from the fitted model.
        
        Parameters
        ----------
        new_data : pandas.DataFrame, optional
            New data for prediction. If None, uses the predictor_data if available, otherwise uses the training data.
        
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        if self.results is None:
            raise ValueError("Model must be fit before making predictions")
        
        if new_data is not None:
            return self.results.predict(new_data)
        elif self.predictor_data is not None:
            return self.results.predict(self.predictor_data)
        else:
            return self.results.predict()
    
    def get_aic(self):
        """Get Akaike Information Criterion for the fitted model."""
        if self.results is None:
            raise ValueError("Model must be fit before getting AIC")
        
        return self.results.aic
    
    def get_bic(self):
        """Get Bayesian Information Criterion for the fitted model."""
        if self.results is None:
            raise ValueError("Model must be fit before getting BIC")
        
        return self.results.bic
    
    def get_deviance(self):
        """Get deviance for the fitted model."""
        if self.results is None:
            raise ValueError("Model must be fit before getting deviance")
        
        return self.results.deviance
    
    def get_df_resid(self):
        """Get residual degrees of freedom for the fitted model."""
        if self.results is None:
            raise ValueError("Model must be fit before getting degrees of freedom")
        
        return self.results.df_resid
    
    def save_model(self, file_path):
        """
        Save the current model to a file using pickle
        
        Parameters
        ----------
        file_path : str
            Path to save the model to
        
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        if not hasattr(self, 'results') or self.results is None:
            raise ValueError("No fitted model to save")
        
        # Get UI state information from the parent window if available
        predictor_roles = {}
        polynomial_terms = {}
        
        if hasattr(self, 'parent_window') and self.parent_window is not None:
            # Get predictor items from the sidebar
            for name, widget in self.parent_window.predictors_sidebar.predictor_items.items():
                predictor_roles[name] = widget.role
                polynomial_terms[name] = widget.polynomial_terms.copy()
        
        # Prepare model data dictionary
        model_data = {
            'formula': self.formula,
            'data': self.data,
            'family': self.family,
            'weights': self.weights,
            'results': self.results,
            'exposure': self.exposure,
            'tweedie_var_power': self.tweedie_var_power,
            # UI state information
            'predictor_roles': predictor_roles,
            'polynomial_terms': polynomial_terms
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, file_path):
        """
        Load a saved model from a file
        
        Parameters
        ----------
        file_path : str
            Path to load the model from
            
        Returns
        -------
        bool
            True if load was successful, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Restore model components
            self.formula = model_data.get('formula')
            self.data = model_data.get('data')
            self.family = model_data.get('family')
            self.weights = model_data.get('weights')
            self.results = model_data.get('results')
            self.exposure = model_data.get('exposure')
            self.tweedie_var_power = model_data.get('tweedie_var_power')
            
            # Extract UI state information
            predictor_roles = model_data.get('predictor_roles', {})
            polynomial_terms = model_data.get('polynomial_terms', {})
            
            # Save UI state information for the parent window to use
            self.saved_predictor_roles = predictor_roles
            self.saved_polynomial_terms = polynomial_terms
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def set_training_data(self, split_column, train_partition):
        """
        Create a filtered dataset for training based on split column and train partition value.
        
        Parameters
        ----------
        split_column : str
            Column name to use for filtering data
        train_partition : str or int or float
            Value in the split column to filter for. Only rows where
            split_column equals train_partition will be included in training_data.
        
        Returns
        -------
        bool
            True if training data creation was successful, False otherwise
        """
        logger.info("Setting training data...")
        logger.info(f"Split column: {split_column}, Train partition: {train_partition}")
        
        if self.data is None:
            raise ValueError("Data must be loaded before creating training data")
            
        if split_column not in self.data.columns:
            raise ValueError(f"Split column '{split_column}' not found in data")
            
        try:
            # Create filtered dataset based on the split column and train partition
            self.training_data = self.data[self.data[split_column] == train_partition].copy()
            logger.info(f"Created training data with {len(self.training_data)} rows")
            
            # Create corresponding weights if weights exist
            if self.weights is not None:
                if isinstance(self.weights, np.ndarray):
                    # Convert weights to Series with same index as data for proper filtering
                    weights_series = pd.Series(self.weights, index=self.data.index)
                    self.training_weights = weights_series.loc[self.training_data.index].values
                elif isinstance(self.weights, pd.Series):
                    self.training_weights = self.weights.loc[self.training_data.index].values
                else:
                    raise ValueError("Weights must be a numpy array or pandas Series")
                logger.info(f"Filtered training weights, length: {len(self.training_weights) if self.training_weights is not None else 'None'}")
            else:
                logger.info("No weights available for training data")
            
            return True
        except Exception as e:
            logger.error(f"Error creating training data: {str(e)}")
            return False
            
    def clear_training_data(self):
        """
        Clear the training data, reverting to using the full dataset for training
        
        Returns
        -------
        bool
            True if successful
        """
        logger.info("Clearing training data, reverting to full dataset for training")
        try:
            self.training_data = None
            self.training_weights = None
            logger.info("Training data cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing training data: {str(e)}")
            return False

    def set_prediction_data(self, split_column, prediction_partition):
        """
        Create a filtered dataset for predictions based on split column and prediction partition value.
        
        Parameters
        ----------
        split_column : str
            Column name to use for filtering data
        prediction_partition : str or int or float or 'all'
            Value in the split column to filter for. Only rows where
            split_column equals prediction_partition will be included in predictor_data.
            If 'all', the entire dataset will be used.
        
        Returns
        -------
        bool
            True if prediction data creation was successful, False otherwise
        """
        logger.info("Setting prediction data...")
        logger.info(f"Split column: {split_column}, Prediction partition: {prediction_partition}") 
        if self.data is None:
            raise ValueError("Data must be loaded before creating prediction data")
            
        if split_column not in self.data.columns and prediction_partition != 'all':
            raise ValueError(f"Split column '{split_column}' not found in data")
            
        try:
            # Initialize predictor_weights to None by default
            self.predictor_weights = None
            
            # If 'all' is selected, use the entire dataset
            if prediction_partition == 'all':
                self.predictor_data = self.data.copy()
                if self.weights is not None:
                    self.predictor_weights = self.weights
                    logger.info(f"Using all weights, length: {len(self.predictor_weights) if self.predictor_weights is not None else 'None'}")
                else:
                    logger.info("No weights available for prediction data")
            else:
                # Create filtered dataset based on the split column and prediction partition
                self.predictor_data = self.data[self.data[split_column] == prediction_partition].copy()
                
                # Create corresponding weights if weights exist
                if self.weights is not None:
                    if isinstance(self.weights, np.ndarray):
                        # Convert weights to Series with same index as data for proper filtering
                        weights_series = pd.Series(self.weights, index=self.data.index)
                        self.predictor_weights = weights_series.loc[self.predictor_data.index].values
                    elif isinstance(self.weights, pd.Series):
                        self.predictor_weights = self.weights.loc[self.predictor_data.index].values
                    else:
                        raise ValueError("Weights must be a numpy array or pandas Series")
                    logger.info(f"Filtered weights for partition, length: {len(self.predictor_weights) if self.predictor_weights is not None else 'None'}")
                else:
                    logger.info("No weights available for prediction data")
            
            logger.info(f"Created prediction data with {len(self.predictor_data)} rows")
            logger.info(f"Predictor weights: {type(self.predictor_weights).__name__}, {len(self.predictor_weights) if self.predictor_weights is not None else 'None'}")
            return True
        except Exception as e:
            logger.error(f"Error creating prediction data: {str(e)}")
            # Reset predictor data on error
            self.predictor_data = None
            self.predictor_weights = None
            return False
            
    def clear_prediction_data(self):
        """
        Clear the prediction data and weights, reverting to using the full dataset.
        
        Returns
        -------
        bool
            True if clearing was successful
        """
        self.predictor_data = None
        self.predictor_weights = None
        return True
