"""
Matplotlib canvas module for Embezzle

This module provides a canvas for embedding Matplotlib plots in PyQt widgets.
"""

import matplotlib
matplotlib.use('qt5agg')  # Use qt5agg which is supported
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt widgets"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create a twin axis for the counts
        self.count_axes = self.axes.twinx()
        
        # Set up figure to better fill available space
        self.fig.subplots_adjust(bottom=0.15, left=0.05, right=0.95, top=0.95)
        
        # Connect the figure to resize events to eliminate whitespace
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        
    def on_resize(self, event):
        """Handle resize events to maintain proper layout"""
        self.fig.tight_layout()
        self.draw()
        
    def clear_plot(self):
        """Clear the plot"""
        self.axes.clear()
        self.count_axes.clear()
        self.draw()
        
    def create_dual_axis_chart(self, data, predictor, response, weights=None, custom_bins=None):
        """
        Create a dual-axis chart with bar chart for counts and line chart for response
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data to plot
        predictor : str
            The predictor column name to use as x-axis
        response : str
            The response column name for the line chart
        weights : str or numpy.ndarray, optional
            The weights column name or array for weighted counts
        custom_bins : dict, optional
            Dictionary with custom binning parameters:
            - min: minimum value for binning
            - max: maximum value for binning
            - n_bins: number of intervals
        """
        # Clear previous plot
        self.clear_plot()
        
        # Log the data source information
        logger.info(f"Creating chart with {len(data)} rows of data for predictor: {predictor}, response: {response}")
        
        # Handle weights properly - could be a column name (str) or numpy array
        weight_column = None
        if isinstance(weights, np.ndarray):
            logger.info(f"Using numpy array weights with length: {len(weights)}")
            # If weights is a numpy array, we'll create a temporary column
            weight_column = '_temp_weights'
            temp_data = data.copy()
            if len(weights) == len(data):
                temp_data[weight_column] = weights
                logger.info(f"Added temporary weights column '{weight_column}' to data")
                data = temp_data
            else:
                logger.warning(f"Weights array length ({len(weights)}) doesn't match data length ({len(data)}). Ignoring weights.")
                weight_column = None
        elif isinstance(weights, str):
            if weights in data.columns:
                logger.info(f"Using weights column: {weights}")
                weight_column = weights
            else:
                logger.warning(f"Weights column '{weights}' not found in data. Ignoring weights.")
                weight_column = None
        else:
            logger.info("No weights specified for chart")
            
        # Log custom binning information
        if custom_bins is not None:
            logger.info(f"Using custom bins: min={custom_bins.get('min')}, max={custom_bins.get('max')}, bins={custom_bins.get('n_bins')}")
        
        if predictor not in data.columns or response not in data.columns:
            logger.warning(f"Missing required columns in data: predictor={predictor in data.columns}, response={response in data.columns}")
            return
            
        # Check if the predictor is numeric or categorical
        if pd.api.types.is_numeric_dtype(data[predictor]):
            # For numeric predictors, bin the data
            if custom_bins is not None and isinstance(custom_bins, dict):
                # Use custom binning parameters if provided
                min_val = custom_bins.get('min')
                max_val = custom_bins.get('max')
                n_bins = custom_bins.get('n_bins', 10)
                
                # Create custom bins
                bins = pd.cut(data[predictor], 
                              bins=np.linspace(min_val, max_val, n_bins + 1),
                              include_lowest=True)
                logger.debug(f"Created {n_bins} custom bins from {min_val} to {max_val}")
            else:
                # Use default binning
                n_bins = min(10, len(data[predictor].unique()))
                bins = pd.cut(data[predictor], bins=n_bins)
                logger.debug(f"Created {n_bins} automatic bins")
            
            # Group by bins
            if weight_column is not None:
                # Log that we're using weights
                logger.info(f"Grouping numeric data with weights from {weight_column}")
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                agg_dict[weight_column] = 'sum'
                
                grouped = data.groupby(bins, observed=False).agg(agg_dict)
                
                # Calculate weighted average (sum of response / sum of weights) if both columns exist
                if (response, 'sum') in grouped.columns and (weight_column, 'sum') in grouped.columns:
                    # Handle division by zero and NaN values
                    weights_sum = grouped[weight_column, 'sum'].replace(0, np.nan)
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'].div(weights_sum)
                
                count_column = (weight_column, 'sum')
            else:
                # Use counts for unweighted data
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                
                grouped = data.groupby(bins, observed=False).agg(agg_dict)
                
                # Calculate simple average (sum of response / count) if both columns exist
                if (response, 'sum') in grouped.columns and (response, 'count') in grouped.columns:
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'] / grouped[response, 'count']
                
                # Add count column for the chart bars
                count_column = (response, 'count')
                
            # Convert categorical bins to string for plotting
            x_values = [str(b) for b in grouped.index]
            
        else:
            # For categorical predictors, group directly
            if weight_column is not None:
                # Log that we're using weights
                logger.info(f"Grouping categorical data with weights from {weight_column}")
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                agg_dict[weight_column] = 'sum'
                
                grouped = data.groupby(predictor, observed=False).agg(agg_dict)
                
                # Calculate weighted average (sum of response / sum of weights) if both columns exist
                if (response, 'sum') in grouped.columns and (weight_column, 'sum') in grouped.columns:
                    # Handle division by zero and NaN values
                    weights_sum = grouped[weight_column, 'sum'].replace(0, np.nan)
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'].div(weights_sum)
                
                count_column = (weight_column, 'sum')
            else:
                # Use counts for unweighted data
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                
                grouped = data.groupby(predictor, observed=False).agg(agg_dict)
                
                # Calculate simple average (sum of response / count) if both columns exist
                if (response, 'sum') in grouped.columns and (response, 'count') in grouped.columns:
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'] / grouped[response, 'count']
                
                # Add count column for the chart bars
                count_column = (response, 'count')
                
            x_values = grouped.index
        
        # Ensure the required columns exist before accessing them
        if (response, 'weighted_avg') not in grouped.columns:
            grouped[response, 'weighted_avg'] = 0.0
        
        # Make sure count_column exists
        if count_column not in grouped.columns:
            if (response, 'count') in grouped.columns:
                count_column = (response, 'count')
            else:
                # Default to a column of ones if nothing else is available
                grouped['_count', ''] = 1
                count_column = ('_count', '')
        
        # Extract values - use weighted_avg instead of simple mean
        response_values = grouped[response, 'weighted_avg']
        count_values = grouped[count_column]
        
        # Create a new axes instance for the chart
        self.axes.clear()
        self.count_axes.clear()
        
        # Draw both elements on the same axes with appropriate scaling
        # First create bars on the main axes with a very low zorder to ensure they're behind
        bar_positions = np.arange(len(x_values))
        
        # Scale down the count values to be in a similar range as the response values
        # This helps ensure the bars don't dominate the chart
        max_response = response_values.max() if not response_values.empty else 1
        max_count = count_values.max() if not count_values.empty else 1
        scale_factor = max_response / max_count if max_count > 0 else 1
        scaled_counts = count_values * scale_factor * 0.5  # Scale to approximately half the response height
        
        # Plot bars in the background
        bars = self.axes.bar(bar_positions, scaled_counts, color='yellow', edgecolor='black', alpha=0.7, zorder=1)
        
        # Add count labels above each bar
        for i, (count, bar) in enumerate(zip(count_values, bars)):
            if not pd.isna(count) and count > 0:
                # Format count as integer if it's whole number, otherwise with 1 decimal place
                count_label = f'{int(count):,}' if count.is_integer() else f'{count:.1f}'
                self.axes.annotate(count_label, 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, color='black',
                    zorder=4)  # Higher than bars, lower than line
        
        # Plot means on top of bars
        self.axes.plot(bar_positions, response_values, color='purple', marker='o', linewidth=2, zorder=5)
        
        # Add value labels above each point for mean values
        for i, y in zip(bar_positions, response_values):
            if not pd.isna(y):
                self.axes.annotate(f'{y:.2f}', 
                    xy=(i, y), xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, color='purple',
                    zorder=6)  # Even higher zorder for labels
        
        # Add model predictions if available
        if '_predicted' in data.columns:
            logging.getLogger(__name__).info("Processing model predictions for chart")
            # Calculate weighted prediction per bin using the same approach as for response
            if pd.api.types.is_numeric_dtype(data[predictor]):
                if weight_column is not None:
                    # For insurance frequency models, we need to sum the predictions and divide by sum of weights
                    # Predictions from the model are already frequency per unit exposure, so we multiply by exposure
                    # before grouping to get total predicted claims
                    data['_predicted_total'] = data['_predicted'] * data[weight_column]
                    
                    pred_grouped = data.groupby(bins, observed=False).agg({
                        '_predicted_total': ['sum'],
                        weight_column: 'sum'
                    })
                    
                    # Calculate frequency as sum(predicted_total) / sum(weights)
                    # This matches how response values are calculated: sum(claims) / sum(exposure)
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted_total', 'sum'] / pred_grouped[weight_column, 'sum']
                    logging.getLogger(__name__).info(f"Using frequency-based predictions (sum(pred*weight)/sum(weights)) for chart")
                else:
                    pred_grouped = data.groupby(bins, observed=False).agg({
                        '_predicted': ['mean', 'count']
                    })
                    # If no weights, use the mean
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted', 'mean']
                    logging.getLogger(__name__).info(f"Using mean-based predictions for chart (no weights available)")
            else:
                if weight_column is not None:
                    # Same approach for categorical predictors
                    # Multiply predictions by weights to get total predicted claims
                    data['_predicted_total'] = data['_predicted'] * data[weight_column]
                    
                    pred_grouped = data.groupby(predictor, observed=False).agg({
                        '_predicted_total': ['sum'],
                        weight_column: 'sum'
                    })
                    
                    # Calculate frequency as sum(predicted_total) / sum(weights)
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted_total', 'sum'] / pred_grouped[weight_column, 'sum']
                    logging.getLogger(__name__).info(f"Using frequency-based predictions (sum(pred*weight)/sum(weights)) for chart")
                else:
                    pred_grouped = data.groupby(predictor, observed=False).agg({
                        '_predicted': ['mean', 'count']
                    })
                    # If no weights, use the mean
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted', 'mean']
                    logging.getLogger(__name__).info(f"Using mean-based predictions for chart (no weights available)")
            
            # Use weighted average for predictions
            self.axes.plot(bar_positions, pred_grouped['_predicted', 'weighted_avg'], 
                color='darkgreen', marker='s', linewidth=2, linestyle='--',
                zorder=5)  # Same zorder as other line
            
            # Add value labels above each point for prediction values
            for i, y in zip(bar_positions, pred_grouped['_predicted', 'weighted_avg']):
                if not pd.isna(y):
                    self.axes.annotate(f'{y:.2f}', 
                        xy=(i, y), xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color='darkgreen',
                        zorder=6)  # Same high zorder as other labels
        
        # Check for predictor-specific prediction column
        predictor_specific_col = f'_predicted_{predictor}'
        if predictor_specific_col in data.columns:
            logger.info(f"Found predictor-specific prediction column: {predictor_specific_col}")
            # Calculate weighted prediction per bin using the same approach as for response
            if pd.api.types.is_numeric_dtype(data[predictor]):
                if weight_column is not None:
                    # For insurance frequency models, we need to sum the predictions and divide by sum of weights
                    # Predictions from the model are already frequency per unit exposure, so we multiply by exposure
                    # before grouping to get total predicted claims
                    data[f'{predictor_specific_col}_total'] = data[predictor_specific_col] * data[weight_column]
                    
                    pred_specific_grouped = data.groupby(bins, observed=False).agg({
                        f'{predictor_specific_col}_total': ['sum'],
                        weight_column: 'sum'
                    })
                    
                    # Calculate frequency as sum(predicted_total) / sum(weights)
                    # This matches how response values are calculated: sum(claims) / sum(exposure)
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[f'{predictor_specific_col}_total', 'sum'] / pred_specific_grouped[weight_column, 'sum']
                    logging.getLogger(__name__).info(f"Using frequency-based predictions (sum(pred*weight)/sum(weights)) for chart")
                else:
                    pred_specific_grouped = data.groupby(bins, observed=False).agg({
                        predictor_specific_col: ['mean', 'count']
                    })
                    # If no weights, use the mean
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[predictor_specific_col, 'mean']
                    logging.getLogger(__name__).info(f"Using mean-based predictions for chart (no weights available)")
            else:
                if weight_column is not None:
                    # Same approach for categorical predictors
                    # Multiply predictions by weights to get total predicted claims
                    data[f'{predictor_specific_col}_total'] = data[predictor_specific_col] * data[weight_column]
                    
                    pred_specific_grouped = data.groupby(predictor, observed=False).agg({
                        f'{predictor_specific_col}_total': ['sum'],
                        weight_column: 'sum'
                    })
                    
                    # Calculate frequency as sum(predicted_total) / sum(weights)
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[f'{predictor_specific_col}_total', 'sum'] / pred_specific_grouped[weight_column, 'sum']
                    logging.getLogger(__name__).info(f"Using frequency-based predictions (sum(pred*weight)/sum(weights)) for chart")
                else:
                    pred_specific_grouped = data.groupby(predictor, observed=False).agg({
                        predictor_specific_col: ['mean', 'count']
                    })
                    # If no weights, use the mean
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[predictor_specific_col, 'mean']
                    logging.getLogger(__name__).info(f"Using mean-based predictions for chart (no weights available)")
            
            # Use weighted average for predictor-specific predictions - plot as neon green line
            self.axes.plot(bar_positions, pred_specific_grouped[predictor_specific_col, 'weighted_avg'], 
                color='#00FF00', marker='s', linewidth=2, linestyle='--',
                zorder=5)  # Same zorder as other line
            
            # Add value labels above each point for prediction values
            for i, y in zip(bar_positions, pred_specific_grouped[predictor_specific_col, 'weighted_avg']):
                if not pd.isna(y):
                    self.axes.annotate(f'{y:.2f}', 
                        xy=(i, y), xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color='#00FF00',
                        zorder=6)  # Same high zorder as other labels
        
        # Set x-axis labels
        self.axes.set_xticks(bar_positions)
        self.axes.set_xticklabels(x_values, rotation=45, ha='right')
        
        # Set chart title
        self.axes.set_title(f'Predicted Values - {predictor}', fontsize=12, pad=10)
        
        # Remove y-axis label
        self.axes.set_ylabel('')
        
        # Add a second y-axis to show the original count scale
        if not self.count_axes.has_data():
            self.count_axes = self.axes.twinx()
        
        # Hide the y-ticks on the count axis to avoid confusion
        self.count_axes.yaxis.set_ticks_position('none')
        self.count_axes.yaxis.set_ticklabels([])
        self.count_axes.set_ylabel('')
        
        # Adjust layout to make room for rotated labels
        self.fig.tight_layout()
        
        # Draw the canvas
        self.draw()
