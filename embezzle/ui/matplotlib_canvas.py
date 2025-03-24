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
        weights : str, optional
            The weights column name for weighted counts
        custom_bins : dict, optional
            Dictionary with custom binning parameters:
            - min: minimum value for binning
            - max: maximum value for binning
            - n_bins: number of intervals
        """
        # Clear previous plot
        self.clear_plot()
        
        if predictor not in data.columns or response not in data.columns:
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
            else:
                # Use default binning
                n_bins = min(10, len(data[predictor].unique()))
                bins = pd.cut(data[predictor], bins=n_bins)
            
            # Group by bins
            if weights is not None and weights in data.columns:
                # Use weights for counts and to calculate weighted response
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                agg_dict[weights] = 'sum'
                
                grouped = data.groupby(bins, observed=False).agg(agg_dict)
                
                # Calculate weighted average (sum of response / sum of weights) if both columns exist
                if (response, 'sum') in grouped.columns and (weights, 'sum') in grouped.columns:
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'] / grouped[weights, 'sum']
                else:
                    # Fallback - create a dummy weighted_avg column with zeros
                    grouped[response, 'weighted_avg'] = 0.0
                
                count_column = (weights, 'sum')
            else:
                # Use counts for unweighted data
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                
                grouped = data.groupby(bins, observed=False).agg(agg_dict)
                
                # Calculate simple average (sum of response / count) if both columns exist
                if (response, 'sum') in grouped.columns and (response, 'count') in grouped.columns:
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'] / grouped[response, 'count']
                else:
                    # Fallback - create a dummy weighted_avg column with zeros
                    grouped[response, 'weighted_avg'] = 0.0
                
                # Add count column for the chart bars
                count_column = (response, 'count')
                
            # Convert categorical bins to string for plotting
            x_values = [str(b) for b in grouped.index]
            
        else:
            # For categorical predictors, group directly
            if weights is not None and weights in data.columns:
                # Use weights for counts and to calculate weighted response
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                agg_dict[weights] = 'sum'
                
                grouped = data.groupby(predictor, observed=False).agg(agg_dict)
                
                # Calculate weighted average (sum of response / sum of weights) if both columns exist
                if (response, 'sum') in grouped.columns and (weights, 'sum') in grouped.columns:
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'] / grouped[weights, 'sum']
                else:
                    # Fallback - create a dummy weighted_avg column with zeros
                    grouped[response, 'weighted_avg'] = 0.0
                
                count_column = (weights, 'sum')
            else:
                # Use counts for unweighted data
                agg_dict = {response: ['count']}
                if response in data.columns:
                    agg_dict[response].append('sum')
                
                grouped = data.groupby(predictor, observed=False).agg(agg_dict)
                
                # Calculate simple average (sum of response / count) if both columns exist
                if (response, 'sum') in grouped.columns and (response, 'count') in grouped.columns:
                    grouped[response, 'weighted_avg'] = grouped[response, 'sum'] / grouped[response, 'count']
                else:
                    # Fallback - create a dummy weighted_avg column with zeros
                    grouped[response, 'weighted_avg'] = 0.0
                
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
        self.axes.bar(bar_positions, scaled_counts, color='yellow', edgecolor='black', alpha=0.7, zorder=1)
        
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
            # Calculate weighted prediction per bin using the same approach as for response
            if pd.api.types.is_numeric_dtype(data[predictor]):
                if weights is not None and weights in data.columns:
                    pred_grouped = data.groupby(bins, observed=False).agg({
                        '_predicted': ['sum'],
                        weights: 'sum'
                    })
                    # Calculate weighted average for predictions (sum of predicted / sum of weights)
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted', 'sum'] / pred_grouped[weights, 'sum']
                else:
                    pred_grouped = data.groupby(bins, observed=False).agg({
                        '_predicted': ['sum', 'count']
                    })
                    # Calculate simple average for predictions
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted', 'sum'] / pred_grouped['_predicted', 'count']
            else:
                if weights is not None and weights in data.columns:
                    pred_grouped = data.groupby(predictor, observed=False).agg({
                        '_predicted': ['sum'],
                        weights: 'sum'
                    })
                    # Calculate weighted average for predictions (sum of predicted / sum of weights)
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted', 'sum'] / pred_grouped[weights, 'sum']
                else:
                    pred_grouped = data.groupby(predictor, observed=False).agg({
                        '_predicted': ['sum', 'count']
                    })
                    # Calculate simple average for predictions
                    pred_grouped['_predicted', 'weighted_avg'] = pred_grouped['_predicted', 'sum'] / pred_grouped['_predicted', 'count']
            
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
            # Calculate weighted prediction per bin using the same approach as for response
            if pd.api.types.is_numeric_dtype(data[predictor]):
                if weights is not None and weights in data.columns:
                    pred_specific_grouped = data.groupby(bins, observed=False).agg({
                        predictor_specific_col: ['sum'],
                        weights: 'sum'
                    })
                    # Calculate weighted average for predictions (sum of predicted / sum of weights)
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[predictor_specific_col, 'sum'] / pred_specific_grouped[weights, 'sum']
                else:
                    pred_specific_grouped = data.groupby(bins, observed=False).agg({
                        predictor_specific_col: ['sum', 'count']
                    })
                    # Calculate simple average for predictions
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[predictor_specific_col, 'sum'] / pred_specific_grouped[predictor_specific_col, 'count']
            else:
                if weights is not None and weights in data.columns:
                    pred_specific_grouped = data.groupby(predictor, observed=False).agg({
                        predictor_specific_col: ['sum'],
                        weights: 'sum'
                    })
                    # Calculate weighted average for predictions (sum of predicted / sum of weights)
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[predictor_specific_col, 'sum'] / pred_specific_grouped[weights, 'sum']
                else:
                    pred_specific_grouped = data.groupby(predictor, observed=False).agg({
                        predictor_specific_col: ['sum', 'count']
                    })
                    # Calculate simple average for predictions
                    pred_specific_grouped[predictor_specific_col, 'weighted_avg'] = pred_specific_grouped[predictor_specific_col, 'sum'] / pred_specific_grouped[predictor_specific_col, 'count']
            
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
