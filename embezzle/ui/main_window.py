"""
Main window module for Embezzle

This module provides a minimal application window with a menu
for loading data files to work with GLMs.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QStatusBar, QFrame, QSplitter, QSizePolicy,
    QLabel, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QTableWidget, 
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
import pandas as pd
import numpy as np
import logging

from embezzle.models.model_builder import GLMBuilder
from embezzle.ui.model_specification_dialog import ModelSpecificationDialog
from embezzle.ui.matplotlib_canvas import MatplotlibCanvas
from embezzle.ui.predictor_widgets import PredictorsSidebar


class EmbezzleMainWindow(QMainWindow):
    """
    Main window for the Embezzle application
    
    This class provides a minimal interface with a menu for loading data.
    """
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        self.setWindowTitle("Embezzle GLM Tool")
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize data and model builder
        self.data = None
        self.model_builder = GLMBuilder()
        self.model_builder.parent_window = self  # Add reference to parent window
        self.model_specs = None
        
        # Dictionary to store binning settings for each predictor
        self.predictor_binning_settings = {}
        
        # Dictionary to store min/max/intervals per factor
        self.factor_ranges = {}
        
        # Initialize menu actions
        self.open_action = None
        self.exit_action = None
        self.specify_model_action = None
        self.save_model_action = None
        self.load_model_action = None
        
        # Create the status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Create the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create welcome/instructions panel
        self.welcome_panel = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_panel)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Welcome title
        welcome_title = QLabel("Welcome to Embezzle GLM Tool")
        welcome_title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        welcome_layout.addWidget(welcome_title)
        
        # Instructions
        instructions = QLabel(
            "To get started:\n\n"
            "1. Load your data via File > Open Data File...\n"
            "2. Specify your model settings via Specify > Model Error and Link...\n"
            "3. Select a response variable in the model specification dialog\n\n"
            "Once you have selected a response variable, the modeling interface will appear."
        )
        instructions.setStyleSheet("font-size: 16px; margin: 20px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(instructions)
        
        main_layout.addWidget(self.welcome_panel)
        
        # Create the main horizontal splitter (sidebar | content)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Create the predictors sidebar (left side)
        self.predictors_sidebar = PredictorsSidebar()
        self.predictors_sidebar.mainwindow_reference = self  # Add direct reference to the main window
        self.main_splitter.addWidget(self.predictors_sidebar)
        
        # Connect the Fit button
        self.predictors_sidebar.fit_button.clicked.connect(self.fit_model)
        
        # Create the content area with nested splitters
        # Vertical splitter for top/bottom sections
        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.vertical_splitter)
        
        # Top section with horizontal splitter
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.vertical_splitter.addWidget(self.top_splitter)
        
        # Top-left panel
        self.top_left_panel = QFrame()
        self.top_left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.top_left_panel.setLayout(QVBoxLayout())
        
        # Add model summary statistics table
        self.model_stats_label = QLabel("Model Summary Statistics")
        self.model_stats_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.top_left_panel.layout().addWidget(self.model_stats_label)
        
        # Create the summary statistics table
        self.model_stats_table = QTableWidget()
        self.model_stats_table.setColumnCount(3)
        self.model_stats_table.setHorizontalHeaderLabels(["Current Model", "Reference Model", "Difference"])
        
        # Set up the rows for the statistics
        self.model_stats_table.setRowCount(4)
        self.model_stats_table.setVerticalHeaderLabels([
            "Fitted Parameters", 
            "Deviance", 
            "Chi-Squared Percentage", 
            "AICc"
        ])
        
        # Initialize with empty values
        for row in range(4):
            for col in range(3):
                if col == 0:  # Only set placeholders for current model column
                    item = QTableWidgetItem("-")
                    self.model_stats_table.setItem(row, col, item)
        
        # Set table properties
        self.model_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model_stats_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model_stats_table.setAlternatingRowColors(True)
        
        self.top_left_panel.layout().addWidget(self.model_stats_table)
        self.top_splitter.addWidget(self.top_left_panel)
        
        # Top-right panel
        self.top_right_panel = QFrame()
        self.top_right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        top_right_layout = QVBoxLayout()
        self.top_right_panel.setLayout(top_right_layout)
        
        # Add continuous factor controls group box
        self.continuous_controls = QGroupBox("Continuous Factor Visual Controls")
        
        # Use horizontal layout for controls
        controls_layout = QHBoxLayout()
        self.continuous_controls.setLayout(controls_layout)
        
        # Create min, max, and intervals spin boxes with labels
        min_layout = QVBoxLayout()
        min_label = QLabel("Min:")
        min_layout.addWidget(min_label)
        self.min_value_spinner = QDoubleSpinBox()
        self.min_value_spinner.setRange(-1000000, 1000000)
        self.min_value_spinner.setDecimals(3)
        self.min_value_spinner.setSingleStep(0.1)
        min_layout.addWidget(self.min_value_spinner)
        
        max_layout = QVBoxLayout()
        max_label = QLabel("Max:")
        max_layout.addWidget(max_label)
        self.max_value_spinner = QDoubleSpinBox()
        self.max_value_spinner.setRange(-1000000, 1000000)
        self.max_value_spinner.setDecimals(3)
        self.max_value_spinner.setSingleStep(0.1)
        max_layout.addWidget(self.max_value_spinner)
        
        intervals_layout = QVBoxLayout()
        intervals_label = QLabel("Intervals:")
        intervals_layout.addWidget(intervals_label)
        self.intervals_spinner = QSpinBox()
        self.intervals_spinner.setRange(2, 50)
        self.intervals_spinner.setValue(10)
        intervals_layout.addWidget(self.intervals_spinner)
        
        # Add layouts to main controls layout
        controls_layout.addLayout(min_layout)
        controls_layout.addLayout(max_layout)
        controls_layout.addLayout(intervals_layout)
        
        # Connect spinners to chart update
        self.min_value_spinner.valueChanged.connect(self.update_chart_with_new_ranges)
        self.max_value_spinner.valueChanged.connect(self.update_chart_with_new_ranges)
        self.intervals_spinner.valueChanged.connect(self.update_chart_with_new_ranges)
        
        # Add factor levels table
        self.levels_table = QTableWidget()
        self.levels_table.setColumnCount(4)
        self.levels_table.setHorizontalHeaderLabels(["Level", "Observations", "Weight", "Response"])
        self.levels_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.levels_table.setAlternatingRowColors(True)
        self.levels_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add parameter stats panel
        self.parameter_stats_panel = QGroupBox("Parameter Statistics")
        param_stats_layout = QVBoxLayout()
        self.parameter_stats_panel.setLayout(param_stats_layout)
        
        # Main parameter info (for continuous factors)
        self.continuous_param_widget = QWidget()
        continuous_layout = QFormLayout(self.continuous_param_widget)
        
        # Create labels for parameter statistics
        self.parameter_name_label = QLabel("Not fitted")
        self.parameter_value_label = QLabel("-")
        self.parameter_se_label = QLabel("-")
        self.parameter_se_pct_label = QLabel("-")
        
        # Add to continuous layout
        continuous_layout.addRow("Parameter:", self.parameter_name_label)
        continuous_layout.addRow("Value (β):", self.parameter_value_label)
        continuous_layout.addRow("Std. Error:", self.parameter_se_label)
        continuous_layout.addRow("SE %:", self.parameter_se_pct_label)
        
        # Categorical levels table
        self.categorical_param_widget = QWidget()
        categorical_layout = QVBoxLayout(self.categorical_param_widget)
        
        self.categorical_label = QLabel("Level Parameters:")
        categorical_layout.addWidget(self.categorical_label)
        
        self.category_params_table = QTableWidget()
        self.category_params_table.setColumnCount(3)
        self.category_params_table.setHorizontalHeaderLabels(["Level", "Value (β)", "SE %"])
        self.category_params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.category_params_table.setAlternatingRowColors(True)
        categorical_layout.addWidget(self.category_params_table)
        
        # Add both widgets to main layout
        param_stats_layout.addWidget(self.continuous_param_widget)
        param_stats_layout.addWidget(self.categorical_param_widget)
        
        # Create layout for table and parameter stats (side by side)
        table_stats_layout = QHBoxLayout()
        table_stats_layout.addWidget(self.levels_table, 7)  # 70% of space
        table_stats_layout.addWidget(self.parameter_stats_panel, 3)  # 30% of space
        
        # Add the controls and table+stats to the top right layout
        top_right_layout.addWidget(self.continuous_controls)
        top_right_layout.addLayout(table_stats_layout, 1)  # Add stretch factor of 1 to make it expand
        # Remove the stretch that was causing empty space
        
        self.top_splitter.addWidget(self.top_right_panel)
        
        # Bottom panel with chart
        self.bottom_panel = QFrame()
        self.bottom_panel.setFrameShape(QFrame.Shape.StyledPanel)
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)  
        bottom_layout.setSpacing(0)  
        self.bottom_panel.setLayout(bottom_layout)
        
        # Create the matplotlib canvas for the chart
        self.chart_canvas = MatplotlibCanvas(self.bottom_panel, width=5, height=4, dpi=100)
        self.chart_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bottom_layout.addWidget(self.chart_canvas)
        
        self.vertical_splitter.addWidget(self.bottom_panel)
        
        # Set the initial sizes for the main splitter (20% for sidebar, 80% for content)
        self.main_splitter.setSizes([200, 800])
        
        # Set initial sizes for the top/bottom vertical split (33% top, 67% bottom)
        # Use a more aggressive ratio (1:5 instead of 33:67) to ensure proper display
        self.vertical_splitter.setSizes([100, 500])
        
        # Set proper size policies for the panels to maintain proportions
        self.top_splitter.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.bottom_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        # Set stretch factors to maintain proper distribution during resizing
        self.vertical_splitter.setStretchFactor(0, 1)  # Top section (index 0)
        self.vertical_splitter.setStretchFactor(1, 3)  # Bottom section (index 1)
        
        # Set initial sizes for the top horizontal split (50% each)
        top_width = 800  # Approximate width of content area
        self.top_splitter.setSizes([int(top_width * 0.5), int(top_width * 0.5)])
        
        # Connect the predictor sidebar's selection callback
        self.predictors_sidebar.on_predictor_selected = self.update_predictor_chart
        
        # Initially hide the modeling interface until a response variable is selected
        self.predictors_sidebar.hide()
        self.main_splitter.hide()
        self.welcome_panel.show()
        
        # Create the menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create the menu bar with File menu"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # Open action
        self.open_action = file_menu.addAction("&Open Data File...")
        self.open_action.triggered.connect(self._load_data)
        
        # Add separator
        file_menu.addSeparator()
        
        # Save Model action
        self.save_model_action = file_menu.addAction("&Save Model...")
        self.save_model_action.triggered.connect(self._save_model)
        
        # Load Model action
        self.load_model_action = file_menu.addAction("&Load Model...")
        self.load_model_action.triggered.connect(self._load_model)
        
        # Add separator
        file_menu.addSeparator()
        
        # Exit action
        self.exit_action = file_menu.addAction("E&xit")
        self.exit_action.triggered.connect(self.close)
        
        # Specify menu
        specify_menu = menu_bar.addMenu("&Specify")
        
        # Model Error and Link action
        self.specify_model_action = specify_menu.addAction("&Model Error and Link...")
        self.specify_model_action.triggered.connect(self.open_model_specification_dialog)
    
    def _load_data(self):
        """Load data from a CSV file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Data File", 
            "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_name:
            try:
                # Load the data
                data = pd.read_csv(file_name)
                
                # Set the data in the model builder
                self.data = data
                self.model_builder.load_data(data)
                
                # Update the status bar
                self.statusBar.showMessage(f"Loaded data from {file_name}")
                
                # Reset the model specs - this will keep the welcome panel showing
                # until user selects a response variable
                self.model_specs = None
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
                self.statusBar.showMessage("Error loading data")
    
    def open_model_specification_dialog(self):
        """Open the model specification dialog"""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        dialog = ModelSpecificationDialog(self, self.data.columns)
        
        if dialog.exec():
            # Get the model specifications
            self.model_specs = dialog.get_model_specs()
            
            # Update the model with the specifications
            self._update_model_with_specs()
            
            # Update the sidebar to exclude selected columns
            excluded_columns = []
            response_variable = self.model_specs.get('response_variable')
            
            if response_variable:
                excluded_columns.append(response_variable)
                
                # Show the modeling interface when a response variable is selected
                self.welcome_panel.hide()
                self.main_splitter.show()
                self.predictors_sidebar.show()
                self.statusBar.showMessage(f"Using '{response_variable}' as response variable")
            else:
                # Hide the modeling interface when no response variable is selected
                self.welcome_panel.show()
                self.main_splitter.hide()
                self.predictors_sidebar.hide()
                self.statusBar.showMessage("No response variable selected")
            
            if self.model_specs.get('weight_column'):
                excluded_columns.append(self.model_specs.get('weight_column'))
            
            if self.model_specs.get('split_column'):
                excluded_columns.append(self.model_specs.get('split_column'))
            
            # Exclude any prediction columns (starting with '_predicted')
            for col in self.data.columns:
                if col.startswith('_predicted'):
                    excluded_columns.append(col)
            
            # Update the predictors sidebar if it's visible
            if self.predictors_sidebar.isVisible():
                self.predictors_sidebar.update_predictors(self.data, excluded_columns)
    
    def _update_model_with_specs(self):
        """Update the model builder with the specifications"""
        if not self.model_specs:
            return
        
        # Get the response and weight columns
        response_variable = self.model_specs.get('response_variable')
        weight_column = self.model_specs.get('weight_column')
        split_column = self.model_specs.get('split_column')
        train_partition = self.model_specs.get('train_partition')
        
        # Set the response variable
        if response_variable:
            self.statusBar.showMessage(f"Using '{response_variable}' as response variable")
        
        # Set the weight column
        if weight_column and weight_column in self.data.columns:
            self.statusBar.showMessage(f"Using '{weight_column}' as weight column")
            # Set the weights in the model builder
            logger = logging.getLogger(__name__)
            logger.info(f"Setting weights from column '{weight_column}'")
            weights = self.data[weight_column]
            self.model_builder.set_weights(weights)
            logger.info(f"Weights set, length: {len(weights)}")
        
        # Set the split column
        if split_column:
            self.statusBar.showMessage(f"Using '{split_column}' for train/test/validation splits")
            # Update the prediction set dropdown in the predictor sidebar
            self.predictors_sidebar.update_prediction_set_combo(split_column, self.data)
            
            # Initialize prediction data based on the current selection in the dropdown
            self.update_prediction_data()
            
            # Store current predictor terms to preserve them
            current_terms = self.predictors_sidebar.get_model_terms()
            had_model = len(current_terms) > 0
            
            # Create filtered training dataset if both split column and train partition are specified
            if train_partition:
                try:
                    # Create the filtered dataset in the model builder
                    self.model_builder.set_training_data(split_column, train_partition)
                    self.statusBar.showMessage(f"Created training dataset filtered by {split_column}={train_partition}")
                except Exception as e:
                    self.statusBar.showMessage(f"Error creating training dataset: {str(e)}")
            else:
                # Clear any existing training data if no train partition is specified
                self.model_builder.clear_training_data()
                self.statusBar.showMessage(f"Using full dataset (no training partition specified)")
        else:
            # If no split column is specified, clear any training data to use the full dataset
            self.model_builder.clear_training_data()
            # No need to preserve model since we're not changing training data
            had_model = False
            current_terms = []
            
        # Update the UI with the current predictor if one is selected
        if hasattr(self.predictors_sidebar, 'selected_predictor') and self.predictors_sidebar.selected_predictor:
            predictor = self.predictors_sidebar.selected_predictor
            
            # Update visualizations with proper error handling
            try:
                self.update_levels_table(predictor, response_variable, weight_column)
                self.update_predictor_chart(predictor)
            except Exception as e:
                self.statusBar.showMessage(f"Error updating visualization: {str(e)}")
                logger = logging.getLogger(__name__)
                logger.error(f"Visualization error: {str(e)}")
        
        # If we had a fitted model before changing training data, re-fit to update parameters
        if had_model and len(current_terms) > 0:
            try:
                # Refit the model with the same terms but new training data
                formula = f"{response_variable} ~ {' + '.join(current_terms)}"
                
                # Configure the model builder with the same formula
                self.model_builder.set_formula(formula)
                
                # Build and fit the model with the new training data
                self.model_builder.build_model()
                self.model_builder.fit_model()
                
                # Update UI to reflect the new fit
                self.update_model_stats_table()
                
                # Update the currently selected predictor's parameters if any
                if hasattr(self.predictors_sidebar, 'selected_predictor') and self.predictors_sidebar.selected_predictor:
                    self.update_parameter_stats(self.predictors_sidebar.selected_predictor)
                
                self.statusBar.showMessage(f"Model automatically refitted with new training data.")
            except Exception as e:
                self.statusBar.showMessage(f"Error refitting model: {str(e)}")
                logger = logging.getLogger(__name__)
                logger.error(f"Model refitting error: {str(e)}")
        
        # Set family and link
        family = self.model_specs.get('family')
        link = self.model_specs.get('link')
        
        # Map to statsmodels family and link
        family_map = {
            'gaussian': 'gaussian',
            'poisson': 'poisson',
            'gamma': 'gamma',
            'binomial': 'binomial',
            'negativebinomial': 'negativebinomial',
            'tweedie': 'tweedie',
            'multinomial': 'multinomial',
            # Add other mappings as needed
        }
        
        link_map = {
            'identity': 'identity',
            'log': 'log',
            'inverse_power': 'inverse_power',
            'logit': 'logit',
            'probit': 'probit',
            'cloglog': 'cloglog',
            'exponential': 'power',  # May need special handling for alpha/lambda
            # Add other mappings as needed
        }
        
        family_name = family_map.get(family, 'gaussian')
        link_name = link_map.get(link, 'identity')
        
        # Set the family in the model builder
        self.model_builder.set_family(family_name, link_name)
        
        # Set additional parameters for specific link functions
        if self.model_specs.get('alpha'):
            self.model_builder.set_alpha(self.model_specs.get('alpha'))
        
        if self.model_specs.get('lambda'):
            self.model_builder.set_lambda(self.model_specs.get('lambda'))
        
        # Set additional parameters for specific error structures
        if self.model_specs.get('variance_power'):
            self.model_builder.set_variance_power(self.model_specs.get('variance_power'))
    
    def fit_model(self):
        """Fit the GLM model with the currently selected predictors and configuration"""
        if self.data is None or not self.model_specs:
            QMessageBox.warning(self, "Cannot Fit Model", 
                               "Please load data and specify a model first.")
            return
            
        # Get response variable from model specs
        response = self.model_specs.get('response_variable')
        if not response:
            QMessageBox.warning(self, "Cannot Fit Model", 
                               "Please specify a response variable in Model Specification dialog.")
            return
            
        # Get all predictor terms from the sidebar
        terms = self.predictors_sidebar.get_model_terms()
        
        if not terms:
            QMessageBox.warning(self, "Cannot Fit Model", 
                               "Please select at least one predictor variable from the sidebar.\n"
                               "Right-click a variable and choose 'Add as Factor' or 'Add as Variate'.")
            return
            
        # Build the formula: response ~ term1 + term2 + ... + termN
        formula = f"{response} ~ {' + '.join(terms)}"
        
        try:
            # First remove any existing prediction columns to avoid duplicates
            cols_to_drop = [col for col in self.data.columns if col.startswith('_predicted')]
            if cols_to_drop:
                self.data = self.data.drop(columns=cols_to_drop)
            
            # Configure the model builder
            self.model_builder.load_data(self.data)
            self.model_builder.set_formula(formula)
            
            # Set family and link if specified
            family = self.model_specs.get('family')
            link = self.model_specs.get('link')
            
            # Map UI names to statsmodels family and link
            family_map = {
                "Normal": "gaussian",
                "Poisson": "poisson",
                "Gamma": "gamma",
                "Binomial": "binomial",
                "Negative Binomial": "negative_binomial",  # May need custom handling
                "Tweedie": "tweedie",  # May need custom handling
                "Multinomial": "multinomial"  # May need custom handling
            }
            
            link_map = {
                "Identity": "identity",
                "Log": "log",
                "Reciprocal": "inverse_power",
                "Logit": "logit",
                "Probit": "probit",
                "Complementary Log-Log": "cloglog",
                "Exponential": "power"  # May need parameters
            }
            
            if family in family_map and link in link_map:
                try:
                    self.model_builder.set_family(family_map[family], link_map[link])
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Warning: {e}. Using default Gaussian family with Identity link.")
                    self.model_builder.set_family("gaussian", "identity")
            
            # Set weights if specified
            weight_column = self.model_specs.get('weight_column')
            if weight_column and weight_column in self.data.columns:
                self.model_builder.set_weights(self.data[weight_column])
            
            # Make sure split column and train partition are applied if specified
            split_column = self.model_specs.get('split_column')
            train_partition = self.model_specs.get('train_partition')
            if split_column and train_partition:
                self.model_builder.set_training_data(split_column, train_partition)
            
            # Build the model - it will use training_data if available
            self.model_builder.build_model()
            
            results = self.model_builder.fit_model()
            
            # Update the model summary statistics table
            self.update_model_stats_table()
            
            # Generate predictions
            predictions = self.model_builder.predict()

            # Add predictions to the data for chart updates
            self.data['_predicted'] = predictions
            
            # Update chart with predictions if a predictor is selected
            if self.predictors_sidebar.selected_predictor:
                self.update_predictor_chart(self.predictors_sidebar.selected_predictor)
            
            # Show a success message
            self.statusBar.showMessage("Model fitted successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Fitting Model", f"An error occurred: {str(e)}")
    
    def update_model_stats_table(self):
        """Update the model summary statistics table with the latest model results"""
        if not self.model_builder.results:
            return
        
        # Get the model results
        results = self.model_builder.results
        
        # Update fitted parameters count
        num_params = len(results.params)
        self.model_stats_table.setItem(0, 0, QTableWidgetItem(str(num_params)))
        
        # Update deviance with comma formatting
        deviance = round(results.deviance, 4)
        formatted_deviance = f"{deviance:,.4f}"
        self.model_stats_table.setItem(1, 0, QTableWidgetItem(formatted_deviance))
        
        # Update Chi-Squared percentage
        # Calculate pearson chi2 divided by degrees of freedom
        chi2_pct = round((results.pearson_chi2 / results.df_resid) * 100, 2)
        self.model_stats_table.setItem(2, 0, QTableWidgetItem(f"{chi2_pct}%"))
        
        # Update AICc (corrected AIC for small sample sizes) with comma formatting
        # AICc = AIC + (2k(k+1))/(n-k-1) where k is num params and n is sample size
        k = num_params
        n = results.nobs
        aic = results.aic
        if n > k + 1:  # Avoid division by zero
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc = aic
        
        formatted_aicc = f"{aicc:,.2f}"
        self.model_stats_table.setItem(3, 0, QTableWidgetItem(formatted_aicc))
    
    def update_predictor_chart(self, predictor_column):
        """
        Update the chart in the bottom panel based on the selected predictor
        
        Parameters
        ----------
        predictor_column : str
            The name of the selected predictor column
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Updating chart for predictor: {predictor_column}")
        
        # Clear the current chart
        self.chart_canvas.axes.clear()
        self.chart_canvas.count_axes.clear()
        
        # Only proceed if we have data and a valid predictor column
        if self.data is None or predictor_column not in self.data.columns:
            self.chart_canvas.clear_plot()
            return
            
        # Get model specifications
        response = None
        weights = None
        
        if self.model_specs:
            response = self.model_specs.get('response_variable')
            weights = self.model_specs.get('weight_column')
        
        # Only create chart if we have a response variable
        if response is None or response not in self.data.columns:
            self.chart_canvas.clear_plot()
            return
            
        # Block signals to avoid recursive updates while setting spinner values
        self.min_value_spinner.blockSignals(True)
        self.max_value_spinner.blockSignals(True)
        self.intervals_spinner.blockSignals(True)
        
        # Set proper ranges for the controls if this is a numeric column
        if pd.api.types.is_numeric_dtype(self.data[predictor_column]):
            # Get binning settings from stored or default values for this predictor
            if predictor_column in self.factor_ranges:
                # Use stored values
                ranges = self.factor_ranges[predictor_column]
                self.min_value_spinner.setValue(ranges['min'])
                self.max_value_spinner.setValue(ranges['max'])
                self.intervals_spinner.setValue(ranges['intervals'])
            else:
                # Default values based on data range
                min_val = self.data[predictor_column].min()
                max_val = self.data[predictor_column].max()
                self.min_value_spinner.setValue(min_val)
                self.max_value_spinner.setValue(max_val)
                self.intervals_spinner.setValue(10)  # Default 10 intervals
                
                # Store these defaults
                self.factor_ranges[predictor_column] = {
                    'min': min_val,
                    'max': max_val,
                    'intervals': 10
                }
            
            # Unblock signals
            self.min_value_spinner.blockSignals(False)
            self.max_value_spinner.blockSignals(False)
            self.intervals_spinner.blockSignals(False)
        
        # Determine the data to use for visualization - prioritize predictor_data over training_data
        if hasattr(self.model_builder, 'predictor_data') and self.model_builder.predictor_data is not None:
            viz_data = self.model_builder.predictor_data.copy()
            viz_weights = self.model_builder.predictor_weights if hasattr(self.model_builder, 'predictor_weights') else None
            data_source = "predictor_data"
            logger.info(f"Using predictor_data for chart with {len(viz_data)} rows")
            logger.debug(f"predictor_data columns: {list(viz_data.columns)}")
        else:
            # If no filtered data available, use the full dataset
            viz_data = self.data.copy()
            viz_weights = self.model_builder.weights if hasattr(self.model_builder, 'weights') else None
            data_source = "full_data"
            logger.info(f"Using full dataset for chart with {len(viz_data)} rows")
            logger.debug(f"full_data columns: {list(viz_data.columns)}")
        
        # Log information about weights being used
        if viz_weights is not None:
            weight_type = type(viz_weights).__name__
            weight_len = len(viz_weights) if hasattr(viz_weights, '__len__') else 'unknown'
            logger.info(f"Data weights: {weight_type}, {weight_len}")
        else:
            logger.info("No data weights available for levels table")
            
        # Update label to show data source
        if hasattr(self, 'data_source_label'):
            self.data_source_label.setText(f"Data Source: {data_source.replace('_', ' ').title()}")
        else:
            self.data_source_label = QLabel(f"Data Source: {data_source.replace('_', ' ').title()}")
            self.data_source_label.setStyleSheet("font-style: italic; color: #666;")
            self.top_right_panel.layout().insertWidget(0, self.data_source_label)
            
        # Add model predictions if we have a fitted model
        if hasattr(self.model_builder, 'results') and self.model_builder.results is not None:
            try:
                logger.info("Adding model predictions to visualization data")
                
                # Get raw predictions from the model
                predictions = self.model_builder.predict()
                
                if len(predictions) == len(viz_data):
                    # For frequency models (like insurance claims), we need to consider the exposure
                    exposure_column = self.model_specs.get('weight_column', None)
                    model_family = getattr(self.model_builder, 'family', None)
                    
                    # Check if this is a frequency model (e.g., Poisson with exposure/weights)
                    if model_family in ['poisson', 'negative_binomial', 'tweedie'] and exposure_column is not None and exposure_column in viz_data.columns:
                        logger.info(f"Adjusting predictions for frequency model with exposure column: {exposure_column}")
                        
                        # The raw predictions represent claim frequency per unit exposure
                        # For visualization, we need to ensure this is properly reflected
                        viz_data['_predicted'] = predictions
                        
                        # Log for verification purposes
                        logger.info(f"Prediction mean: {predictions.mean():.4f}, min: {predictions.min():.4f}, max: {predictions.max():.4f}")
                    else:
                        # For other model types, use raw predictions
                        viz_data['_predicted'] = predictions
                    
                    logger.info(f"Added {len(predictions)} predictions to visualization data")
                else:
                    logger.warning(f"Prediction length ({len(predictions)}) doesn't match data length ({len(viz_data)})")
            except Exception as e:
                logger.error(f"Error adding predictions to visualization data: {str(e)}")
        
        # Get custom binning parameters if available
        custom_bins = None
        if pd.api.types.is_numeric_dtype(self.data[predictor_column]):
            # Get the min/max/intervals from the spinners
            min_val = self.min_value_spinner.value()
            max_val = self.max_value_spinner.value()
            n_bins = self.intervals_spinner.value()
            
            # Ensure valid binning parameters
            if min_val >= max_val or n_bins < 2:
                # Use default bin range based on data
                min_val = viz_data[predictor_column].min()
                max_val = viz_data[predictor_column].max()
                n_bins = 10
                logger.warning(f"Invalid binning parameters, using defaults: min={min_val}, max={max_val}, bins={n_bins}")
            
            # Create bins
            bins = pd.cut(viz_data[predictor_column], 
                        bins=np.linspace(min_val, max_val, n_bins + 1),
                        include_lowest=True)
            
            # Update stored values for this predictor
            self.factor_ranges[predictor_column] = {
                'min': min_val,
                'max': max_val,
                'intervals': n_bins
            }
            
            custom_bins = {'min': min_val, 'max': max_val, 'n_bins': n_bins}
            logger.info(f"Using custom binning for chart: min={min_val}, max={max_val}, bins={n_bins}")
        
        # Create the dual-axis chart with explicit weights argument
        logger.info(f"Creating chart with predictor={predictor_column}, response={response}")
        self.chart_canvas.create_dual_axis_chart(
            data=viz_data, 
            predictor=predictor_column, 
            response=response,
            weights=viz_weights,  # Pass weights directly as numpy array if available
            custom_bins=custom_bins
        )
        
        # Update parameter statistics for this predictor
        self.update_parameter_stats(predictor_column)
        
        # Update levels table for this predictor
        self.update_levels_table(predictor_column, response, weights)
        
        # Store the selected predictor for reference
        self.predictors_sidebar.selected_predictor = predictor_column
            
        # Get weights column if specified
        weights = self.model_specs.get('weight_column')
        
        # Determine if the predictor is continuous
        is_continuous = pd.api.types.is_numeric_dtype(self.data[predictor_column])
        
        # Show or hide continuous controls depending on predictor type
        self.continuous_controls.setVisible(is_continuous)
    
    def update_parameter_stats(self, predictor_column):
        """
        Update the parameter statistics panel for the selected predictor
        
        Parameters
        ----------
        predictor_column : str
            The name of the selected predictor column
        """
        # Determine if the predictor is continuous
        if self.data is None or predictor_column not in self.data.columns:
            # Hide widgets and exit
            self.continuous_param_widget.setVisible(False)
            self.categorical_param_widget.setVisible(True)
            self.categorical_label.setText("No data available")
            self.category_params_table.setRowCount(0)
            return
            
        is_continuous = pd.api.types.is_numeric_dtype(self.data[predictor_column])
        
        # Always use the table view (categorical widget) for all factors
        self.continuous_param_widget.setVisible(False)
        self.categorical_param_widget.setVisible(True)
        
        # Clear category table
        self.category_params_table.setRowCount(0)
        
        # Check if model is fitted
        if not hasattr(self.model_builder, 'results') or self.model_builder.results is None:
            self.categorical_label.setText(f"{predictor_column} (not fitted)")
            return
        
        # Get parameter info
        results = self.model_builder.results
        
        if is_continuous:
            # For continuous variables, check for the base parameter and any powers
            self.categorical_label.setText(f"{predictor_column} Parameters:")
            continuous_terms = []
            
            # Check for the direct parameter name (linear term)
            if predictor_column in results.params:
                value = results.params[predictor_column]
                se = results.bse[predictor_column] if predictor_column in results.bse else None
                se_pct = None
                if se is not None and value != 0:
                    se_pct = (se / abs(value)) * 100
                
                continuous_terms.append({
                    'term': 'Linear',
                    'param_name': predictor_column,
                    'value': value,
                    'se': se,
                    'se_pct': se_pct
                })
            
            # Look for power terms using multiple possible patterns
            # Pattern 1: predictor^2, predictor^3
            power_pattern = f"{predictor_column}^"
            # Pattern 2: predictor:predictor (squared term)
            squared_pattern = f"{predictor_column}:{predictor_column}"
            # Pattern 3: np.power(predictor, 2), np.power(predictor, 3)
            np_power_pattern = f"np.power({predictor_column}"
            # Pattern 4: I(predictor**2), I(predictor**3)
            i_power_pattern = f"I({predictor_column}**"
            # Pattern 5: poly(predictor, 2)[1], poly(predictor, 3)[2]
            poly_pattern = f"poly({predictor_column}"
            
            # Search through all parameters for matching patterns
            for param_name in results.params.index:
                power = None
                
                # Check pattern 1: predictor^2, predictor^3
                if power_pattern in param_name:
                    try:
                        power = int(param_name.split(power_pattern, 1)[1])
                    except (ValueError, IndexError):
                        pass
                
                # Check pattern 2: predictor:predictor (squared term)
                elif param_name == squared_pattern:
                    power = 2
                
                # Check pattern 3: np.power
                elif np_power_pattern in param_name:
                    try:
                        # Extract power from np.power(predictor, X)
                        power_text = param_name.split(np_power_pattern, 1)[1]
                        power_text = power_text.split(',', 1)[1].strip()
                        power_text = power_text.split(')', 1)[0].strip()
                        power = int(power_text)
                    except (ValueError, IndexError):
                        pass
                
                # Check pattern 4: I(predictor**X)
                elif i_power_pattern in param_name:
                    try:
                        power_text = param_name.split(i_power_pattern, 1)[1]
                        power = int(power_text.split(')', 1)[0])
                    except (ValueError, IndexError):
                        pass
                
                # Check pattern 5: poly(predictor, X)[Y]
                elif poly_pattern in param_name:
                    try:
                        # For polynomial terms, we need both the polynomial degree and the term index
                        # For example, poly(x, 3)[2] represents the 3rd order term (cubic)
                        degree_text = param_name.split(poly_pattern, 1)[1]
                        degree_text = degree_text.split(',', 1)[1].strip()
                        degree_text = degree_text.split(')', 1)[0].strip()
                        
                        term_index_text = param_name.split(']', 1)[0].split('[', 1)[1]
                        term_index = int(term_index_text)
                        
                        # Term index is 0-based, so add 1 to get the power 
                        power = term_index + 1
                    except (ValueError, IndexError):
                        pass
                
                # If we identified a power, add this term to our list
                if power is not None and power > 1:  # Skip power 1 (linear) as it's handled separately
                    power_name = {2: 'Quadratic', 3: 'Cubic'}.get(power, f'Power {power}')
                    
                    value = results.params[param_name]
                    se = results.bse[param_name] if param_name in results.bse else None
                    se_pct = None
                    if se is not None and value != 0:
                        se_pct = (se / abs(value)) * 100
                    
                    continuous_terms.append({
                        'term': power_name,
                        'param_name': param_name,
                        'value': value,
                        'se': se,
                        'se_pct': se_pct
                    })
            
            # Debug: Print to console what we found
            logger = logging.getLogger(__name__)
            logger.debug(f"Found {len(continuous_terms)} terms for {predictor_column}:")
            for term in continuous_terms:
                logger.debug(f"  {term['term']} ({term['param_name']}): {term['value']:.4f}")
            
            # Sort terms by power (Linear first, then Quadratic, Cubic, etc.)
            term_order = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}
            continuous_terms.sort(key=lambda x: term_order.get(x['term'], 
                                                  int(x['term'].split(' ')[1]) if ' ' in x['term'] else 99))
            
            # Set up the table
            self.category_params_table.setRowCount(len(continuous_terms))
            
            for i, term in enumerate(continuous_terms):
                # Order/Term
                term_item = QTableWidgetItem(term['term'])
                self.category_params_table.setItem(i, 0, term_item)
                
                # Value
                value_item = QTableWidgetItem(f"{term['value']:.4f}")
                self.category_params_table.setItem(i, 1, value_item)
                
                # SE%
                if term['se_pct'] is not None:
                    se_pct_item = QTableWidgetItem(f"{term['se_pct']:.2f}%")
                    
                    # Color-code SE% based on value
                    if term['se_pct'] < 50:
                        se_pct_item.setForeground(Qt.GlobalColor.darkGreen)
                    elif 50 <= term['se_pct'] <= 60:
                        se_pct_item.setForeground(Qt.GlobalColor.darkGray)
                    else:  # > 60%
                        se_pct_item.setForeground(Qt.GlobalColor.red)
                else:
                    se_pct_item = QTableWidgetItem("N/A")
                
                self.category_params_table.setItem(i, 2, se_pct_item)
        else:
            # For categorical variables, find all related parameters
            self.categorical_label.setText(f"{predictor_column} Parameters:")
            
            # Different pattern formats that statsmodels might use
            patterns = [
                f"{predictor_column}[T.",            # Most common
                f"{predictor_column}_",              # Alternative
                f"C({predictor_column})[T.",         # When using C() in formula
                f"C({predictor_column})_"            # Another alternative
            ]
            
            # Find all parameters related to this predictor
            level_params = []
            for pattern in patterns:
                for param_name in results.params.index:
                    if pattern in param_name:
                        # Extract level name - everything after the pattern until closing bracket or end
                        if ']' in param_name:
                            level = param_name.split(pattern, 1)[1].split(']', 1)[0]
                        else:
                            level = param_name.split(pattern, 1)[1]
                            
                        value = results.params[param_name]
                        se = results.bse[param_name] if param_name in results.bse else None
                        
                        # Calculate SE%
                        se_pct = None
                        if se is not None and value != 0:
                            se_pct = (se / abs(value)) * 100
                            
                        level_params.append({
                            'level': level,
                            'value': value,
                            'se': se,
                            'se_pct': se_pct
                        })
            
            # Add rows to the table
            self.category_params_table.setRowCount(len(level_params))
            
            for i, param in enumerate(level_params):
                # Level
                level_item = QTableWidgetItem(str(param['level']))
                self.category_params_table.setItem(i, 0, level_item)
                
                # Value
                value_item = QTableWidgetItem(f"{param['value']:.4f}")
                self.category_params_table.setItem(i, 1, value_item)
                
                # SE%
                if param['se_pct'] is not None:
                    se_pct_item = QTableWidgetItem(f"{param['se_pct']:.2f}%")
                    
                    # Color-code SE% based on value
                    if param['se_pct'] < 50:
                        se_pct_item.setForeground(Qt.GlobalColor.green)
                    elif 50 <= param['se_pct'] <= 60:
                        se_pct_item.setForeground(Qt.GlobalColor.gray)
                    else:  # > 60%
                        se_pct_item.setForeground(Qt.GlobalColor.red)
                else:
                    se_pct_item = QTableWidgetItem("N/A")
                self.category_params_table.setItem(i, 2, se_pct_item)
    
    def update_levels_table(self, predictor_column, response, weights=None):
        """
        Update the table showing information for each level of the selected factor
        
        Parameters
        ----------
        predictor_column : str
            The name of the selected predictor column
        response : str
            The name of the response column
        weights : str, optional
            The name of the weights column
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Updating levels table for predictor: {predictor_column}")
        
        # Clear the current table content
        self.levels_table.clearContents()
        self.levels_table.setRowCount(0)
        
        # Make sure we have valid data and columns
        if (self.data is None or 
            predictor_column not in self.data.columns or 
            response not in self.data.columns):
            logger.warning(f"Missing required data or columns for levels table")
            return
        
        # Always use training_data as first choice for the levels table
        if hasattr(self.model_builder, 'training_data') and self.model_builder.training_data is not None:
            data = self.model_builder.training_data
            data_weights = self.model_builder.training_weights if hasattr(self.model_builder, 'training_weights') else None
            data_source = "training_data"
            logger.info(f"Using training_data for levels table with {len(data)} rows")
        else:
            # If no training data available, use the full dataset
            data = self.data
            data_weights = self.model_builder.weights if hasattr(self.model_builder, 'weights') else None
            data_source = "full_data"
            logger.info(f"Using full dataset for levels table with {len(data)} rows")
        
        # Log information about weights being used
        if data_weights is not None:
            weight_type = type(data_weights).__name__
            weight_len = len(data_weights) if hasattr(data_weights, '__len__') else 'unknown'
            logger.info(f"Data weights: {weight_type}, {weight_len}")
        else:
            logger.info("No data weights available for levels table")
            
        # Update label to show data source
        if hasattr(self, 'data_source_label'):
            self.data_source_label.setText(f"Data Source: {data_source.replace('_', ' ').title()}")
        
        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(data[predictor_column])
        
        # Create levels based on column type
        if is_numeric:
            # Get binning settings from custom spinner values
            min_val = self.min_value_spinner.value()
            max_val = self.max_value_spinner.value()
            n_bins = self.intervals_spinner.value()
            
            # Ensure valid binning parameters
            if min_val >= max_val or n_bins < 2:
                # Use default bin range based on data
                min_val = data[predictor_column].min()
                max_val = data[predictor_column].max()
                n_bins = 10
                logger.warning(f"Invalid binning parameters, using defaults: min={min_val}, max={max_val}, bins={n_bins}")
            
            # Create bins
            bins = pd.cut(data[predictor_column], 
                        bins=np.linspace(min_val, max_val, n_bins + 1),
                        include_lowest=True)
            
            if weights is not None and weights in data.columns:
                # Calculate counts weighted by weight column
                level_groups = data.groupby(bins)
                bin_counts = level_groups[weights].sum()
                bin_response = level_groups[response].sum() / level_groups[weights].sum()
            elif data_weights is not None and len(data_weights) == len(data):
                # Use the filtered weights directly
                temp_df = data.copy()
                temp_df['_weights'] = data_weights
                level_groups = temp_df.groupby(bins)
                bin_counts = level_groups['_weights'].sum()
                bin_response = level_groups[response].sum() / level_groups['_weights'].sum()
            else:
                # Calculate counts without weights
                level_groups = data.groupby(bins)
                bin_counts = level_groups.size()
                bin_response = level_groups[response].mean()
            
            # Convert bins to strings for table
            levels = [str(b) for b in bin_counts.index]
        else:
            # For categorical variables, group by unique values
            if weights is not None and weights in data.columns:
                # Calculate counts weighted by weight column
                level_groups = data.groupby(predictor_column)
                bin_counts = level_groups[weights].sum()
                bin_response = level_groups[response].sum() / level_groups[weights].sum()
            elif data_weights is not None and len(data_weights) == len(data):
                # Use the filtered weights directly
                temp_df = data.copy()
                temp_df['_weights'] = data_weights
                level_groups = temp_df.groupby(predictor_column)
                bin_counts = level_groups['_weights'].sum()
                bin_response = level_groups[response].sum() / level_groups['_weights'].sum()
            else:
                # Calculate counts without weights
                level_groups = data.groupby(predictor_column)
                bin_counts = level_groups.size()
                bin_response = level_groups[response].mean()
            
            # Convert to strings for consistency
            levels = [str(l) for l in bin_counts.index]
        
        # Set up table with the correct number of rows
        self.levels_table.setRowCount(len(levels))
        
        # Fill the table
        for row, level in enumerate(levels):
            # Level name
            level_item = QTableWidgetItem(level)
            self.levels_table.setItem(row, 0, level_item)
            
            # Number of observations
            count = bin_counts.iloc[row]
            count_item = QTableWidgetItem(f"{count:.0f}")
            self.levels_table.setItem(row, 1, count_item)
            
            # Weight (only if we have weights)
            if weights is not None or data_weights is not None:
                weight_item = QTableWidgetItem(f"{count:.2f}")
                self.levels_table.setItem(row, 2, weight_item)
            else:
                self.levels_table.setItem(row, 2, QTableWidgetItem("-"))
            
            # Response
            if row < len(bin_response):
                resp_value = bin_response.iloc[row]
                resp_item = QTableWidgetItem(f"{resp_value:.4f}")
                self.levels_table.setItem(row, 3, resp_item)
                
        # Resize columns to content
        self.levels_table.resizeColumnsToContents()
    
    def update_chart_with_new_ranges(self):
        """Update the chart using the custom ranges specified in the spinners"""
        if hasattr(self.predictors_sidebar, 'selected_predictor'):
            predictor = self.predictors_sidebar.selected_predictor
            
            # Store the current values for this predictor
            if predictor is not None:
                self.factor_ranges[predictor] = {
                    'min': self.min_value_spinner.value(),
                    'max': self.max_value_spinner.value(),
                    'intervals': self.intervals_spinner.value()
                }
            
            # Update the chart
            self.update_predictor_chart(predictor)
    
    def _save_model(self):
        """Save the current model to a file"""
        if not hasattr(self.model_builder, 'results') or self.model_builder.results is None:
            QMessageBox.warning(self, "No Model", "You must fit a model before saving it.")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Model", 
            "", 
            "Embezzle Model Files (*.emdl);;All Files (*)"
        )
        
        if file_name:
            # Add .emdl extension if not provided
            if not file_name.endswith('.emdl'):
                file_name += '.emdl'
                
            try:
                success = self.model_builder.save_model(file_name)
                if success:
                    self.statusBar.showMessage(f"Model saved to {file_name}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to save model.")
                    self.statusBar.showMessage("Error saving model")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
                self.statusBar.showMessage("Error saving model")

    def _load_model(self):
        """Load a model from a file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Model", 
            "", 
            "Embezzle Model Files (*.emdl);;All Files (*)"
        )
        
        if file_name:
            try:
                success = self.model_builder.load_model(file_name)
                if success:
                    # Get the data from the model builder
                    self.data = self.model_builder.data
                    
                    # Extract model specs from the loaded model
                    family_name = type(self.model_builder.family).__name__
                    link_name = type(self.model_builder.family.link).__name__
                    
                    # Extract formula components
                    formula_parts = self.model_builder.formula.split('~')
                    if len(formula_parts) == 2:
                        response = formula_parts[0].strip()
                        predictors = formula_parts[1].strip()
                    else:
                        response = ""
                        predictors = ""
                    
                    # Update model specs
                    self.model_specs = {
                        'response_variable': response,
                        'predictors': predictors,
                        'error_family': family_name,
                        'link_function': link_name
                    }
                    
                    # Check if weights were used
                    if self.model_builder.weights is not None:
                        # Try to determine weight column by checking data columns
                        # This is a best-effort approach
                        for col in self.data.columns:
                            if hasattr(self.model_builder.weights, 'name') and col == self.model_builder.weights.name:
                                self.model_specs['weight_column'] = col
                                break
                    
                    # Switch to the modeling interface if not already shown
                    if self.welcome_panel.isVisible():
                        self.welcome_panel.hide()
                        self.main_splitter.show()
                        self.predictors_sidebar.show()
                    
                    # Build list of columns to exclude from predictor sidebar
                    excluded_columns = [self.model_specs.get('response_variable')]
                    if 'weight_column' in self.model_specs:
                        excluded_columns.append(self.model_specs.get('weight_column'))
                    
                    # Exclude prediction columns (starting with '_predicted')
                    for col in self.data.columns:
                        if col.startswith('_predicted'):
                            excluded_columns.append(col)
                    
                    # Update the predictor sidebar with the loaded data
                    self.predictors_sidebar.update_predictors(self.data, excluded_columns)
                    
                    # Restore predictor roles and polynomial terms
                    if hasattr(self.model_builder, 'saved_predictor_roles'):
                        for predictor, role in self.model_builder.saved_predictor_roles.items():
                            if predictor in self.predictors_sidebar.predictor_items:
                                self.predictors_sidebar.set_predictor_role(predictor, role)
                                
                                # Also restore the polynomial terms for variates
                                if (role == 'variate' and 
                                    hasattr(self.model_builder, 'saved_polynomial_terms') and 
                                    predictor in self.model_builder.saved_polynomial_terms):
                                    
                                    widget = self.predictors_sidebar.predictor_items[predictor]
                                    saved_terms = self.model_builder.saved_polynomial_terms[predictor]
                                    
                                    # Update the checkbox state to match the saved state
                                    for power, checked in saved_terms.items():
                                        if power in widget.term_checkboxes:
                                            widget.term_checkboxes[power].setChecked(checked)
                                            widget.polynomial_terms[power] = checked
                    
                    # Update model statistics
                    self.update_model_stats_table()
                    
                    self.statusBar.showMessage(f"Model loaded from {file_name}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to load model.")
                    self.statusBar.showMessage("Error loading model")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                self.statusBar.showMessage("Error loading model")

    def update_prediction_data(self):
        """Update the model builder's prediction data based on the selected prediction set"""
        logger = logging.getLogger(__name__)
        logger.debug("update_prediction_data method called")
        
        # Check if we have data and model specs
        if self.data is not None and self.model_specs is not None:
            logger.debug("Data and model specs are available")
            
            # Get the split column from model specs
            split_column = self.model_specs.get('split_column')
            if split_column:
                logger.debug(f"Split column from model_specs: {split_column}")
                
                try:
                    # Get the selected prediction set from sidebar
                    prediction_partition = self.predictors_sidebar.get_selected_prediction_set()
                    logger.debug(f"Selected prediction partition: {prediction_partition}")
                    
                    # Convert None to 'all' for the set_prediction_data method
                    if prediction_partition is None:
                        prediction_partition = 'all'
                        logger.debug("Converted None to 'all' for prediction partition")
                    
                    # Update the prediction data in the model builder
                    logger.debug(f"Calling model_builder.set_prediction_data({split_column}, {prediction_partition})")
                    self.model_builder.set_prediction_data(split_column, prediction_partition)
                    
                    # Update status bar message
                    if prediction_partition == 'all':
                        self.statusBar.showMessage(f"Using all data for predictions")
                    else:
                        self.statusBar.showMessage(f"Using data filtered by {split_column}={prediction_partition} for predictions")
                    
                    # If we have a fitted model, update visualizations
                    if hasattr(self.predictors_sidebar, 'selected_predictor'):
                        predictor = self.predictors_sidebar.selected_predictor
                        if predictor:
                            logger.debug(f"Updating chart for predictor: {predictor}")
                            
                            # Force refresh the chart with the new prediction data
                            self.update_predictor_chart(predictor)
                            
                            # Get the response variable to update the levels table
                            response = self.model_specs.get('response_variable')
                            weights = self.model_specs.get('weight_column')
                            
                            # Force refresh the levels table with the new prediction data
                            if response:
                                logger.debug(f"Updating levels table for predictor: {predictor}")
                                self.update_levels_table(predictor, response, weights)
                except Exception as e:
                    self.statusBar.showMessage(f"Error setting prediction data: {str(e)}")
                    logger.error(f"Prediction data error: {str(e)}")
            else:
                logger.debug("Data or model_specs is None")


def run_app():
    """
    Run the Embezzle application
    """
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Embezzle application")
    
    # Create the application
    app = QApplication(sys.argv)
    main_window = EmbezzleMainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
