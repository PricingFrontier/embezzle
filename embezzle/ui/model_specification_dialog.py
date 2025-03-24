"""
Model specification dialog module for Embezzle

This module provides a dialog for specifying GLM model parameters, including
error families, link functions, and other configuration options.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QRadioButton, 
    QGroupBox, QDialogButtonBox, QLineEdit, QButtonGroup, QHBoxLayout
)
from PyQt6.QtCore import Qt


class ModelSpecificationDialog(QDialog):
    """Dialog for specifying model parameters"""
    
    def __init__(self, parent, columns):
        super().__init__(parent)
        self.setWindowTitle("Model Specification")
        self.setMinimumWidth(400)
        
        # Store parent reference for access to model_specs
        self.parent = parent
        
        # Create the layout
        layout = QVBoxLayout(self)
        
        # Create form layout for the fields
        form_layout = QFormLayout()
        
        # Response variable
        self.response_combo = QComboBox()
        self.response_combo.addItem("(None)")
        self._add_columns_to_combo(self.response_combo, columns)
        form_layout.addRow("Response Variable:", self.response_combo)
        
        # Weight column
        self.weight_combo = QComboBox()
        self.weight_combo.addItem("(None)")
        self._add_columns_to_combo(self.weight_combo, columns)
        form_layout.addRow("Weight Column:", self.weight_combo)
        
        # Split column
        self.split_combo = QComboBox()
        self.split_combo.addItem("(None)")
        self._add_columns_to_combo(self.split_combo, columns)
        form_layout.addRow("Split Column:", self.split_combo)
        
        # Add the form layout to the main layout
        layout.addLayout(form_layout)
        
        # Create the link function group box
        layout.addWidget(self._create_link_group())
        
        # Create the error structure (family) group box
        layout.addWidget(self._create_error_group())
        
        # Add the button box
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Pre-populate fields if model_specs exist in parent
        self.pre_populate_fields()
        
    def pre_populate_fields(self):
        """Pre-populate fields with existing model_specs if available"""
        if not hasattr(self.parent, 'model_specs') or not self.parent.model_specs:
            return
            
        # Get current specs
        specs = self.parent.model_specs
        
        # Set response variable
        if 'response_variable' in specs and specs['response_variable']:
            index = self.response_combo.findText(specs['response_variable'])
            if index >= 0:
                self.response_combo.setCurrentIndex(index)
                
        # Set weight column
        if 'weight_column' in specs and specs['weight_column']:
            index = self.weight_combo.findText(specs['weight_column'])
            if index >= 0:
                self.weight_combo.setCurrentIndex(index)
                
        # Set split column
        if 'split_column' in specs and specs['split_column']:
            index = self.split_combo.findText(specs['split_column'])
            if index >= 0:
                self.split_combo.setCurrentIndex(index)
                
        # Set family/error structure
        if 'family' in specs:
            family = specs['family']
            if family == 'gaussian':
                self.error_normal.setChecked(True)
            elif family == 'poisson':
                self.error_poisson.setChecked(True)
            elif family == 'gamma':
                self.error_gamma.setChecked(True)
            elif family == 'binomial':
                self.error_binomial.setChecked(True)
            elif family == 'negativebinomial':
                self.error_negbin.setChecked(True)
            elif family == 'tweedie':
                self.error_tweedie.setChecked(True)
            elif family == 'multinomial':
                self.error_multinomial.setChecked(True)
                
        # Set link
        if 'link' in specs:
            link = specs['link']
            if link == 'identity':
                self.link_identity.setChecked(True)
            elif link == 'log':
                self.link_log.setChecked(True)
            elif link == 'inverse_power':
                self.link_reciprocal.setChecked(True)
            elif link == 'logit':
                self.link_logit.setChecked(True)
            elif link == 'probit':
                self.link_probit.setChecked(True)
            elif link == 'cloglog':
                self.link_cloglog.setChecked(True)
            elif link == 'exponential':
                self.link_exponential.setChecked(True)
                
    def _add_columns_to_combo(self, combo, columns):
        """Add columns to a combo box"""
        for column in columns:
            combo.addItem(column)
            
    def _create_error_group(self):
        """Create the error structure (family) group box"""
        error_group = QGroupBox("Error Structure")
        error_layout = QVBoxLayout()
        
        self.error_group = QButtonGroup(self)
        
        # Create error structure radio buttons
        self.error_normal = QRadioButton("Normal (Gaussian)")
        self.error_poisson = QRadioButton("Poisson")
        self.error_gamma = QRadioButton("Gamma")
        self.error_binomial = QRadioButton("Binomial")
        self.error_negbin = QRadioButton("Negative Binomial")
        self.error_tweedie = QRadioButton("User Defined (Tweedie)")
        self.error_multinomial = QRadioButton("Multinomial")
        
        # Add buttons to group for mutual exclusivity
        self.error_group.addButton(self.error_normal)
        self.error_group.addButton(self.error_poisson)
        self.error_group.addButton(self.error_gamma)
        self.error_group.addButton(self.error_binomial)
        self.error_group.addButton(self.error_negbin)
        self.error_group.addButton(self.error_tweedie)
        self.error_group.addButton(self.error_multinomial)
        
        # Add to layout
        error_layout.addWidget(self.error_normal)
        error_layout.addWidget(self.error_poisson)
        error_layout.addWidget(self.error_gamma)
        error_layout.addWidget(self.error_binomial)
        error_layout.addWidget(self.error_negbin)
        
        # Tweedie parameters
        tweedie_layout = QHBoxLayout()
        tweedie_layout.addWidget(self.error_tweedie)
        
        variance_layout = QFormLayout()
        self.variance_input = QLineEdit()
        variance_layout.addRow("Variance Power:", self.variance_input)
        
        tweedie_layout.addLayout(variance_layout)
        error_layout.addLayout(tweedie_layout)
        
        error_layout.addWidget(self.error_multinomial)
        
        # Set default
        self.error_normal.setChecked(True)
        
        error_group.setLayout(error_layout)
        return error_group
        
    def _create_link_group(self):
        """Create the link function group box"""
        link_group = QGroupBox("Link Function")
        link_layout = QVBoxLayout()
        
        self.link_group = QButtonGroup(self)
        
        # Create link function radio buttons
        self.link_identity = QRadioButton("Identity")
        self.link_log = QRadioButton("Log")
        self.link_reciprocal = QRadioButton("Reciprocal (Inverse)")
        self.link_logit = QRadioButton("Logit")
        self.link_probit = QRadioButton("Probit")
        self.link_cloglog = QRadioButton("Complementary Log-Log")
        self.link_exponential = QRadioButton("Exponential")
        
        # Add buttons to group for mutual exclusivity
        self.link_group.addButton(self.link_identity)
        self.link_group.addButton(self.link_log)
        self.link_group.addButton(self.link_reciprocal)
        self.link_group.addButton(self.link_logit)
        self.link_group.addButton(self.link_probit)
        self.link_group.addButton(self.link_cloglog)
        self.link_group.addButton(self.link_exponential)
        
        # Add to layout
        link_layout.addWidget(self.link_identity)
        link_layout.addWidget(self.link_log)
        link_layout.addWidget(self.link_reciprocal)
        link_layout.addWidget(self.link_logit)
        link_layout.addWidget(self.link_probit)
        link_layout.addWidget(self.link_cloglog)
        
        # Exponential parameters
        exponential_layout = QHBoxLayout()
        exponential_layout.addWidget(self.link_exponential)
        
        param_layout = QFormLayout()
        self.alpha_input = QLineEdit()
        self.lambda_input = QLineEdit()
        param_layout.addRow("Alpha:", self.alpha_input)
        param_layout.addRow("Lambda:", self.lambda_input)
        
        exponential_layout.addLayout(param_layout)
        link_layout.addLayout(exponential_layout)
        
        # Set default
        self.link_identity.setChecked(True)
        
        link_group.setLayout(link_layout)
        return link_group
        
    def get_model_specs(self):
        """Get the model specifications from the dialog"""
        specs = {
            'response_variable': self.response_combo.currentText() if self.response_combo.currentText() != "(None)" else None,
            'weight_column': self.weight_combo.currentText() if self.weight_combo.currentText() != "(None)" else None,
            'split_column': self.split_combo.currentText() if self.split_combo.currentText() != "(None)" else None,
            'family': self._get_selected_family(),
            'link': self._get_selected_link()
        }
        
        # Add additional parameters for specific link functions
        if self.link_exponential.isChecked():
            try:
                specs['alpha'] = float(self.alpha_input.text()) if self.alpha_input.text() else None
                specs['lambda'] = float(self.lambda_input.text()) if self.lambda_input.text() else None
            except ValueError:
                # Handle invalid input
                pass
        
        # Add additional parameters for specific error structures
        if self.error_tweedie.isChecked():
            try:
                specs['variance_power'] = float(self.variance_input.text()) if self.variance_input.text() else None
            except ValueError:
                # Handle invalid input
                pass
        
        return specs
        
    def _get_selected_family(self):
        """Get the selected family/error structure"""
        if self.error_normal.isChecked():
            return "gaussian"
        elif self.error_poisson.isChecked():
            return "poisson"
        elif self.error_gamma.isChecked():
            return "gamma"
        elif self.error_binomial.isChecked():
            return "binomial"
        elif self.error_negbin.isChecked():
            return "negativebinomial"
        elif self.error_tweedie.isChecked():
            return "tweedie"
        elif self.error_multinomial.isChecked():
            return "multinomial"
        return "gaussian"  # Default
        
    def _get_selected_link(self):
        """Get the selected link function"""
        if self.link_identity.isChecked():
            return "identity"
        elif self.link_log.isChecked():
            return "log"
        elif self.link_reciprocal.isChecked():
            return "inverse_power"
        elif self.link_logit.isChecked():
            return "logit"
        elif self.link_probit.isChecked():
            return "probit"
        elif self.link_cloglog.isChecked():
            return "cloglog"
        elif self.link_exponential.isChecked():
            return "exponential"
        return "identity"  # Default
