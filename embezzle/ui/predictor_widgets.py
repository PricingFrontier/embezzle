"""
Predictor widget components for Embezzle

This module provides widgets for displaying and configuring predictor variables
in the Embezzle application.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, 
    QListWidget, QListWidgetItem, QPushButton, QMenu, QSizePolicy,
    QComboBox, QFormLayout
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction


class PredictorItem(QWidget):
    """Widget for a predictor item with expandable options"""
    def __init__(self, predictor_name, parent=None):
        super().__init__(parent)
        self.predictor_name = predictor_name
        self.is_expanded = False
        self.is_selected = False
        self.role = None  # None, 'factor', or 'variate'
        self.polynomial_terms = {1: True, 2: False, 3: False}  # Default: only X^1 selected
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Main row with name and indicators
        main_row = QHBoxLayout()
        
        # Predictor name
        self.name_label = QLabel(predictor_name)
        self.name_label.setStyleSheet("font-weight: bold;")
        
        # Role indicator (F for Factor, V for Variate)
        self.role_label = QLabel("")
        self.role_label.setStyleSheet("color: #555; font-style: italic;")
        
        main_row.addWidget(self.name_label)
        main_row.addStretch()
        main_row.addWidget(self.role_label)
        
        layout.addLayout(main_row)
        
        # Expandable options container (for variate polynomial terms)
        self.options_container = QWidget()
        options_layout = QVBoxLayout(self.options_container)
        options_layout.setContentsMargins(10, 0, 0, 0)
        
        # Create polynomial term checkboxes
        self.term_checkboxes = {}
        for power in [1, 2, 3]:
            checkbox = QCheckBox(f"X^{power}")
            checkbox.setChecked(power == 1)  # X^1 is checked by default
            checkbox.stateChanged.connect(lambda state, p=power: self.on_term_changed(p, state))
            self.term_checkboxes[power] = checkbox
            options_layout.addWidget(checkbox)
        
        self.options_container.setVisible(False)
        layout.addWidget(self.options_container)
        
        # Set fixed height initially
        self.setFixedHeight(30)
    
    def set_role(self, role):
        """Set the role of this predictor (factor or variate)"""
        self.role = role
        
        if role == 'factor':
            self.role_label.setText("F")
            self.options_container.setVisible(False)
            self.is_expanded = False
            self.setFixedHeight(30)
        elif role == 'variate':
            self.role_label.setText("V")
            self.options_container.setVisible(True)
            self.is_expanded = True
            self.setFixedHeight(90)  # Enough height for 3 checkboxes
        else:
            self.role_label.setText("")
            self.options_container.setVisible(False)
            self.is_expanded = False
            self.setFixedHeight(30)
    
    def on_term_changed(self, power, state):
        """Handle checkbox state changes for polynomial terms"""
        self.polynomial_terms[power] = (state == Qt.CheckState.Checked.value)
    
    def get_formula_terms(self):
        """Get the formula terms for this predictor based on its role and settings"""
        if self.role == 'factor':
            return [f"C({self.predictor_name})"]
        elif self.role == 'variate':
            terms = []
            for power, selected in self.polynomial_terms.items():
                if selected:
                    if power == 1:
                        terms.append(self.predictor_name)
                    else:
                        terms.append(f"np.power({self.predictor_name}, {power})")
            return terms
        return []
    
    def set_selected(self, selected):
        """Set whether this predictor is selected"""
        self.is_selected = selected
        
        # Update the visual style
        if selected:
            self.setStyleSheet("background-color: #e0e0e0; border-radius: 4px;")
        else:
            self.setStyleSheet("")


class PredictorsSidebar(QWidget):
    """
    Sidebar widget for displaying and configuring predictor variables.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the layout
        layout = QVBoxLayout(self)
        
        # Create form layout for the prediction set selection
        form_layout = QFormLayout()
        
        # Create a dropdown for prediction set selection (populated by split column values)
        self.prediction_set_combo = QComboBox()
        self.prediction_set_combo.addItem("All")
        form_layout.addRow("Prediction Set:", self.prediction_set_combo)
        
        # Add the form layout to the main layout
        layout.addLayout(form_layout)
        
        # Add Fit button after the prediction set combo
        self.fit_button = QPushButton("Fit")
        self.fit_button.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.fit_button)
        
        # Create scrollable list for predictor items
        self.predictors_list = QListWidget()
        self.predictors_list.setStyleSheet("""
            QListWidget::item { 
                border-bottom: 1px solid #ddd; 
                padding: 2px; 
            }
            QListWidget::item:selected { 
                background-color: #e0e0e0;
                border-radius: 4px;
                border: none;
            }
        """)
        self.predictors_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.predictors_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.predictors_list)
        
        # Set minimum width for the sidebar (20% of typical screen)
        self.setMinimumWidth(200)
        
        # Track the predictor items and currently selected item
        self.predictor_items = {}
        self.selected_predictor = None
        
        # Store predictor roles for persistence
        self.predictor_roles = {}
        
        # Signal callback for when a predictor is selected
        self.on_predictor_selected = None
        
        # Connect selection changed signal
        self.predictors_list.itemClicked.connect(self.handle_predictor_clicked)
    
    def update_prediction_set_combo(self, split_column, data=None):
        """Update the prediction set dropdown with values from the split column"""
        self.prediction_set_combo.clear()
        self.prediction_set_combo.addItem("All")
        
        if split_column and data is not None and split_column in data.columns:
            # Get unique values from the split column
            unique_values = data[split_column].unique()
            for value in unique_values:
                self.prediction_set_combo.addItem(str(value))
    
    def show_context_menu(self, position):
        """Show context menu for the predictor item"""
        item = self.predictors_list.itemAt(position)
        if not item:
            return
            
        # Get the predictor name
        list_item = self.predictors_list.itemFromIndex(self.predictors_list.indexAt(position))
        predictor_name = list_item.data(Qt.ItemDataRole.UserRole)
        
        # Create context menu
        context_menu = QMenu(self)
        
        # Add actions
        add_factor_action = QAction("Add as Factor", self)
        add_variate_action = QAction("Add as Variate", self)
        remove_action = QAction("Remove", self)
        
        # Connect actions to handlers
        add_factor_action.triggered.connect(lambda: self.set_predictor_role(predictor_name, 'factor'))
        add_variate_action.triggered.connect(lambda: self.set_predictor_role(predictor_name, 'variate'))
        remove_action.triggered.connect(lambda: self.set_predictor_role(predictor_name, None))
        
        # Add actions to menu
        context_menu.addAction(add_factor_action)
        context_menu.addAction(add_variate_action)
        context_menu.addSeparator()
        context_menu.addAction(remove_action)
        
        # Show the menu
        context_menu.exec(self.predictors_list.mapToGlobal(position))
    
    def set_predictor_role(self, predictor_name, role):
        """Set the role of a predictor (factor or variate)"""
        if predictor_name in self.predictor_items:
            # Set the visual role in the widget
            self.predictor_items[predictor_name].set_role(role)
            
            # Store the role for persistence
            self.predictor_roles[predictor_name] = role
            
            # Find the list widget item
            for i in range(self.predictors_list.count()):
                list_item = self.predictors_list.item(i)
                if list_item.data(Qt.ItemDataRole.UserRole) == predictor_name:
                    # Update the height of the list widget item
                    self.predictors_list.setItemWidget(list_item, self.predictor_items[predictor_name])
                    if role == 'variate':
                        list_item.setSizeHint(QSize(list_item.sizeHint().width(), 90))
                    else:
                        list_item.setSizeHint(QSize(list_item.sizeHint().width(), 30))
                    break
    
    def update_predictors(self, data, excluded_columns=None):
        """
        Update the predictors sidebar with columns from the data,
        excluding any specified columns
        """
        if excluded_columns is None:
            excluded_columns = []
            
        # Clear the current list
        self.predictors_list.clear()
        self.predictor_items = {}
        
        # Save current predictor roles before clearing
        saved_roles = self.predictor_roles.copy()
        
        # Add each column as a predictor item
        for column in data.columns:
            if column not in excluded_columns:
                # Create a custom widget for this predictor
                predictor_widget = PredictorItem(column)
                self.predictor_items[column] = predictor_widget
                
                # Restore previous role if it exists
                if column in saved_roles:
                    predictor_widget.set_role(saved_roles[column])
                
                # Add it to the list widget
                item = QListWidgetItem(self.predictors_list)
                item.setData(Qt.ItemDataRole.UserRole, column)  # Store column name as user data
                item.setSizeHint(predictor_widget.sizeHint())
                self.predictors_list.setItemWidget(item, predictor_widget)
    
    def handle_predictor_clicked(self, item):
        """Handle click on a predictor item"""
        predictor = item.data(Qt.ItemDataRole.UserRole)
        
        # Set the item as selected in the list widget
        self.predictors_list.setCurrentItem(item)
        
        # Update the currently selected predictor
        self.selected_predictor = predictor
        
        # Call the callback if it exists
        if self.on_predictor_selected:
            self.on_predictor_selected(predictor)
    
    def get_model_terms(self):
        """Get all terms for the model formula"""
        terms = []
        for name, widget in self.predictor_items.items():
            if widget.role:  # If it has a role (factor or variate)
                terms.extend(widget.get_formula_terms())
        return terms
        
    def get_selected_prediction_set(self):
        """Get the currently selected prediction set"""
        prediction_set = self.prediction_set_combo.currentText()
        return None if prediction_set == "All" else prediction_set
