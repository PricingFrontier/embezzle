# Embezzle

A Python library for visualizing and building Generalized Linear Models (GLMs) using statsmodels with a PyQt interface.

## Features

- Load and preprocess data from various sources
- Build and fit GLMs using the statsmodels backend
- Interactive UI for model specification and exploration
- Visualize model diagnostics and results
- Export models and predictions

## Installation

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/embezzle.git
cd embezzle
pip install -e .
```

## Usage

### Running the GUI Application

To start the GUI application:

```bash
embezzle
```

Or from Python:

```python
from embezzle.ui.main_window import run_app
run_app()
```

### Using the GLM Builder Programmatically

```python
import pandas as pd
from embezzle.models.model_builder import GLMBuilder

# Load your data
data = pd.read_csv('your_data.csv')

# Create a GLM builder
builder = GLMBuilder()
builder.load_data(data)

# Set up the model
builder.set_formula('y ~ x1 + x2 + x3')
builder.set_family('poisson', 'log')

# Build and fit the model
model = builder.build_model()
results = builder.fit_model()

# Get model summary
print(builder.get_summary())

# Make predictions
predictions = builder.predict()
```

### Data Utilities

```python
from embezzle.utils.data_utils import standardize_numeric, encode_categorical, split_data

# Preprocess your data
data = standardize_numeric(data, columns=['x1', 'x2'])
data = encode_categorical(data, columns=['category1', 'category2'])

# Split data for training and testing
X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, response_var='y')
```

## Directory Structure

```
embezzle/
├── embezzle/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_builder.py
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py
│   └── utils/
│       ├── __init__.py
│       └── data_utils.py
├── data/
│   └── (sample data files)
├── examples/
│   └── (example scripts)
├── setup.py
└── README.md
```

## Requirements

- Python 3.8+
- statsmodels
- PyQt6
- pandas
- numpy
- matplotlib
- scikit-learn

## License

MIT License
