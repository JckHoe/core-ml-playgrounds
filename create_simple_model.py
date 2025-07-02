#!/usr/bin/env python3
"""
Create a simple Core ML model for testing inference.
This creates a basic linear regression model: y = 2*x + 1
"""

import coremltools as ct
import numpy as np
from sklearn.linear_model import LinearRegression

# Create simple training data
X = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y = np.array([3, 5, 7, 9, 11], dtype=np.float32)  # y = 2*x + 1

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Convert to Core ML
coreml_model = ct.converters.sklearn.convert(
    model,
    input_features=[("input", "double")],
    output_feature_names=["output"]
)

# Add metadata
coreml_model.short_description = "Simple linear regression: y = 2*x + 1"
coreml_model.author = "CoreML Experiment"
coreml_model.version = "1.0"

# Save the model
coreml_model.save("simple_model.mlmodel")
print("✅ Created simple_model.mlmodel")
print("   Formula: y = 2*x + 1")
print("   Input: 'input' (double)")
print("   Output: 'output' (double)")