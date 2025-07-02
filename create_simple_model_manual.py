#!/usr/bin/env python3
"""
Create a simple Core ML model manually using Core ML tools.
This creates a basic linear function: y = 2*x + 1
"""

import coremltools as ct
from coremltools.models import datatypes, MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder
import numpy as np

# Create a neural network builder
builder = NeuralNetworkBuilder(
    input_features=[('input', datatypes.Array(1))],
    output_features=[('output', datatypes.Array(1))]
)

# Add a linear layer that computes y = 2*x + 1
# This is equivalent to a fully connected layer with weight=2 and bias=1
builder.add_inner_product(
    name='linear',
    W=np.array([[2.0]]),  # weight matrix
    b=np.array([1.0]),    # bias vector
    input_channels=1,
    output_channels=1,
    has_bias=True,
    input_name='input',
    output_name='output'
)

# Create the model
model_spec = builder.spec
model = MLModel(model_spec)

# Add metadata
model.short_description = "Simple linear function: y = 2*x + 1"
model.author = "CoreML Experiment"
model.version = "1.0"

# Save the model
model.save("simple_model.mlmodel")
print("✅ Created simple_model.mlmodel")
print("   Formula: y = 2*x + 1")
print("   Input: 'input' (array of size 1)")
print("   Output: 'output' (array of size 1)")
print("   Test: input [3.0] should output [7.0]")