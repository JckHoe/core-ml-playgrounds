# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Build and Run
```bash
# Build the Swift package
swift build

# Run the executable with commands
swift run coreml-experiment load path/to/model.mlmodel
swift run coreml-experiment compile input.mlmodel output.mlmodelc
swift run coreml-experiment info path/to/model.mlmodel
swift run coreml-experiment infer path/to/model.mlmodel 3.0

# Run tests
swift test
```

### Python Model Creation
```bash
# Activate virtual environment (if using venv)
source venv/bin/activate

# Create test models using Python scripts
python create_simple_model.py
python create_simple_model_manual.py
```

## Architecture Overview

This is a Swift Package Manager project that provides Core ML model loading, compilation, and inference capabilities. The codebase consists of:

### Core Components

- **CoreMLExperiment** (executable): Command-line interface with four main commands:
  - `load`: Load and validate Core ML models
  - `compile`: Compile .mlmodel to optimized .mlmodelc binaries
  - `info`: Display detailed model information and metadata
  - `infer`: Run inference with numeric inputs

- **CoreMLLoader** (library): Provides the core functionality:
  - `ModelLoader` class handles all Core ML operations
  - Model loading, compilation, and optimization
  - Input/output array creation and extraction
  - Inference execution with MLFeatureProvider

### Platform Support
- Supports macOS 13.0+, iOS 16.0+, watchOS 9.0+, tvOS 16.0+
- Optimized for Apple Neural Engine and GPU acceleration

### Python Integration
The project includes Python scripts for creating test Core ML models:
- `create_simple_model.py`: Uses scikit-learn to create a linear regression model
- `create_simple_model_manual.py`: Creates models using Core ML Tools neural network builder
- Both create models with the formula y = 2*x + 1

### Model Files
- `.mlmodel`: Source Core ML model format
- `.mlmodelc`: Compiled optimized binary format for deployment
- Test model included: `simple_model.mlmodel` implementing y = 2*x + 1

## Development Notes

The project uses async/await for the main executable entry point. The ModelLoader class provides synchronous methods for Core ML operations. Input handling supports multiple numeric values for inference commands.