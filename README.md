# Core ML Experiment

A Swift package for loading, compiling, and optimizing Core ML models for Apple devices.

## Features

- Load and validate Core ML models
- Compile models to optimized binaries
- Cross-device compatibility (iOS, macOS, watchOS, tvOS)
- Model information inspection

## Usage

### Build the project
```bash
swift build
```

### Run commands
```bash
# Load and validate a model
swift run coreml-experiment load path/to/model.mlmodel

# Compile model to optimized binary
swift run coreml-experiment compile input.mlmodel output.mlmodelc

# Show model information
swift run coreml-experiment info path/to/model.mlmodel
```

## Supported Platforms

- macOS 13.0+
- iOS 16.0+
- watchOS 9.0+
- tvOS 16.0+

## Model Compilation

The compiled `.mlmodelc` binary can run on any Apple device that supports the minimum platform requirements. The compilation process optimizes the model for Apple's Neural Engine and GPU acceleration.# core-ml-playgrounds
