import Foundation
import CoreML
import CoreMLLoader

@main
struct CoreMLExperiment {
    static func main() async throws {
        let loader = ModelLoader()
        
        let args = CommandLine.arguments
        guard args.count > 1 else {
            printUsage()
            return
        }
        
        let command = args[1]
        
        switch command {
        case "load":
            try await handleLoad(args: Array(args.dropFirst(2)), loader: loader)
        case "compile":
            try await handleCompile(args: Array(args.dropFirst(2)), loader: loader)
        case "info":
            try await handleInfo(args: Array(args.dropFirst(2)), loader: loader)
        case "infer":
            try await handleInfer(args: Array(args.dropFirst(2)), loader: loader)
        default:
            print("Unknown command: \(command)")
            printUsage()
        }
    }
    
    static func printUsage() {
        print("""
        Usage:
          coreml-experiment load <model-path>     - Load and validate a Core ML model
          coreml-experiment compile <input> <output> - Compile model to optimized binary
          coreml-experiment info <model-path>     - Show model information
          coreml-experiment infer <model-path> <input-values> - Run inference on model
        """)
    }
    
    static func handleLoad(args: [String], loader: ModelLoader) async throws {
        guard let modelPath = args.first else {
            print("Error: Model path required")
            return
        }
        
        let url = URL(fileURLWithPath: modelPath)
        
        do {
            let model = try loader.loadModel(from: url)
            print("✅ Successfully loaded model: \(modelPath)")
            
            let description = loader.getModelInfo(model)
            print("Model type: \(description.metadata[MLModelMetadataKey.description] ?? "Unknown")")
            print("Input features: \(description.inputDescriptionsByName.count)")
            print("Output features: \(description.outputDescriptionsByName.count)")
            
        } catch {
            print("❌ Failed to load model: \(error.localizedDescription)")
        }
    }
    
    static func handleCompile(args: [String], loader: ModelLoader) async throws {
        guard args.count >= 2 else {
            print("Error: Input and output paths required")
            return
        }
        
        let inputPath = args[0]
        let outputPath = args[1]
        
        let inputURL = URL(fileURLWithPath: inputPath)
        let outputURL = URL(fileURLWithPath: outputPath)
        
        do {
            try loader.compileModel(from: inputURL, to: outputURL)
            print("✅ Successfully compiled model to: \(outputPath)")
            print("📦 Binary is optimized for Apple devices")
        } catch {
            print("❌ Failed to compile model: \(error.localizedDescription)")
        }
    }
    
    static func handleInfo(args: [String], loader: ModelLoader) async throws {
        guard let modelPath = args.first else {
            print("Error: Model path required")
            return
        }
        
        let url = URL(fileURLWithPath: modelPath)
        
        do {
            let model = try loader.loadModel(from: url)
            let description = loader.getModelInfo(model)
            
            print("📋 Model Information")
            print("==================")
            print("Path: \(modelPath)")
            print("Description: \(description.metadata[MLModelMetadataKey.description] ?? "N/A")")
            print("Author: \(description.metadata[MLModelMetadataKey.author] ?? "N/A")")
            print("Version: \(description.metadata[MLModelMetadataKey.versionString] ?? "N/A")")
            print("")
            
            print("📥 Input Features:")
            for (name, feature) in description.inputDescriptionsByName {
                print("  - \(name): \(feature.type)")
            }
            
            print("")
            print("📤 Output Features:")
            for (name, feature) in description.outputDescriptionsByName {
                print("  - \(name): \(feature.type)")
            }
            
        } catch {
            print("❌ Failed to analyze model: \(error.localizedDescription)")
        }
    }
    
    static func handleInfer(args: [String], loader: ModelLoader) async throws {
        guard args.count >= 2 else {
            print("Error: Model path and input values required")
            print("Example: coreml-experiment infer simple_model.mlmodel 3.0")
            return
        }
        
        let modelPath = args[0]
        let inputValues = Array(args.dropFirst(1))
        
        let url = URL(fileURLWithPath: modelPath)
        
        do {
            let model = try loader.loadModel(from: url)
            
            let doubleValues = inputValues.compactMap { Double($0) }
            guard !doubleValues.isEmpty else {
                print("❌ Error: Invalid input values. Please provide numeric values.")
                return
            }
            
            let inputs = try loader.createInputArray(doubleValues)
            let prediction = try loader.runInference(model: model, inputs: inputs)
            
            if let outputs = loader.extractOutputArray(from: prediction) {
                print(outputs)
            } else {
                print("❌ Failed to extract output from prediction")
            }
            
        } catch {
            print("❌ Failed to run inference: \(error.localizedDescription)")
        }
    }
}
