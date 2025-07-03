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
        
        Examples:
          coreml-experiment infer simple_model.mlmodel 3.0
          coreml-experiment infer image_model.mlmodel path/to/image.jpg
          coreml-experiment infer multi_input_model.mlmodel 1.0 path/to/image.png
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
            print("Examples:")
            print("  coreml-experiment infer simple_model.mlmodel 3.0")
            print("  coreml-experiment infer image_model.mlmodel path/to/image.jpg")
            print("  coreml-experiment infer multi_input_model.mlmodel 1.0 path/to/image.png")
            return
        }
        
        let modelPath = args[0]
        let inputValues = Array(args.dropFirst(1))
        
        let url = URL(fileURLWithPath: modelPath)
        
        do {
            let model = try loader.loadModel(from: url)
            let description = loader.getModelInfo(model)
            
            print("🔍 Model inputs:")
            for (name, feature) in description.inputDescriptionsByName {
                print("  - \(name): \(feature.type)")
            }
            print("")
            
            let inputs = try loader.createInputs(from: inputValues, for: model)
            print("📥 Running inference with inputs: \(inputValues)")
            
            let prediction = try loader.runInference(model: model, inputs: inputs)
            
            print("📤 Outputs:")
            for (name, feature) in description.outputDescriptionsByName {
                if let output = prediction.featureValue(for: name) {
                    switch feature.type {
                    case .multiArray:
                        if let outputArray = loader.extractOutputArray(from: prediction, name: name) {
                            print("  - \(name): \(outputArray)")
                        }
                    case .image:
                        print("  - \(name): Image output")
                    case .string:
                        print("  - \(name): \(output.stringValue ?? "nil")")
                    case .double:
                        print("  - \(name): \(output.doubleValue)")
                    case .int64:
                        print("  - \(name): \(output.int64Value)")
                    default:
                        print("  - \(name): \(output)")
                    }
                }
            }
            
        } catch {
            print("❌ Failed to run inference: \(error.localizedDescription)")
        }
    }
}
