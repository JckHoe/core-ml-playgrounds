import CoreML
import Foundation
import AppKit
import CoreVideo
import Tokenizers

public class ModelLoader {
    public init() {}
    
    public func loadModel(from url: URL) throws -> MLModel {
        // For stateful models, we need to load with proper configuration
        let configuration = MLModelConfiguration()
        configuration.allowLowPrecisionAccumulationOnGPU = true
        
        return try MLModel(contentsOf: url, configuration: configuration)
    }
    
    public func compileModel(from sourceURL: URL, to destinationURL: URL) throws {
        let compiledURL = try MLModel.compileModel(at: sourceURL)
        try FileManager.default.moveItem(at: compiledURL, to: destinationURL)
    }
    
    public func optimizeForDevice(_ model: MLModel, configuration: MLModelConfiguration = MLModelConfiguration()) -> MLModel {
        return model
    }
    
    public func getModelInfo(_ model: MLModel) -> MLModelDescription {
        return model.modelDescription
    }
    
    public func runInference(model: MLModel, inputs: [String: Any]) throws -> MLFeatureProvider {
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let prediction = try model.prediction(from: inputProvider)
        return prediction
    }
    
    // New method specifically for stateful LLM inference
    public func runStatefulInference(model: MLModel, inputs: [String: Any]) throws -> MLFeatureProvider {
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputs)
        
        // For stateful models, we need to handle the inference differently
        // Use MLModelConfiguration to enable stateful inference
        let configuration = MLModelConfiguration()
        configuration.allowLowPrecisionAccumulationOnGPU = true
        
        let options = MLPredictionOptions()
        // Try to configure options for stateful inference
        
        let prediction = try model.prediction(from: inputProvider, options: options)
        return prediction
    }
    
    // Load model specifically for stateful inference
    public func loadModelForStatefulInference(from url: URL) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.allowLowPrecisionAccumulationOnGPU = true
        
        // Try to load with stateful configuration
        return try MLModel(contentsOf: url, configuration: configuration)
    }
    
    // Check if model requires stateful inference (has MLState inputs)
    public func requiresStatefulInference(model: MLModel) -> Bool {
        let inputFeatures = model.modelDescription.inputDescriptionsByName
        
        // Check for common stateful input names
        let stateInputNames = ["keyCache", "valueCache", "state", "past_key_values", "hidden_state"]
        
        for inputName in stateInputNames {
            if inputFeatures.keys.contains(inputName) {
                return true
            }
        }
        
        // Check for MLState types
        for (_, inputFeature) in inputFeatures {
            if case .state = inputFeature.type {
                return true
            }
        }
        
        // For LLM models, check if we have typical LLM inputs (inputIds + causalMask)
        // These often indicate a stateful model even if keyCache isn't exposed
        let hasInputIds = inputFeatures.keys.contains("inputIds")
        let hasCausalMask = inputFeatures.keys.contains("causalMask")
        
        if hasInputIds && hasCausalMask {
            print("🔍 Detected LLM model pattern - may require stateful inference")
            return true
        }
        
        return false
    }
    
    public func createInputArray(_ values: [Double], name: String = "input") throws -> [String: Any] {
        let multiArray = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: .double)
        for (index, value) in values.enumerated() {
            multiArray[index] = NSNumber(value: value)
        }
        return [name: multiArray]
    }
    
    public func createInputs(from values: [String], for model: MLModel) throws -> [String: Any] {
        let description = model.modelDescription
        var inputs: [String: Any] = [:]
        
        let inputFeatures = description.inputDescriptionsByName
        guard !inputFeatures.isEmpty else {
            throw NSError(domain: "ModelLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "No input features found"])
        }
        
        var valueIndex = 0
        
        for (inputName, inputFeature) in inputFeatures {
            switch inputFeature.type {
            case .multiArray:
                if let constraint = inputFeature.multiArrayConstraint {
                    inputs[inputName] = try createMultiArrayInput(
                        name: inputName,
                        constraint: constraint,
                        values: values,
                        valueIndex: &valueIndex
                    )
                }
            case .string:
                inputs[inputName] = valueIndex < values.count ? values[valueIndex] : ""
                valueIndex += 1
            case .int64:
                let value = valueIndex < values.count ? values[valueIndex] : "0"
                inputs[inputName] = Int64(value) ?? 0
                valueIndex += 1
            case .double:
                let value = valueIndex < values.count ? values[valueIndex] : "0.0"
                inputs[inputName] = Double(value) ?? 0.0
                valueIndex += 1
            case .image:
                if valueIndex < values.count {
                    let imagePath = values[valueIndex]
                    let url = URL(fileURLWithPath: imagePath)
                    do {
                        let imageData = try Data(contentsOf: url)
                        if let image = NSImage(data: imageData) {
                            // Get the required size from the model's image constraint
                            var targetSize = CGSize(width: 416, height: 416) // TinyYolo default
                            if let imageConstraint = inputFeature.imageConstraint {
                                targetSize = CGSize(width: imageConstraint.pixelsWide, height: imageConstraint.pixelsHigh)
                            }
                            
                            if let pixelBuffer = image.toCVPixelBuffer(size: targetSize) {
                                inputs[inputName] = pixelBuffer
                            } else {
                                throw NSError(domain: "ModelLoader", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to convert NSImage to CVPixelBuffer"])
                            }
                        } else {
                            throw NSError(domain: "ModelLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create NSImage from data"])
                        }
                    } catch {
                        throw NSError(domain: "ModelLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load image from path '\(imagePath)': \(error.localizedDescription)"])
                    }
                    valueIndex += 1
                }
            default:
                let doubleValues = values.compactMap { Double($0) }
                if !doubleValues.isEmpty {
                    let multiArray = try MLMultiArray(shape: [NSNumber(value: doubleValues.count)], dataType: .double)
                    for (index, value) in doubleValues.enumerated() {
                        multiArray[index] = NSNumber(value: value)
                    }
                    inputs[inputName] = multiArray
                }
            }
        }
        
        return inputs
    }
    
    private func createMultiArrayInput(name: String, constraint: MLMultiArrayConstraint, values: [String], valueIndex: inout Int) throws -> MLMultiArray {
        let shape = constraint.shape
        let dataType = constraint.dataType
        
        // Handle special cases for common language model inputs
        if name.lowercased().contains("mask") || name.lowercased().contains("causal") {
            return try createCausalMask(shape: shape, dataType: dataType)
        }
        
        if name.lowercased().contains("position") || name.lowercased().contains("pos") {
            return try createPositionalIds(shape: shape, dataType: dataType)
        }
        
        // Default: try to fill with provided values
        let multiArray = try MLMultiArray(shape: shape, dataType: dataType)
        let totalElements = shape.reduce(1) { $0 * $1.intValue }
        
        // Fill with provided values or defaults
        for i in 0..<totalElements {
            let value: Any
            if valueIndex < values.count {
                if let doubleVal = Double(values[valueIndex]) {
                    value = doubleVal
                } else {
                    value = 0.0
                }
                if i == 0 { valueIndex += 1 } // Only increment once per input
            } else {
                value = 0.0
            }
            multiArray[i] = NSNumber(value: value as! Double)
        }
        
        return multiArray
    }
    
    private func createCausalMask(shape: [NSNumber], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let multiArray = try MLMultiArray(shape: shape, dataType: dataType)
        
        // For causal masks, typically shape is [batch, seq_len, seq_len]
        if shape.count >= 2 {
            let seqLen = shape[shape.count - 1].intValue
            let prevSeqLen = shape[shape.count - 2].intValue
            
            // Create lower triangular mask (causal)
            for i in 0..<prevSeqLen {
                for j in 0..<seqLen {
                    let index = i * seqLen + j
                    multiArray[index] = NSNumber(value: j <= i ? 1.0 : 0.0)
                }
            }
        }
        
        return multiArray
    }
    
    private func createPositionalIds(shape: [NSNumber], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let multiArray = try MLMultiArray(shape: shape, dataType: dataType)
        let totalElements = shape.reduce(1) { $0 * $1.intValue }
        
        // Fill with sequential position IDs
        for i in 0..<totalElements {
            multiArray[i] = NSNumber(value: i)
        }
        
        return multiArray
    }
    
    public func extractOutputArray(from prediction: MLFeatureProvider, name: String = "output") -> [Double]? {
        guard let output = prediction.featureValue(for: name)?.multiArrayValue else {
            return nil
        }
        
        var results: [Double] = []
        for i in 0..<output.count {
            results.append(output[i].doubleValue)
        }
        return results
    }
    
    // MARK: - Text Processing for LLM Models
    
    public func tokenizeText(_ text: String, modelName: String? = nil) -> [Int32] {
        // Try to use proper tokenization if model name is provided
        if let modelName = modelName {
            return tokenizeWithHuggingFace(text, modelName: modelName)
        }
        
        // Fallback to basic tokenization for unknown models
        return tokenizeBasic(text)
    }
    
    private func tokenizeWithHuggingFace(_ text: String, modelName: String) -> [Int32] {
        // For now, fall back to basic tokenization due to async requirements
        // TODO: Implement proper async tokenization support
        print("Note: Using basic tokenization - proper HuggingFace tokenization requires async support")
        return tokenizeBasic(text)
    }
    
    private func tokenizeBasic(_ text: String) -> [Int32] {
        // Simple tokenization - split by whitespace and convert to basic token IDs
        // This is a fallback when proper tokenization is not available
        let words = text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
        
        var tokenIds: [Int32] = []
        
        for word in words {
            // Convert characters to basic token IDs (simplified approach)
            for char in word {
                let ascii = Int32(char.asciiValue ?? 0)
                tokenIds.append(ascii)
            }
            // Add space token (32 is ASCII space)
            tokenIds.append(32)
        }
        
        // Remove trailing space if present
        if tokenIds.last == 32 {
            tokenIds.removeLast()
        }
        
        return tokenIds
    }
    
    public func createTextInputs(from text: String, for model: MLModel) throws -> [String: Any] {
        let description = model.modelDescription
        let inputFeatures = description.inputDescriptionsByName
        
        var inputs: [String: Any] = [:]
        let tokenIds = tokenizeText(text, modelName: "meta-llama/Llama-3.2-1B")
        
        // First, try to create basic inputs that we can detect
        for (inputName, inputFeature) in inputFeatures {
            switch inputFeature.type {
            case .multiArray:
                if let constraint = inputFeature.multiArrayConstraint {
                    if inputName.lowercased().contains("input") || inputName.lowercased().contains("token") {
                        // This is likely the input token array
                        inputs[inputName] = try createTokenIdArray(
                            tokenIds: tokenIds,
                            constraint: constraint
                        )
                    } else if inputName.lowercased().contains("mask") || inputName.lowercased().contains("causal") {
                        // This is likely the causal attention mask
                        inputs[inputName] = try createCausalMaskForTokens(
                            tokenCount: tokenIds.count,
                            constraint: constraint
                        )
                    } else {
                        // Default handling for other arrays
                        inputs[inputName] = try createDefaultArrayInput(constraint: constraint)
                    }
                }
            case .state:
                // Handle MLState inputs (like keyCache)
                // For stateful LLM models, we skip state inputs on first inference
                // The model will create and manage the state internally
                print("⚠️  Skipping MLState input '\(inputName)' - will be managed by model")
                // Don't add this input to the dictionary - let CoreML handle it
            default:
                // Handle other types with default values
                inputs[inputName] = try createDefaultInput(for: inputFeature)
            }
        }
        
        return inputs
    }
    
    public func createTextInputsWithStateManagement(from text: String, for model: MLModel) throws -> [String: Any] {
        // This method attempts to handle models that require state management
        // including both explicit MLState inputs and hidden state requirements
        
        var inputs = try createTextInputs(from: text, for: model)
        
        // Try to create hidden state inputs that might not be in the model description
        // This is common for models that have been compiled with state management
        inputs = try addHiddenStateInputs(to: inputs, for: model)
        
        return inputs
    }
    
    // New method to handle models that require empty state initialization
    public func createTextInputsWithEmptyState(from text: String, for model: MLModel) throws -> [String: Any] {
        // For models with keyCache requirements, just create regular inputs
        // The stateful inference should handle state initialization
        return try createTextInputs(from: text, for: model)
    }
    
    // Method to handle LLM models that may have hidden keyCache inputs
    public func createTextInputsForLLM(from text: String, for model: MLModel) throws -> [String: Any] {
        // For LLM models, just create regular text inputs
        // The stateful inference should handle the keyCache automatically
        return try createTextInputs(from: text, for: model)
    }
    
    private func createTokenIdArray(tokenIds: [Int32], constraint: MLMultiArrayConstraint) throws -> MLMultiArray {
        let shape = constraint.shape
        let dataType = constraint.dataType
        
        let multiArray = try MLMultiArray(shape: shape, dataType: dataType)
        
        // Fill the array with token IDs
        let maxTokens = min(tokenIds.count, multiArray.count)
        for i in 0..<maxTokens {
            multiArray[i] = NSNumber(value: tokenIds[i])
        }
        
        // Pad with zeros if needed
        for i in maxTokens..<multiArray.count {
            multiArray[i] = NSNumber(value: 0)
        }
        
        return multiArray
    }
    
    private func createCausalMaskForTokens(tokenCount: Int, constraint: MLMultiArrayConstraint) throws -> MLMultiArray {
        let shape = constraint.shape
        let dataType = constraint.dataType
        
        let multiArray = try MLMultiArray(shape: shape, dataType: dataType)
        
        // Create a causal mask based on token count
        if shape.count >= 2 {
            let seqLen = min(tokenCount, shape.last?.intValue ?? tokenCount)
            let batchSize = shape.count > 2 ? shape[0].intValue : 1
            
            for batch in 0..<batchSize {
                for i in 0..<seqLen {
                    for j in 0..<seqLen {
                        let index = batch * seqLen * seqLen + i * seqLen + j
                        if index < multiArray.count {
                            // Causal mask: 1 for positions that can attend, 0 for future positions
                            multiArray[index] = NSNumber(value: j <= i ? 1.0 : 0.0)
                        }
                    }
                }
            }
        }
        
        return multiArray
    }
    
    
    private func addHiddenStateInputs(to inputs: [String: Any], for model: MLModel) throws -> [String: Any] {
        var updatedInputs = inputs
        
        // Common hidden state input names that might not be in the model description
        let potentialStateInputs = ["keyCache", "valueCache", "state", "hidden_state", "past_key_values"]
        
        for stateName in potentialStateInputs {
            if updatedInputs[stateName] == nil {
                // Try to create a default state input
                // This is a fallback for models with hidden state requirements
                if let stateArray = try? createDefaultStateArray(name: stateName) {
                    // For now, just use MLMultiArray - MLState creation is complex
                    updatedInputs[stateName] = MLFeatureValue(multiArray: stateArray)
                }
            }
        }
        
        return updatedInputs
    }
    
    private func createDefaultStateArray(name: String) throws -> MLMultiArray {
        // Create a default state array for common state patterns
        // This is a heuristic-based approach for models with hidden state
        let defaultShape = [1, 1, 1] // Minimal shape that most models can accept
        let stateArray = try MLMultiArray(shape: defaultShape.map { NSNumber(value: $0) }, dataType: .float32)
        
        // Initialize with zeros
        for i in 0..<stateArray.count {
            stateArray[i] = NSNumber(value: 0.0)
        }
        
        return stateArray
    }
    
    public func createIOSurfaceArray(shape: [Int], dataType: MLMultiArrayDataType = .float16) -> MLMultiArray? {
        // Create an IOSurface-backed MLMultiArray for better performance
        // This avoids CPU-ANE copies and improves memory efficiency
        guard
            shape.count > 0,
            let width = shape.last
        else { return nil }
        
        let height = shape[0..<shape.count-1].reduce(1) { $0 * $1 }
        
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        
        let pixelFormat: OSType = dataType == .float16 ? kCVPixelFormatType_OneComponent16Half : kCVPixelFormatType_OneComponent32Float
        
        guard kCVReturnSuccess == CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat,
            attributes,
            &pixelBuffer)
        else { return nil }
        
        guard let pixelBuffer = pixelBuffer else { return nil }
        
        // Initialize with zeros
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        memset(CVPixelBufferGetBaseAddress(pixelBuffer), 0, CVPixelBufferGetDataSize(pixelBuffer))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return MLMultiArray(pixelBuffer: pixelBuffer, shape: shape.map { NSNumber(value: $0) })
    }
    
    private func createDefaultArrayInput(constraint: MLMultiArrayConstraint) throws -> MLMultiArray {
        let multiArray = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
        
        // Fill with zeros
        for i in 0..<multiArray.count {
            multiArray[i] = NSNumber(value: 0.0)
        }
        
        return multiArray
    }
    
    private func createDefaultInput(for feature: MLFeatureDescription) throws -> Any {
        switch feature.type {
        case .string:
            return ""
        case .int64:
            return Int64(0)
        case .double:
            return 0.0
        default:
            throw NSError(domain: "ModelLoader", code: 101, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported input type: \(feature.type)"
            ])
        }
    }
}

extension NSImage {
    func toCVPixelBuffer(size: CGSize = CGSize(width: 416, height: 416)) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA, // Common format for CoreML
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        let drawRect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
        
        context.clear(drawRect)
        
        if let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            context.draw(cgImage, in: drawRect)
        }
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}