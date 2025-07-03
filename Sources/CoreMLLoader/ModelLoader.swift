import CoreML
import Foundation
import AppKit
import CoreVideo

public class ModelLoader {
    public init() {}
    
    public func loadModel(from url: URL) throws -> MLModel {
        return try MLModel(contentsOf: url)
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