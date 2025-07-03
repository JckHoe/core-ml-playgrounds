import CoreML
import Foundation
import AppKit

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
        
        guard let firstInput = description.inputDescriptionsByName.first else {
            throw NSError(domain: "ModelLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "No input features found"])
        }
        
        let inputName = firstInput.key
        let inputFeature = firstInput.value
        
        switch inputFeature.type {
        case .multiArray:
            let doubleValues = values.compactMap { Double($0) }
            if !doubleValues.isEmpty {
                let multiArray = try MLMultiArray(shape: [NSNumber(value: doubleValues.count)], dataType: .double)
                for (index, value) in doubleValues.enumerated() {
                    multiArray[index] = NSNumber(value: value)
                }
                inputs[inputName] = multiArray
            }
        case .string:
            inputs[inputName] = values.first ?? ""
        case .int64:
            inputs[inputName] = Int64(values.first ?? "0") ?? 0
        case .double:
            inputs[inputName] = Double(values.first ?? "0.0") ?? 0.0
        case .image:
            if let imagePath = values.first {
                let url = URL(fileURLWithPath: imagePath)
                do {
                    let imageData = try Data(contentsOf: url)
                    if let image = NSImage(data: imageData) {
                        inputs[inputName] = image
                        print("✅ Successfully loaded image: \(imagePath)")
                    } else {
                        throw NSError(domain: "ModelLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create NSImage from data"])
                    }
                } catch {
                    throw NSError(domain: "ModelLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load image from path '\(imagePath)': \(error.localizedDescription)"])
                }
            } else {
                throw NSError(domain: "ModelLoader", code: 4, userInfo: [NSLocalizedDescriptionKey: "No image path provided"])
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
        
        return inputs
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