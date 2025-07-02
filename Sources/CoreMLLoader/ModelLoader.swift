import CoreML
import Foundation

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