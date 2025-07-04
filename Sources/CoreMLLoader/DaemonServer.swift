import Foundation
import CoreML
import Network

public class DaemonServer {
    private let socketPath: String
    private var listener: NWListener?
    private var connections: [NWConnection] = []
    private var modelCache: [String: MLModel] = [:]
    private let modelLoader = ModelLoader()
    private let queue = DispatchQueue(label: "daemon.server")
    
    public init(socketPath: String = "/tmp/coreml-daemon.sock") {
        self.socketPath = socketPath
    }
    
    public func start() throws {
        // Remove existing socket file if it exists
        try? FileManager.default.removeItem(atPath: socketPath)
        
        // Create TCP listener on localhost
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters, on: 8765)
        
        listener?.newConnectionHandler = { [weak self] connection in
            self?.handleNewConnection(connection)
        }
        
        listener?.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                print("🚀 CoreML Daemon listening on localhost:8765")
            case .failed(let error):
                print("❌ Daemon failed to start: \(error)")
            case .cancelled:
                print("🛑 Daemon stopped")
            default:
                break
            }
        }
        
        listener?.start(queue: queue)
    }
    
    public func stop() {
        listener?.cancel()
        connections.forEach { $0.cancel() }
        connections.removeAll()
        try? FileManager.default.removeItem(atPath: socketPath)
    }
    
    private func handleNewConnection(_ connection: NWConnection) {
        connections.append(connection)
        
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.receiveMessage(from: connection)
            case .cancelled, .failed:
                self?.connections.removeAll { $0 === connection }
            default:
                break
            }
        }
        
        connection.start(queue: queue)
    }
    
    private func receiveMessage(from connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            if let data = data, !data.isEmpty {
                self?.processMessage(data: data, connection: connection)
            }
            
            if !isComplete {
                self?.receiveMessage(from: connection)
            }
        }
    }
    
    private func processMessage(data: Data, connection: NWConnection) {
        do {
            let request = try JSONDecoder().decode(DaemonRequest.self, from: data)
            let response = handleRequest(request)
            let responseData = try JSONEncoder().encode(response)
            
            connection.send(content: responseData, completion: .contentProcessed { error in
                if let error = error {
                    print("❌ Failed to send response: \(error)")
                }
            })
        } catch {
            let errorResponse = DaemonResponse(
                success: false,
                error: "Invalid request format: \(error.localizedDescription)",
                data: nil
            )
            
            if let errorData = try? JSONEncoder().encode(errorResponse) {
                connection.send(content: errorData, completion: .contentProcessed { _ in })
            }
        }
    }
    
    private func handleRequest(_ request: DaemonRequest) -> DaemonResponse {
        switch request.command {
        case "load":
            return handleLoadCommand(request)
        case "compile":
            return handleCompileCommand(request)
        case "info":
            return handleInfoCommand(request)
        case "infer":
            return handleInferCommand(request)
        case "status":
            return handleStatusCommand()
        case "cache":
            return handleCacheCommand()
        default:
            return DaemonResponse(success: false, error: "Unknown command: \(request.command)", data: nil)
        }
    }
    
    private func handleLoadCommand(_ request: DaemonRequest) -> DaemonResponse {
        guard let modelPath = request.arguments.first else {
            return DaemonResponse(success: false, error: "Model path required", data: nil)
        }
        
        let url = URL(fileURLWithPath: modelPath)
        
        do {
            let model = try modelLoader.loadModel(from: url)
            modelCache[modelPath] = model
            
            let description = modelLoader.getModelInfo(model)
            let modelInfo = ModelInfo(
                path: modelPath,
                description: description.metadata[MLModelMetadataKey.description] as? String ?? "Unknown",
                inputFeatures: description.inputDescriptionsByName.count,
                outputFeatures: description.outputDescriptionsByName.count
            )
            
            return DaemonResponse(success: true, error: nil, data: AnyCodable(modelInfo))
        } catch {
            return DaemonResponse(success: false, error: "Failed to load model: \(error.localizedDescription)", data: nil)
        }
    }
    
    private func handleCompileCommand(_ request: DaemonRequest) -> DaemonResponse {
        guard request.arguments.count >= 2 else {
            return DaemonResponse(success: false, error: "Input and output paths required", data: nil)
        }
        
        let inputPath = request.arguments[0]
        let outputPath = request.arguments[1]
        
        let inputURL = URL(fileURLWithPath: inputPath)
        let outputURL = URL(fileURLWithPath: outputPath)
        
        do {
            try modelLoader.compileModel(from: inputURL, to: outputURL)
            return DaemonResponse(success: true, error: nil, data: AnyCodable(["outputPath": outputPath]))
        } catch {
            return DaemonResponse(success: false, error: "Failed to compile model: \(error.localizedDescription)", data: nil)
        }
    }
    
    private func handleInfoCommand(_ request: DaemonRequest) -> DaemonResponse {
        guard let modelPath = request.arguments.first else {
            return DaemonResponse(success: false, error: "Model path required", data: nil)
        }
        
        // Try to get from cache first
        if let cachedModel = modelCache[modelPath] {
            let description = modelLoader.getModelInfo(cachedModel)
            let modelInfo = createDetailedModelInfo(path: modelPath, description: description)
            return DaemonResponse(success: true, error: nil, data: AnyCodable(modelInfo))
        }
        
        // Load fresh if not cached
        let url = URL(fileURLWithPath: modelPath)
        do {
            let model = try modelLoader.loadModel(from: url)
            let description = modelLoader.getModelInfo(model)
            let modelInfo = createDetailedModelInfo(path: modelPath, description: description)
            return DaemonResponse(success: true, error: nil, data: AnyCodable(modelInfo))
        } catch {
            return DaemonResponse(success: false, error: "Failed to analyze model: \(error.localizedDescription)", data: nil)
        }
    }
    
    private func handleInferCommand(_ request: DaemonRequest) -> DaemonResponse {
        guard request.arguments.count >= 2 else {
            return DaemonResponse(success: false, error: "Model path and input values required", data: nil)
        }
        
        let modelPath = request.arguments[0]
        let inputValues = Array(request.arguments.dropFirst(1))
        
        // Try to get from cache first
        var model: MLModel
        if let cachedModel = modelCache[modelPath] {
            model = cachedModel
        } else {
            // Load and cache the model
            let url = URL(fileURLWithPath: modelPath)
            do {
                model = try modelLoader.loadModel(from: url)
                modelCache[modelPath] = model
            } catch {
                return DaemonResponse(success: false, error: "Failed to load model: \(error.localizedDescription)", data: nil)
            }
        }
        
        do {
            let inputs = try modelLoader.createInputs(from: inputValues, for: model)
            let prediction = try modelLoader.runInference(model: model, inputs: inputs)
            
            let description = modelLoader.getModelInfo(model)
            var outputs: [String: Any] = [:]
            
            for (name, feature) in description.outputDescriptionsByName {
                if let output = prediction.featureValue(for: name) {
                    switch feature.type {
                    case .multiArray:
                        if let outputArray = modelLoader.extractOutputArray(from: prediction, name: name) {
                            outputs[name] = outputArray
                        }
                    case .string:
                        outputs[name] = output.stringValue
                    case .double:
                        outputs[name] = output.doubleValue
                    case .int64:
                        outputs[name] = output.int64Value
                    default:
                        outputs[name] = String(describing: output)
                    }
                }
            }
            
            return DaemonResponse(success: true, error: nil, data: AnyCodable(outputs))
        } catch {
            return DaemonResponse(success: false, error: "Failed to run inference: \(error.localizedDescription)", data: nil)
        }
    }
    
    private func handleStatusCommand() -> DaemonResponse {
        let status = DaemonStatus(
            isRunning: true,
            cachedModels: modelCache.count,
            modelPaths: Array(modelCache.keys),
            uptime: ProcessInfo.processInfo.systemUptime
        )
        return DaemonResponse(success: true, error: nil, data: AnyCodable(status))
    }
    
    private func handleCacheCommand() -> DaemonResponse {
        let cacheInfo = CacheInfo(
            modelCount: modelCache.count,
            models: modelCache.keys.map { path in
                let model = modelCache[path]!
                let description = modelLoader.getModelInfo(model)
                return CachedModelInfo(
                    path: path,
                    description: description.metadata[MLModelMetadataKey.description] as? String ?? "Unknown",
                    inputFeatures: description.inputDescriptionsByName.count,
                    outputFeatures: description.outputDescriptionsByName.count
                )
            }
        )
        return DaemonResponse(success: true, error: nil, data: AnyCodable(cacheInfo))
    }
    
    private func createDetailedModelInfo(path: String, description: MLModelDescription) -> DetailedModelInfo {
        let inputFeatures = description.inputDescriptionsByName.map { (name, feature) in
            FeatureInfo(name: name, type: String(describing: feature.type))
        }
        
        let outputFeatures = description.outputDescriptionsByName.map { (name, feature) in
            FeatureInfo(name: name, type: String(describing: feature.type))
        }
        
        return DetailedModelInfo(
            path: path,
            description: description.metadata[MLModelMetadataKey.description] as? String ?? "N/A",
            author: description.metadata[MLModelMetadataKey.author] as? String ?? "N/A",
            version: description.metadata[MLModelMetadataKey.versionString] as? String ?? "N/A",
            inputFeatures: inputFeatures,
            outputFeatures: outputFeatures
        )
    }
}

// MARK: - Data Transfer Objects

public struct DaemonRequest: Codable {
    let command: String
    let arguments: [String]
}

public struct DaemonResponse: Codable {
    public let success: Bool
    public let error: String?
    public let data: AnyCodable?
}

public struct ModelInfo: Codable {
    let path: String
    let description: String
    let inputFeatures: Int
    let outputFeatures: Int
}

public struct DetailedModelInfo: Codable {
    let path: String
    let description: String
    let author: String
    let version: String
    let inputFeatures: [FeatureInfo]
    let outputFeatures: [FeatureInfo]
}

public struct FeatureInfo: Codable {
    let name: String
    let type: String
}

public struct DaemonStatus: Codable {
    let isRunning: Bool
    let cachedModels: Int
    let modelPaths: [String]
    let uptime: TimeInterval
}

public struct CacheInfo: Codable {
    let modelCount: Int
    let models: [CachedModelInfo]
}

public struct CachedModelInfo: Codable {
    let path: String
    let description: String
    let inputFeatures: Int
    let outputFeatures: Int
}

// Helper for encoding arbitrary data
public struct AnyCodable: Codable {
    public let value: Any
    
    init<T>(_ value: T) {
        self.value = value
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        if let value = value as? String {
            try container.encode(value)
        } else if let value = value as? Int {
            try container.encode(value)
        } else if let value = value as? Double {
            try container.encode(value)
        } else if let value = value as? Bool {
            try container.encode(value)
        } else if let value = value as? TimeInterval {
            try container.encode(value)
        } else if let value = value as? Int64 {
            try container.encode(value)
        } else if let value = value as? ModelInfo {
            try container.encode(value)
        } else if let value = value as? DetailedModelInfo {
            try container.encode(value)
        } else if let value = value as? DaemonStatus {
            try container.encode(value)
        } else if let value = value as? CacheInfo {
            try container.encode(value)
        } else if let value = value as? [Double] {
            try container.encode(value)
        } else if let value = value as? [String: String] {
            try container.encode(value)
        } else if let value = value as? [String: Any] {
            let dict = value.mapValues { AnyCodable($0) }
            try container.encode(dict)
        } else if let codable = value as? Codable {
            try container.encode(codable)
        } else {
            try container.encode(String(describing: value))
        }
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let value = try? container.decode(String.self) {
            self.value = value
        } else if let value = try? container.decode(Int.self) {
            self.value = value
        } else if let value = try? container.decode(Double.self) {
            self.value = value
        } else if let value = try? container.decode(Bool.self) {
            self.value = value
        } else if let value = try? container.decode(ModelInfo.self) {
            self.value = value
        } else if let value = try? container.decode(DetailedModelInfo.self) {
            self.value = value
        } else if let value = try? container.decode(DaemonStatus.self) {
            self.value = value
        } else if let value = try? container.decode(CacheInfo.self) {
            self.value = value
        } else if let value = try? container.decode([String: AnyCodable].self) {
            self.value = value.mapValues { $0.value }
        } else {
            self.value = NSNull()
        }
    }
}