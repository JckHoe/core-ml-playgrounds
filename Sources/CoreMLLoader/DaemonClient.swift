import Foundation
import Network

public class DaemonClient {
    private let socketPath: String
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "daemon.client")
    
    public init(socketPath: String = "/tmp/coreml-daemon.sock") {
        self.socketPath = socketPath
    }
    
    public func connect() throws {
        let endpoint = NWEndpoint.hostPort(host: "localhost", port: 8765)
        connection = NWConnection(to: endpoint, using: .tcp)
        
        var isConnected = false
        var connectionError: Error?
        
        connection?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                isConnected = true
            case .failed(let error):
                connectionError = error
            default:
                break
            }
        }
        
        connection?.start(queue: queue)
        
        // Wait for connection to be established
        let timeout = DispatchTime.now() + .seconds(5)
        while !isConnected && connectionError == nil && DispatchTime.now() < timeout {
            Thread.sleep(forTimeInterval: 0.01)
        }
        
        if let error = connectionError {
            throw error
        }
        
        if !isConnected {
            throw DaemonError.connectionTimeout
        }
    }
    
    public func disconnect() {
        connection?.cancel()
        connection = nil
    }
    
    public func sendCommand(_ command: String, arguments: [String]) throws -> DaemonResponse {
        guard let connection = connection else {
            throw DaemonError.notConnected
        }
        
        let request = DaemonRequest(command: command, arguments: arguments)
        let requestData = try JSONEncoder().encode(request)
        
        var responseData: Data?
        var sendError: Error?
        
        connection.send(content: requestData, completion: .contentProcessed { error in
            sendError = error
        })
        
        if let error = sendError {
            throw error
        }
        
        // Receive response
        let semaphore = DispatchSemaphore(value: 0)
        
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, _, error in
            if let error = error {
                sendError = error
            } else if let data = data {
                responseData = data
            }
            semaphore.signal()
        }
        
        let timeout = DispatchTime.now() + .seconds(30)
        if semaphore.wait(timeout: timeout) == .timedOut {
            throw DaemonError.responseTimeout
        }
        
        if let error = sendError {
            throw error
        }
        
        guard let data = responseData else {
            throw DaemonError.noResponse
        }
        
        return try JSONDecoder().decode(DaemonResponse.self, from: data)
    }
    
    // Convenience methods for each command
    public func loadModel(path: String) throws -> DaemonResponse {
        return try sendCommand("load", arguments: [path])
    }
    
    public func compileModel(inputPath: String, outputPath: String) throws -> DaemonResponse {
        return try sendCommand("compile", arguments: [inputPath, outputPath])
    }
    
    public func getModelInfo(path: String) throws -> DaemonResponse {
        return try sendCommand("info", arguments: [path])
    }
    
    public func runInference(modelPath: String, inputs: [String]) throws -> DaemonResponse {
        var args = [modelPath]
        args.append(contentsOf: inputs)
        return try sendCommand("infer", arguments: args)
    }
    
    public func getStatus() throws -> DaemonResponse {
        return try sendCommand("status", arguments: [])
    }
    
    public func getCacheInfo() throws -> DaemonResponse {
        return try sendCommand("cache", arguments: [])
    }
    
    public static func isDaemonRunning(socketPath: String = "/tmp/coreml-daemon.sock") -> Bool {
        // Try to connect to check if daemon is running
        let client = DaemonClient(socketPath: socketPath)
        do {
            try client.connect()
            client.disconnect()
            return true
        } catch {
            return false
        }
    }
}

public enum DaemonError: Error, LocalizedError {
    case notConnected
    case connectionTimeout
    case responseTimeout
    case noResponse
    case serverError(String)
    
    public var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to daemon"
        case .connectionTimeout:
            return "Connection to daemon timed out"
        case .responseTimeout:
            return "Response from daemon timed out"
        case .noResponse:
            return "No response from daemon"
        case .serverError(let message):
            return "Server error: \(message)"
        }
    }
}