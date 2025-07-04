import Foundation
import CoreML
import CoreMLLoader

@main
struct CoreMLExperiment {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            printUsage()
            return
        }
        
        let command = args[1]
        
        // Handle daemon management commands
        switch command {
        case "daemon":
            try await handleDaemonCommand(args: Array(args.dropFirst(2)))
            return
        case "start-daemon":
            try DaemonManager.startDaemon()
            return
        case "stop-daemon":
            DaemonManager.stopDaemon()
            return
        case "restart-daemon":
            DaemonManager.restartDaemon()
            return
        case "daemon-status":
            DaemonManager.getDaemonStatus()
            return
        case "cache-info":
            DaemonManager.getCacheInfo()
            return
        default:
            break
        }
        
        // Check for --daemon flag
        let useDaemon = args.contains("--daemon") || args.contains("-d")
        let cleanArgs = args.filter { !["--daemon", "-d"].contains($0) }
        
        // Try daemon mode first if requested or if daemon is running
        if useDaemon || DaemonManager.isDaemonRunning() {
            do {
                try await handleWithDaemon(command: command, args: Array(cleanArgs.dropFirst(2)))
                return
            } catch {
                if useDaemon {
                    print("❌ Failed to use daemon: \(error)")
                    print("💡 Try starting daemon with: coreml-experiment start-daemon")
                    return
                }
                // Fall back to direct mode if daemon flag wasn't explicitly set
                print("⚠️  Daemon unavailable, falling back to direct mode")
            }
        }
        
        // Direct mode
        let loader = ModelLoader()
        switch command {
        case "load":
            try await handleLoad(args: Array(cleanArgs.dropFirst(2)), loader: loader)
        case "compile":
            try await handleCompile(args: Array(cleanArgs.dropFirst(2)), loader: loader)
        case "info":
            try await handleInfo(args: Array(cleanArgs.dropFirst(2)), loader: loader)
        case "infer":
            try await handleInfer(args: Array(cleanArgs.dropFirst(2)), loader: loader)
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
        
        Daemon Commands:
          coreml-experiment start-daemon          - Start the daemon server
          coreml-experiment stop-daemon           - Stop the daemon server
          coreml-experiment restart-daemon        - Restart the daemon server
          coreml-experiment daemon-status         - Show daemon status
          coreml-experiment cache-info            - Show cached model information
        
        Flags:
          --daemon, -d                           - Use daemon mode (faster for repeated calls)
        
        Examples:
          # Direct mode (loads model each time)
          coreml-experiment infer simple_model.mlmodel 3.0
          
          # Daemon mode (keeps model in memory)
          coreml-experiment start-daemon
          coreml-experiment infer simple_model.mlmodel 3.0  # Auto-uses daemon
          coreml-experiment infer --daemon simple_model.mlmodel 3.0  # Explicit daemon
          
          # Image inference
          coreml-experiment infer image_model.mlmodel path/to/image.jpg
          
          # Multi-input model
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
                        print("  - \(name): \(output.stringValue)")
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
    
    // MARK: - Daemon Mode Handlers
    
    static func handleDaemonCommand(args: [String]) async throws {
        guard let subcommand = args.first else {
            print("Error: Daemon subcommand required")
            print("Available subcommands: start, stop, restart, status, cache")
            return
        }
        
        switch subcommand {
        case "start":
            try DaemonManager.startDaemon()
        case "stop":
            DaemonManager.stopDaemon()
        case "restart":
            DaemonManager.restartDaemon()
        case "status":
            DaemonManager.getDaemonStatus()
        case "cache":
            DaemonManager.getCacheInfo()
        default:
            print("Unknown daemon subcommand: \(subcommand)")
        }
    }
    
    static func handleWithDaemon(command: String, args: [String]) async throws {
        let client = DaemonClient()
        try client.connect()
        defer { client.disconnect() }
        
        switch command {
        case "load":
            try await handleLoadDaemon(args: args, client: client)
        case "compile":
            try await handleCompileDaemon(args: args, client: client)
        case "info":
            try await handleInfoDaemon(args: args, client: client)
        case "infer":
            try await handleInferDaemon(args: args, client: client)
        default:
            throw DaemonError.serverError("Unknown command: \(command)")
        }
    }
    
    static func handleLoadDaemon(args: [String], client: DaemonClient) async throws {
        guard let modelPath = args.first else {
            print("Error: Model path required")
            return
        }
        
        let response = try client.loadModel(path: modelPath)
        
        if response.success {
            print("✅ Successfully loaded model: \(modelPath)")
            if let modelInfo = response.data?.value as? [String: Any] {
                if let description = modelInfo["description"] as? String {
                    print("Model type: \(description)")
                }
                if let inputFeatures = modelInfo["inputFeatures"] as? Int,
                   let outputFeatures = modelInfo["outputFeatures"] as? Int {
                    print("Input features: \(inputFeatures)")
                    print("Output features: \(outputFeatures)")
                }
            }
        } else {
            print("❌ Failed to load model: \(response.error ?? "Unknown error")")
        }
    }
    
    static func handleCompileDaemon(args: [String], client: DaemonClient) async throws {
        guard args.count >= 2 else {
            print("Error: Input and output paths required")
            return
        }
        
        let inputPath = args[0]
        let outputPath = args[1]
        
        let response = try client.compileModel(inputPath: inputPath, outputPath: outputPath)
        
        if response.success {
            print("✅ Successfully compiled model to: \(outputPath)")
            print("📦 Binary is optimized for Apple devices")
        } else {
            print("❌ Failed to compile model: \(response.error ?? "Unknown error")")
        }
    }
    
    static func handleInfoDaemon(args: [String], client: DaemonClient) async throws {
        guard let modelPath = args.first else {
            print("Error: Model path required")
            return
        }
        
        let response = try client.getModelInfo(path: modelPath)
        
        if response.success {
            print("📋 Model Information")
            print("==================")
            
            if let modelInfo = response.data?.value as? [String: Any] {
                print("Path: \(modelPath)")
                print("Description: \(modelInfo["description"] as? String ?? "N/A")")
                print("Author: \(modelInfo["author"] as? String ?? "N/A")")
                print("Version: \(modelInfo["version"] as? String ?? "N/A")")
                print("")
                
                if let inputFeatures = modelInfo["inputFeatures"] as? [[String: Any]] {
                    print("📥 Input Features:")
                    for feature in inputFeatures {
                        if let name = feature["name"] as? String,
                           let type = feature["type"] as? String {
                            print("  - \(name): \(type)")
                        }
                    }
                }
                
                print("")
                
                if let outputFeatures = modelInfo["outputFeatures"] as? [[String: Any]] {
                    print("📤 Output Features:")
                    for feature in outputFeatures {
                        if let name = feature["name"] as? String,
                           let type = feature["type"] as? String {
                            print("  - \(name): \(type)")
                        }
                    }
                }
            }
        } else {
            print("❌ Failed to analyze model: \(response.error ?? "Unknown error")")
        }
    }
    
    static func handleInferDaemon(args: [String], client: DaemonClient) async throws {
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
        
        print("📥 Running inference with inputs: \(inputValues)")
        
        let response = try client.runInference(modelPath: modelPath, inputs: inputValues)
        
        if response.success {
            print("📤 Outputs:")
            if let outputs = response.data?.value as? [String: Any] {
                for (name, value) in outputs {
                    if let arrayValue = value as? [Double] {
                        print("  - \(name): \(arrayValue)")
                    } else {
                        print("  - \(name): \(value)")
                    }
                }
            }
        } else {
            print("❌ Failed to run inference: \(response.error ?? "Unknown error")")
        }
    }
}

// MARK: - Daemon Manager

struct DaemonManager {
    static let pidFile = "/tmp/coreml-daemon.pid"
    static let socketPath = "/tmp/coreml-daemon.sock"
    
    static func startDaemon() throws {
        if isDaemonRunning() {
            print("✅ Daemon is already running")
            return
        }
        
        print("🚀 Starting CoreML daemon...")
        
        let server = DaemonServer(socketPath: socketPath)
        
        // Create PID file
        let pid = ProcessInfo.processInfo.processIdentifier
        try String(pid).write(toFile: pidFile, atomically: true, encoding: .utf8)
        
        // Set up signal handlers for graceful shutdown
        setupSignalHandlers(server: server)
        
        // Start the server
        try server.start()
        
        // Keep the process running
        RunLoop.main.run()
    }
    
    static func stopDaemon() {
        guard let pid = getDaemonPID() else {
            print("❌ No daemon PID found")
            return
        }
        
        print("🛑 Stopping daemon (PID: \(pid))...")
        
        // Send SIGTERM to the daemon process
        kill(pid, SIGTERM)
        
        // Wait for process to exit
        var attempts = 0
        while isDaemonRunning() && attempts < 50 {
            usleep(100000) // 100ms
            attempts += 1
        }
        
        if isDaemonRunning() {
            print("⚠️  Daemon didn't stop gracefully, sending SIGKILL...")
            kill(pid, SIGKILL)
        }
        
        // Clean up files
        cleanupFiles()
        
        print("✅ Daemon stopped")
    }
    
    static func restartDaemon() {
        stopDaemon()
        // Give it a moment to clean up
        usleep(500000) // 500ms
        do {
            try startDaemon()
        } catch {
            print("❌ Failed to restart daemon: \(error)")
        }
    }
    
    static func getDaemonStatus() {
        if !isDaemonRunning() {
            print("❌ Daemon is not running")
            return
        }
        
        do {
            let client = DaemonClient(socketPath: socketPath)
            try client.connect()
            
            let response = try client.getStatus()
            client.disconnect()
            
            if response.success {
                print("✅ Daemon is running")
                if let statusData = response.data?.value as? [String: Any] {
                    if let cachedModels = statusData["cachedModels"] as? Int {
                        print("📋 Cached models: \(cachedModels)")
                    }
                    if let uptime = statusData["uptime"] as? TimeInterval {
                        print("⏱️  Uptime: \(formatUptime(uptime))")
                    }
                    if let modelPaths = statusData["modelPaths"] as? [String] {
                        if !modelPaths.isEmpty {
                            print("📁 Loaded models:")
                            for path in modelPaths {
                                print("  - \(path)")
                            }
                        }
                    }
                }
            } else {
                print("❌ Failed to get daemon status: \(response.error ?? "Unknown error")")
            }
        } catch {
            print("❌ Failed to connect to daemon: \(error)")
        }
    }
    
    static func getCacheInfo() {
        guard isDaemonRunning() else {
            print("❌ Daemon is not running")
            return
        }
        
        do {
            let client = DaemonClient(socketPath: socketPath)
            try client.connect()
            
            let response = try client.getCacheInfo()
            client.disconnect()
            
            if response.success {
                print("📋 Model Cache Information")
                print("========================")
                
                if let cacheData = response.data?.value as? [String: Any] {
                    if let modelCount = cacheData["modelCount"] as? Int {
                        print("Total models: \(modelCount)")
                    }
                    
                    if let models = cacheData["models"] as? [[String: Any]] {
                        for model in models {
                            if let path = model["path"] as? String,
                               let description = model["description"] as? String,
                               let inputFeatures = model["inputFeatures"] as? Int,
                               let outputFeatures = model["outputFeatures"] as? Int {
                                print("\n📁 \(path)")
                                print("   Description: \(description)")
                                print("   Inputs: \(inputFeatures), Outputs: \(outputFeatures)")
                            }
                        }
                    }
                }
            } else {
                print("❌ Failed to get cache info: \(response.error ?? "Unknown error")")
            }
        } catch {
            print("❌ Failed to connect to daemon: \(error)")
        }
    }
    
    static func isDaemonRunning() -> Bool {
        guard let pid = getDaemonPID() else {
            return false
        }
        
        // Check if process is still running
        return kill(pid, 0) == 0
    }
    
    private static func getDaemonPID() -> pid_t? {
        guard FileManager.default.fileExists(atPath: pidFile) else {
            return nil
        }
        
        do {
            let pidString = try String(contentsOfFile: pidFile)
            return pid_t(pidString.trimmingCharacters(in: .whitespacesAndNewlines))
        } catch {
            return nil
        }
    }
    
    private static func cleanupFiles() {
        try? FileManager.default.removeItem(atPath: pidFile)
        try? FileManager.default.removeItem(atPath: socketPath)
    }
    
    private static func setupSignalHandlers(server: DaemonServer) {
        // Handle SIGTERM and SIGINT for graceful shutdown
        // Note: This is a simplified version - production code would need more robust signal handling
        print("💡 Use Ctrl+C to stop the daemon gracefully")
    }
    
    private static func formatUptime(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        
        if hours > 0 {
            return String(format: "%dh %dm %ds", hours, minutes, secs)
        } else if minutes > 0 {
            return String(format: "%dm %ds", minutes, secs)
        } else {
            return String(format: "%ds", secs)
        }
    }
}
