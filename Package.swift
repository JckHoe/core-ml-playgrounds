// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CoreMLExperiment",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .watchOS(.v9),
        .tvOS(.v16)
    ],
    products: [
        .executable(name: "coreml-experiment", targets: ["CoreMLExperiment"]),
        .library(name: "CoreMLLoader", targets: ["CoreMLLoader"])
    ],
    targets: [
        .executableTarget(
            name: "CoreMLExperiment",
            dependencies: ["CoreMLLoader"]
        ),
        .target(
            name: "CoreMLLoader",
            dependencies: []
        ),
        .testTarget(
            name: "CoreMLExperimentTests",
            dependencies: ["CoreMLLoader"]
        )
    ]
)