// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CoreMLExperiment",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .watchOS(.v10),
        .tvOS(.v17)
    ],
    products: [
        .executable(name: "coreml-experiment", targets: ["CoreMLExperiment"]),
        .library(name: "CoreMLLoader", targets: ["CoreMLLoader"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .executableTarget(
            name: "CoreMLExperiment",
            dependencies: ["CoreMLLoader"]
        ),
        .target(
            name: "CoreMLLoader",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "CoreMLExperimentTests",
            dependencies: ["CoreMLLoader"]
        )
    ]
)