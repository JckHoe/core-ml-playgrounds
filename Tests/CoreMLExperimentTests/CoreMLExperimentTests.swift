import XCTest
@testable import CoreMLLoader

final class CoreMLExperimentTests: XCTestCase {
    func testModelLoaderCreation() {
        let loader = ModelLoader()
        XCTAssertNotNil(loader)
    }
}