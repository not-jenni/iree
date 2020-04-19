// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

// CHECK: [[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: [[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK: func @reshape_collapse_single_dim
// CHECK: %[[IN:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x28x28x1xf32>
// CHECK: %[[OUT:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x784xf32>
module {
  func @reshape_collapse_single_dim() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<1x28x28x1xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<1x784xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// CHECK: %[[RESULT:.*]] = linalg.reshape %[[IN]] [
// CHECK-SAME: [[MAP0]], [[MAP1]]
// CHECK-SAME: ] : memref<{{.*}}xf32> into memref<{{.*}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: [[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: [[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK: [[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: func @reshape_collapse
// CHECK: %[[IN:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x2x2x3xf32>
// CHECK: %[[OUT:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4x3xf32>
module {
  func @reshape_collapse() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x2x2x3xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<2x2x2x3xf32>) -> tensor<2x4x3xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<2x4x3xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// CHECK: %[[RESULT:.*]] = linalg.reshape %[[IN]] [
// CHECK-SAME: [[MAP0]], [[MAP1]], [[MAP2]]
// CHECK-SAME: ] : memref<{{.*}}xf32> into memref<{{.*}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: [[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: [[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: func @reshape_expand
// CHECK: %[[IN:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<2x8xf32>
// CHECK: %[[OUT:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<2x4x2xf32>
module {
  func @reshape_expand() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<2x8xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<2x4x2xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// CHECK: %[[RESULT:.*]] = linalg.reshape %[[IN]] [
// CHECK-SAME: [[MAP0]], [[MAP1]]
// CHECK-SAME: ] : memref<{{.*}}xf32> into memref<{{.*}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: [[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @reshape_single_expand
// CHECK: %[[IN:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<8xf32>
// CHECK: %[[OUT:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x4x2xf32>
module {
  func @reshape_single_expand() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<8xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<8xf32>) -> tensor<1x4x2xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<1x4x2xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// CHECK: %[[RESULT:.*]] = linalg.reshape %[[IN]] [
// CHECK-SAME: [[MAP0]]
// CHECK-SAME: ] : memref<{{.*}}xf32> into memref<{{.*}}xf32>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])
