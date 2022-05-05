// RUN: iree-opt -split-input-file -pass-pipeline="vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_alloc
  vm.func @list_alloc(%arg0: i32) {
    // CHECK: %[[REF:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK: %[[REFPTR:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    %list = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    %list_dno = util.do_not_optimize(%list) : !vm.list<i32>
    // CHECK: util.do_not_optimize(%[[REFPTR]]) : !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    vm.return
  }

  // CHECK-LABEL: @my_module_list_size
  vm.func @list_size(%arg0: i32) {
    %list = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    // CHECK: %[[REF:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK: %[[REFPTR:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    %size = vm.list.size %list : (!vm.list<i32>) -> i32
    // CHECK: %[[SIZE:.+]] = emitc.call "iree_vm_list_size"(%{{.+}})
    %size_dno = util.do_not_optimize(%size) : i32
    // CHECK: util.do_not_optimize(%[[SIZE]]) : i32
    vm.return
  }
}

// -----

vm.module @my_module {
  vm.rodata private @byte_buffer dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-LABEL: @my_module_ref
  vm.export @ref
  vm.func @ref(%arg0: i32) {
    // CHECK: %[[REF:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK: %[[REFPTR:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    %buffer = vm.const.ref.rodata @byte_buffer : !vm.buffer
    %buffer_dno = util.do_not_optimize(%buffer) : !vm.buffer
    // CHECK: util.do_not_optimize(%[[REFPTR]]) : !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    vm.return
  }
}

// -----

// Test the func.func conversion, which is needed as a second step after the
// vm.func conversion. All references in the signature should be converted to
// emitc pointers.
vm.module @my_module {
  // CHECK: func @fn(%arg0: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, %arg1: i32)
  func.func @fn(%arg0 : !vm.ref<?>, %arg1 : i32) -> () {
    return
  }
}