// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s

// check that we don't hoist convert_layout above its operand definition.
// CHECK-LABEL: convert_cannot_hoist
//       CHECK:   %[[CVTS:.+]] = ttg.local_alloc
//       CHECK:   ttg.local_load %[[CVTS]]
//       CHECK:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_hoist(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = ttg.local_alloc %9 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %11 = ttg.local_load %10 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %11, %cst_0, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: no_move_alloc_for_scalar_src
//       CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
//       CHECK: %[[SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<32x32xf32, #blocked>
//       CHECK: ttg.async_wait {num = 0 : i32}
//       CHECK: ttg.local_alloc %[[SPLAT]] : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_move_alloc_for_scalar_src() {
    %cst = arith.constant 0.000000e+00 : f32
    %t = tt.splat %cst : f32 -> tensor<32x32xf32, #blocked>
    ttg.async_wait {num = 0 : i32}
    %alloc = ttg.local_alloc %t : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_dealloc
//       CHECK: ttg.async_wait {num = 0 : i32}
//       CHECK: ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//       CHECK: ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//       CHECK: %3 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1_negative
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #{{.*}}, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: ttng.arrive_barrier
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #{{.*}}, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1_negative(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    ttng.arrive_barrier %bar, 2, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// check that we don't sink convert_layout if it has multi users
// CHECK-LABEL: convert_cannot_sink
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
//       CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_sink(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A0 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS0 = ttg.local_alloc %A0 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD0 = ttg.local_load %AS0 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD0, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %A1 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS1 = ttg.local_alloc %A1 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD1 = ttg.local_load %AS1 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %13 = tt.dot %AD1, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#transposed = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @alloc_stays_after_completion_wait
  tt.func public @alloc_stays_after_completion_wait(%ptr: tensor<128x64x!tt.ptr<f16>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %true = arith.constant true
    %false = arith.constant false
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: %[[OUTER_VALUE:.*]] = tt.load
    %outer_value = tt.load %ptr : tensor<128x64x!tt.ptr<f16>, #blocked>
    // CHECK: scf.for
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      // CHECK: %[[VALUE:.*]] = tt.load
      %value = tt.load %ptr : tensor<128x64x!tt.ptr<f16>, #blocked>
      %has_previous = arith.cmpi sgt, %i, %c0 : i32
      // CHECK: scf.if
      scf.if %has_previous {
        // CHECK: ttng.wait_barrier
        ttng.wait_barrier %bar, %c0, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      // Reuse this allocation only after the previous iteration's MMA finishes.
      // CHECK: %[[A:.*]] = ttg.local_alloc %[[VALUE]]
      %a = ttg.local_alloc %value : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // Keep an allocation with a loop-invariant initializer inside the loop too.
      // CHECK: %[[B_STORAGE:.*]] = ttg.local_alloc %[[OUTER_VALUE]]
      %b_storage = ttg.local_alloc %outer_value : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b = ttg.memdesc_trans %b_storage {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #transposed, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma %[[A]],
      %mma = ttng.tc_gen5_mma %a, %b, %acc[], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #transposed, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_commit %bar, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    }
    ttng.wait_barrier %bar, %c1, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

// Check that a local_load is sunk down to its user when the tile it loads
// would otherwise sit in registers while another tensor is loaded from global
// memory. %BD stays where it is: no tt.load lies between it and its user.
// CHECK-LABEL: sink_local_load_past_load
//       CHECK: %[[AS:.+]] = ttg.local_alloc
//  CHECK-NEXT: %[[B:.+]] = tt.load
//  CHECK-NEXT: %[[BS:.+]] = ttg.local_alloc %[[B]]
//  CHECK-NEXT: %[[BD:.+]] = ttg.local_load %[[BS]]
//  CHECK-NEXT: %[[AD:.+]] = ttg.local_load %[[AS]]
//  CHECK-NEXT: %[[A2:.+]] = arith.mulf %[[AD]]
//  CHECK-NEXT: %[[B2:.+]] = arith.mulf %[[BD]]
//  CHECK-NEXT: tt.dot %[[A2]], %[[B2]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_local_load_past_load(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %sa = arith.constant dense<2.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %sb = arith.constant dense<3.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %B = tt.load %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A2 = arith.mulf %AD, %sa : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %B2 = arith.mulf %BD, %sb : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %A2, %B2, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Check that a local_load feeding tt.dot directly is not sunk past a tt.load.
// CHECK-LABEL: local_load_for_dot_not_sunk
//       CHECK: %[[AS:.+]] = ttg.local_alloc
//  CHECK-NEXT: %[[AD:.+]] = ttg.local_load %[[AS]]
//  CHECK-NEXT: tt.load
//       CHECK: tt.dot %[[AD]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_load_for_dot_not_sunk(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %AS = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %AD = ttg.local_load %AS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %B = tt.load %arg1 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Check that a local_load is not sunk past a store to shared memory.
// CHECK-LABEL: local_load_cannot_sink_past_store
//       CHECK: %[[OLD:.+]] = ttg.local_load
//  CHECK-NEXT: %[[X:.+]] = tt.load
//  CHECK-NEXT: ttg.local_store %[[X]]
//  CHECK-NEXT: arith.mulf %[[OLD]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_load_cannot_sink_past_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_store %arg1, %buf : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %old = ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %x = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    ttg.local_store %x, %buf : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %y = arith.mulf %old, %arg1 : tensor<32x32xf32, #blocked>
    tt.return %y : tensor<32x32xf32, #blocked>
  }
}

// -----

// Check that a local_load is not sunk past the dealloc of its buffer.
// CHECK-LABEL: local_load_cannot_sink_past_dealloc
//       CHECK: %[[V:.+]] = ttg.local_load
//  CHECK-NEXT: %[[X:.+]] = tt.load
//  CHECK-NEXT: ttg.local_dealloc
//  CHECK-NEXT: arith.mulf %[[V]], %[[X]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_load_cannot_sink_past_dealloc(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_store %arg1, %buf : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %v = ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %x = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    ttg.local_dealloc %buf : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %y = arith.mulf %v, %x : tensor<32x32xf32, #blocked>
    tt.return %y : tensor<32x32xf32, #blocked>
  }
}

// -----

// Check that a local_load with more than one use is not sunk.
// CHECK-LABEL: local_load_multi_use_not_sunk
//       CHECK: %[[V:.+]] = ttg.local_load
//  CHECK-NEXT: %[[X:.+]] = tt.load
//  CHECK-NEXT: %[[Y:.+]] = arith.mulf %[[V]], %[[X]]
//  CHECK-NEXT: arith.addf %[[V]], %[[Y]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_load_multi_use_not_sunk(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %buf = ttg.local_alloc %arg1 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %v = ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %x = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %y = arith.mulf %v, %x : tensor<32x32xf32, #blocked>
    %z = arith.addf %v, %y : tensor<32x32xf32, #blocked>
    tt.return %z : tensor<32x32xf32, #blocked>
  }
}
