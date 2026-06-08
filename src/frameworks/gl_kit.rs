/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Stub for `GLKit.framework/GLKit`.
//!
//! GLKit provides math utilities, texture loading, and view controllers for
//! OpenGL ES apps. Zombie Catchers and other iOS 5+ games link it primarily
//! for `GLKMatrix4Identity` and related constants.
//!
//! This stub satisfies the Mach-O dependency and exports the most commonly
//! referenced constants. Real GLKit rendering classes are not implemented.

use crate::dyld::{ConstantExports, FunctionExports, HostConstant};
use crate::mem::{ConstVoidPtr, SafeRead};
use crate::Environment;

// GLKMatrix4Identity — 4x4 identity matrix of 32-bit floats (16 × 4 = 64 bytes).
// Layout matches Apple's `_GLKMatrix4 { float m[16]; }`.
#[repr(C)]
struct GLKMatrix4 {
    m: [f32; 16],
}
unsafe impl SafeRead for GLKMatrix4 {}

fn glk_matrix4_identity(env: &mut Environment) -> ConstVoidPtr {
    let identity = GLKMatrix4 {
        m: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };
    env.mem.alloc_and_write(identity).cast_void().cast_const()
}

pub const CONSTANTS: ConstantExports = &[(
    "_GLKMatrix4Identity",
    HostConstant::Custom(glk_matrix4_identity),
)];

pub const FUNCTIONS: FunctionExports = &[];

pub const DYLIB: crate::dyld::HostDylib = crate::dyld::HostDylib {
    path: "/System/Library/Frameworks/GLKit.framework/GLKit",
    aliases: &[],
    class_exports: &[],
    constant_exports: &[CONSTANTS],
    function_exports: &[FUNCTIONS],
};
