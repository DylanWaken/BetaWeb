/*
 * GAUSSIAN SPLATTING RENDERER
 * ===========================
 * 
 * This is a WebGL-based implementation of Gaussian Splatting for real-time novel view synthesis.
 * The code renders 3D Gaussian primitives as 2D splats on screen using alpha blending.
 * 
 * KEY CONCEPTS:
 * 1. DATA PACKING FORMAT: Each splat is 32 bytes (8 floats):
 *    - Position (XYZ): 12 bytes (Float32 × 3) - 3D world coordinates
 *    - Scale (XYZ): 12 bytes (Float32 × 3) - Gaussian ellipsoid radii
 *    - Color (RGBA): 4 bytes (uint8 × 4) - Color and opacity
 *    - Rotation (IJKL): 4 bytes (uint8 × 4) - Quaternion for orientation
 * 
 * 2. RENDERING PIPELINE:
 *    - Worker thread: Depth sorting and texture packing
 *    - Vertex shader: Projects 3D Gaussians to 2D ellipses 
 *    - Fragment shader: Evaluates Gaussian kernel and alpha blending (VOLUMETRIC RENDERING)
 * 
 * 3. SORTING: Depth-based sorting using counting sort for proper alpha blending
 * 
 * 4. SPHERICAL HARMONICS: Converted to RGB colors during PLY processing
 * 
 * 5. WHERE KEY OPERATIONS HAPPEN:
 *    - DATA PACKING: generateTexture() function in worker
 *    - PRIMITIVE SORTING: runSort() function in worker  
 *    - GAUSSIAN KERNEL EVALUATION: Fragment shader
 *    - SPHERICAL HARMONICS EVALUATION: processPlyBuffer() function
 *    - VOLUMETRIC RENDERING: Fragment shader alpha blending
 */

// CAMERA PARAMETERS AND VIEW SETUP
// =================================
// Pre-defined camera positions and intrinsics for the scene
// Each camera contains:
// - id: camera identifier
// - img_name: associated image filename  
// - width/height: image dimensions
// - position: 3D world position [x, y, z]
// - rotation: 3x3 rotation matrix (row-major)
// - fx, fy: focal lengths in pixels
let cameras = [
    {
        id: 0,
        img_name: "00001",
        width: 1959,
        height: 1090,
        position: [
            -3.0089893469241797, -0.11086489695181866, -3.7527640949141428,
        ],
        rotation: [
            [0.876134201218856, 0.06925962026449776, 0.47706599800804744],
            [-0.04747421839895102, 0.9972110940209488, -0.057586739349882114],
            [-0.4797239414934443, 0.027805376500959853, 0.8769787916452908],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 1,
        img_name: "00009",
        width: 1959,
        height: 1090,
        position: [
            -2.5199776022057296, -0.09704735754873686, -3.6247725540304545,
        ],
        rotation: [
            [0.9982731285632193, -0.011928707708098955, -0.05751927260507243],
            [0.0065061360949636325, 0.9955928229282383, -0.09355533724430458],
            [0.058381769258182864, 0.09301955098900708, 0.9939511719154457],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 2,
        img_name: "00017",
        width: 1959,
        height: 1090,
        position: [
            -0.7737533667465242, -0.3364271945329695, -2.9358969417573753,
        ],
        rotation: [
            [0.9998813418672372, 0.013742375651625236, -0.0069605529394208224],
            [-0.014268370388586709, 0.996512943252834, -0.08220929105659476],
            [0.00580653013657589, 0.08229885200307129, 0.9965907801935302],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 3,
        img_name: "00025",
        width: 1959,
        height: 1090,
        position: [
            1.2198221749590001, -0.2196687861401182, -2.3183162007028453,
        ],
        rotation: [
            [0.9208648867765482, 0.0012010625395201253, 0.389880004297208],
            [-0.06298204172269357, 0.987319521752825, 0.14571693239364383],
            [-0.3847611242348369, -0.1587410451475895, 0.9092635249821667],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 4,
        img_name: "00033",
        width: 1959,
        height: 1090,
        position: [
            1.742387858893817, -0.13848225198886954, -2.0566370113193146,
        ],
        rotation: [
            [0.24669889292141334, -0.08370189346592856, -0.9654706879349405],
            [0.11343747891376445, 0.9919082664242816, -0.05700815184573074],
            [0.9624300466054861, -0.09545671285663988, 0.2541976029815521],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 5,
        img_name: "00041",
        width: 1959,
        height: 1090,
        position: [
            3.6567309419223935, -0.16470990600750707, -1.3458085590422042,
        ],
        rotation: [
            [0.2341293058324528, -0.02968330457755884, -0.9717522161434825],
            [0.10270823606832301, 0.99469554638321, -0.005638106875665722],
            [0.9667649592295676, -0.09848690996657204, 0.2359360976431732],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 6,
        img_name: "00049",
        width: 1959,
        height: 1090,
        position: [
            3.9013554243203497, -0.2597500978038105, -0.8106154188297828,
        ],
        rotation: [
            [0.6717235545638952, -0.015718162115524837, -0.7406351366386528],
            [0.055627354673906296, 0.9980224478387622, 0.029270992841185218],
            [0.7387104058127439, -0.060861588786650656, 0.6712695459756353],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 7,
        img_name: "00057",
        width: 1959,
        height: 1090,
        position: [4.742994605467533, -0.05591660945412069, 0.9500365976084458],
        rotation: [
            [-0.17042655709210375, 0.01207080756938, -0.9852964448542146],
            [0.1165090336695526, 0.9931575292530063, -0.00798543433078162],
            [0.9784581921120181, -0.1161568667478904, -0.1706667764862097],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 8,
        img_name: "00065",
        width: 1959,
        height: 1090,
        position: [4.34676307626522, 0.08168160516967145, 1.0876221470355405],
        rotation: [
            [-0.003575447631888379, -0.044792503246552894, -0.9989899137764799],
            [0.10770152645126597, 0.9931680875192705, -0.04491693593046672],
            [0.9941768441149182, -0.10775333677534978, 0.0012732004866391048],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
    {
        id: 9,
        img_name: "00073",
        width: 1959,
        height: 1090,
        position: [3.264984351114202, 0.078974937336732, 1.0117200284114904],
        rotation: [
            [-0.026919994628162257, -0.1565891128261527, -0.9872968974090509],
            [0.08444552208239385, 0.983768234577625, -0.1583319754069128],
            [0.9960643893290491, -0.0876350978794554, -0.013259786205163005],
        ],
        fy: 1164.6601287484507,
        fx: 1159.5880733038064,
    },
];

let camera = cameras[0];

/**
 * PROJECTION MATRIX COMPUTATION
 * =============================
 * Converts camera intrinsics to OpenGL projection matrix for 3D to 2D projection.
 * This matrix transforms 3D world coordinates to normalized device coordinates (NDC).
 * 
 * @param {number} fx - Focal length in x direction (pixels)
 * @param {number} fy - Focal length in y direction (pixels)  
 * @param {number} width - Image width (pixels)
 * @param {number} height - Image height (pixels)
 * @returns {Array} 4x4 projection matrix in column-major order
 */
function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;   // Near clipping plane
    const zfar = 200;    // Far clipping plane
    
    // OpenGL projection matrix for perspective projection
    // Converts from camera coordinates to normalized device coordinates
    return [
        [(2 * fx) / width, 0, 0, 0],                           // Scale X by focal length
        [0, -(2 * fy) / height, 0, 0],                         // Scale Y by focal length (flipped)
        [0, 0, zfar / (zfar - znear), 1],                      // Z mapping for depth buffer
        [0, 0, -(zfar * znear) / (zfar - znear), 0],          // Z translation for depth buffer
    ].flat(); // Flatten to 1D array for WebGL
}

/**
 * VIEW MATRIX COMPUTATION  
 * =======================
 * Converts camera pose (position + rotation) to view matrix.
 * Transforms world coordinates to camera coordinates.
 * 
 * @param {Object} camera - Camera object with position and rotation
 * @returns {Array} 4x4 view matrix in column-major order
 */
function getViewMatrix(camera) {
    const R = camera.rotation.flat();  // 3x3 rotation matrix -> 1D array
    const t = camera.position;         // 3D translation vector
    
    // Construct camera-to-world transformation matrix
    // Then return it directly (this is actually world-to-camera due to the specific format used)
    const camToWorld = [
        [R[0], R[1], R[2], 0],         // First row of rotation
        [R[3], R[4], R[5], 0],         // Second row of rotation  
        [R[6], R[7], R[8], 0],         // Third row of rotation
        [
            // Translation component (combined with rotation)
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],  // X translation
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],  // Y translation
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],  // Z translation
            1,
        ],
    ].flat();
    return camToWorld;
}

/**
 * 4x4 MATRIX MULTIPLICATION
 * =========================
 * Multiplies two 4x4 matrices in column-major order.
 * Used for combining transformations (e.g., projection * view).
 * 
 * @param {Array} a - First 4x4 matrix (16 elements)
 * @param {Array} b - Second 4x4 matrix (16 elements)  
 * @returns {Array} Result matrix (16 elements)
 */
function multiply4(a, b) {
    return [
        // First column of result
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        // Second column of result
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        // Third column of result
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        // Fourth column of result
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

/**
 * 4x4 MATRIX INVERSION
 * ====================
 * Inverts a 4x4 transformation matrix using cofactor expansion.
 * Critical for converting between camera and world coordinate systems.
 * 
 * @param {Array} a - Input 4x4 matrix (16 elements)
 * @returns {Array|null} Inverted matrix or null if singular
 */
function invert4(a) {
    // Calculate 2x2 determinants for cofactor expansion
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    
    // Calculate determinant
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null; // Matrix is singular
    
    // Calculate cofactor matrix and divide by determinant
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

/**
 * 4x4 MATRIX ROTATION
 * ===================
 * Applies rotation around arbitrary axis to existing transformation matrix.
 * Used for camera controls and navigation.
 * 
 * @param {Array} a - Input 4x4 matrix
 * @param {number} rad - Rotation angle in radians
 * @param {number} x - X component of rotation axis
 * @param {number} y - Y component of rotation axis  
 * @param {number} z - Z component of rotation axis
 * @returns {Array} Rotated matrix
 */
function rotate4(a, rad, x, y, z) {
    // Normalize rotation axis
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    
    // Calculate rotation matrix components using Rodrigues' formula
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    
    // Build 3x3 rotation matrix
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    
    // Apply rotation to first 3 columns of matrix
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16), // Keep translation unchanged
    ];
}

/**
 * 4x4 MATRIX TRANSLATION
 * ======================
 * Applies translation to existing transformation matrix.
 * Used for camera movement and positioning.
 * 
 * @param {Array} a - Input 4x4 matrix
 * @param {number} x - X translation
 * @param {number} y - Y translation
 * @param {number} z - Z translation
 * @returns {Array} Translated matrix
 */
function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12), // Keep rotation/scale unchanged
        // Update translation column
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

/**
 * WEB WORKER FOR GAUSSIAN SPLATTING PROCESSING
 * ============================================
 * This worker runs in a separate thread to handle computationally intensive tasks:
 * 1. DEPTH SORTING: Sort splats by depth for proper alpha blending
 * 2. DATA PACKING: Pack splat data into GPU-friendly texture format  
 * 3. TEXTURE GENERATION: Convert splat attributes to texture data for shaders
 * 
 * CRITICAL: This is where the core Gaussian splatting data processing happens!
 */
function createWorker(self) {
    let buffer;           // Raw splat data buffer (32 bytes per splat)
    let vertexCount = 0;  // Number of splats
    let viewProj;         // Combined view-projection matrix for depth calculation

    const TOTAL_SIZE = 16;
    const IN_BUFFER_SPLAT_SIZE = 20;
    
    // DATA PACKING FORMAT (80 bytes per splat):
    // See later buffer specifications for more details
    const rowLength = IN_BUFFER_SPLAT_SIZE * 4;
    
    let lastProj = [];              // Last projection matrix (for optimization)
    let depthIndex = new Uint32Array();  // Sorted indices by depth
    let lastVertexCount = 0;        // Last vertex count (for optimization)

    // Float16 encoding utilities for efficient GPU texture packing
    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    /**
     * FLOAT TO HALF-PRECISION CONVERSION
     * =================================
     * Converts 32-bit float to 16-bit half precision for efficient GPU storage.
     * Used in covariance matrix packing.
     */
    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;           // Extract sign bit
        var exp = (f >> 23) & 0x00ff;            // Extract exponent 
        var frac = f & 0x007fffff;               // Extract mantissa

        var newExp;
        if (exp == 0) {
            newExp = 0;                          // Zero/denormal
        } else if (exp < 113) {
            newExp = 0;                          // Too small, flush to zero
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;                  // Normal range
        } else {
            newExp = 31;                         // Infinity/NaN
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    /**
     * PACK TWO HALF-PRECISION FLOATS INTO 32-BIT INTEGER
     * =================================================
     * Efficiently stores two 16-bit values in one 32-bit texture element.
     */
    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    /**
     * TEXTURE GENERATION - CRITICAL DATA PACKING FUNCTION
     * ==================================================
     * THIS IS WHERE DATA IS PACKED FOR GPU CONSUMPTION!
     * 
     * Converts raw splat buffer into GPU texture format:
     * 
     * Input Buffer Layout:
     *      [0] (32bit) - X
     *      [1] (32bit) - Y
     *      [2] (32bit) - Z
     *      [3] (32bit) - Beta Parameter
     * 
     *      [4] (32bit) - Scale X
     *      [5] (32bit) - Scale Y
     *      [6] (32bit) - Scale Z
     * 
     *      [7] (32bit) - qx
     *      [8] (32bit) - qy
     *      [9] (32bit) - qz
     *      [10] (32bit) - qw
     * 
     *      [11] (32bit) - RGBA Color (uint8 × 4)
     * 
     *      [12] (32bit) - SB1: Theta
     *      [13]  (32bit) - SB1: Phi
     *      [14]  (32bit) - SB1: Beta Param
     *      [15]  (32bit) - SB1: RGBA Color (uint8 × 4)
     * 
     *      [16] (32bit) - SB2: Theta
     *      [17] (32bit) - SB2: Phi
     *      [18] (32bit) - SB2: Beta Param
     *      [19] (32bit) - SB2: RGBA Color (uint8 × 4)
     * 
     * Output TextureLayout:
     *      TEXEL 0  (4x32 bits): Position (XYZ) - Beta Parameter (foat 32)
     *      TEXEL 1  (4x32 bits): Covariance Matrix (6 values packed as 3 half2 pairs), Base Color (RGBA)
     *      TEXEL 2  (4x32 bits): Theta, Phi, Beta Param, RGBA Color (uint8 × 4)
     *      TEXEL 3  (4x32 bits): Theta, Phi, Beta Param, RGBA Color (uint8 × 4)
     * 
     * COVARIANCE COMPUTATION: This is where 3D Gaussian ellipsoids are
     * converted to 2D screen-space covariance for splatting!
     */
    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);  // Float view of buffer
        const u_buffer = new Uint8Array(buffer);    // Byte view of buffer

        // Texture dimensions - each splat needs 2 horizontal texels
        var texwidth = 1024 * 4;  // Width in texels
        var texheight = Math.ceil((4 * vertexCount) / texwidth);  // Height needed
        var texdata = new Uint32Array(texwidth * texheight * 4); // RGBA32UI format

        var texdata_c = new Uint8Array(texdata.buffer);           // Byte view
        var texdata_f = new Float32Array(texdata.buffer);         // Float view

        // PROCESS EACH SPLAT: Convert from .splat format to texture format
        for (let i = 0; i < vertexCount; i++) {
            // FIRST TEXEL: Position (XYZ) - Beta Parameter as floats
            texdata_f[TOTAL_SIZE * i + 0] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 0];  // X position
            texdata_f[TOTAL_SIZE * i + 1] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 1];  // Y position  
            texdata_f[TOTAL_SIZE * i + 2] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 2];  // Z position
            texdata_f[TOTAL_SIZE * i + 3] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 3];  // Beta parameter

            // EXTRACT SCALE AND ROTATION FROM BUFFER
            let scale = [
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 4 + 0],  // Scale X
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 4 + 1],  // Scale Y
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 4 + 2],  // Scale Z
            ];
            let rot = [
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 7 + 0],  // qx
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 7 + 1],  // qy
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 7 + 2],  // qz
                f_buffer[IN_BUFFER_SPLAT_SIZE * i + 7 + 3],  // qw
            ];

            // COMPUTE 3D COVARIANCE MATRIX: Σ = R * S * S^T * R^T
            // This defines the 3D Gaussian ellipsoid shape and orientation
            const M = [
                // Rotation matrix from quaternion, scaled by scale factors
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),  // M[0,0]
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),        // M[0,1]
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),        // M[0,2]

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),        // M[1,0]
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),  // M[1,1]
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),        // M[1,2]

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),        // M[2,0]
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),        // M[2,1]
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),  // M[2,2]
            ].map((k, i) => k * scale[Math.floor(i / 3)]);  // Apply scaling

            // COMPUTE 3D COVARIANCE: σ = M * M^T (symmetric 3x3 matrix)
            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],    // σ[0,0]
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],    // σ[0,1] 
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],    // σ[0,2]
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],    // σ[1,1]
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],    // σ[1,2]
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],    // σ[2,2]
            ];

            // SECOND TEXEL: Pack 3D covariance matrix (6 unique values) 
            // as 3 pairs of half-precision floats (×4 scale for numerical stability)
            texdata[TOTAL_SIZE * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);  // σ00, σ01
            texdata[TOTAL_SIZE * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);  // σ02, σ11  
            texdata[TOTAL_SIZE * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);  // σ12, σ22

            // RGBA Color (uint8 × 4)
            texdata_c[(TOTAL_SIZE * i + 7) * 4 + 0] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 10) * 4 + 0];  // R
            texdata_c[(TOTAL_SIZE * i + 7) * 4 + 1] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 10) * 4 + 1];  // G
            texdata_c[(TOTAL_SIZE * i + 7) * 4 + 2] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 10) * 4 + 2];  // B
            texdata_c[(TOTAL_SIZE * i + 7) * 4 + 3] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 10) * 4 + 3];  // A
            
            // THIRD TEXEL: Theta, Phi, Beta Param, RGBA Color (uint8 × 4)
            texdata_f[TOTAL_SIZE * i + 8] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 12];  // Theta
            texdata_f[TOTAL_SIZE * i + 9] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 13];  // Phi
            texdata_f[TOTAL_SIZE * i + 10] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 14];  // Beta Param
            texdata_c[(TOTAL_SIZE * i + 11) * 4 + 0] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 15) * 4 + 0];  // R
            texdata_c[(TOTAL_SIZE * i + 11) * 4 + 1] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 15) * 4 + 1];  // G
            texdata_c[(TOTAL_SIZE * i + 11) * 4 + 2] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 15) * 4 + 2];  // B
            texdata_c[(TOTAL_SIZE * i + 11) * 4 + 3] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 15) * 4 + 3];  // A

            // FOURTH TEXEL: Theta, Phi, Beta Param, RGBA Color (uint8 × 4)
            texdata_f[TOTAL_SIZE * i + 12] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 16];  // Theta
            texdata_f[TOTAL_SIZE * i + 13] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 17];  // Phi
            texdata_f[TOTAL_SIZE * i + 14] = f_buffer[IN_BUFFER_SPLAT_SIZE * i + 18];  // Beta Param
            texdata_c[(TOTAL_SIZE * i + 15) * 4 + 0] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 19) * 4 + 0];  // R
            texdata_c[(TOTAL_SIZE * i + 15) * 4 + 1] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 19) * 4 + 1];  // G
            texdata_c[(TOTAL_SIZE * i + 15) * 4 + 2] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 19) * 4 + 2];  // B
            texdata_c[(TOTAL_SIZE * i + 15) * 4 + 3] = u_buffer[(IN_BUFFER_SPLAT_SIZE * i + 19) * 4 + 3];  // A
        }

        // Send packed texture data back to main thread
        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    /**
     * DEPTH SORTING ALGORITHM - CRITICAL FOR ALPHA BLENDING
     * ====================================================
     * THIS IS WHERE PRIMITIVE SORTING HAPPENS!
     * 
     * Sorts all splats by depth (back-to-front) for proper alpha blending.
     * Uses a fast single-pass 16-bit counting sort algorithm.
     * 
     * WHY SORTING IS NEEDED:
     * - Gaussian splatting uses alpha blending for composition
     * - Alpha blending requires back-to-front rendering order
     * - Without proper sorting, transparent objects appear incorrect
     * 
     * @param {Array} viewProj - Combined view-projection matrix for depth calculation
     */
    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        
        // OPTIMIZATION: Skip sorting if camera hasn't moved much
        if (lastVertexCount == vertexCount) {
            // Calculate dot product between current and last view directions
            let dot =
                lastProj[2] * viewProj[2] +     // Z-axis comparison
                lastProj[6] * viewProj[6] +     // (camera forward direction)
                lastProj[10] * viewProj[10];
            // If camera direction is nearly the same, skip re-sorting
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            // Vertex count changed, regenerate texture
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
        
        // STEP 1: CALCULATE DEPTH FOR EACH SPLAT
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        
        for (let i = 0; i < vertexCount; i++) {
            // Transform splat position to camera space and extract Z coordinate
            // This is the depth calculation: depth = viewProj * position.z
            let depth =
                ((viewProj[2] * f_buffer[IN_BUFFER_SPLAT_SIZE * i + 0] +      // X contribution
                    viewProj[6] * f_buffer[IN_BUFFER_SPLAT_SIZE * i + 1] +    // Y contribution  
                    viewProj[10] * f_buffer[IN_BUFFER_SPLAT_SIZE * i + 2]) *  // Z contribution
                    4096) |   // Scale and convert to integer for sorting
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // STEP 2: 16-BIT COUNTING SORT IMPLEMENTATION
        // This is a very fast O(n) sorting algorithm for integer keys
        
        // Map depth values to 16-bit range [0, 65535]
        let depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);  // Histogram buckets
        
        // Count occurrences of each depth value
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;  // Increment count for this depth
        }
        
        // Calculate starting positions for each depth value  
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        
        // STEP 3: BUILD SORTED INDEX ARRAY
        // Create array of splat indices sorted by depth (back-to-front)
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        // Cache current projection matrix for next frame optimization
        lastProj = viewProj;
        
        // Send sorted indices back to main thread for rendering
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    /**
     * PLY FILE PROCESSING - SPHERICAL HARMONICS EVALUATION
     * ===================================================
     * THIS IS WHERE SPHERICAL HARMONICS ARE EVALUATED!
     * 
     * Processes PLY files (standard 3D format) and converts them to the .splat format.
     * Key operations:
     * 1. Parse PLY header to understand data layout
     * 2. Convert spherical harmonic coefficients to RGB colors
     * 3. Sort splats by importance (size × opacity) 
     * 4. Pack data into efficient 32-byte format
     * 
     * @param {ArrayBuffer} inputBuffer - Raw PLY file data
     * @returns {ArrayBuffer} Processed splat data buffer
     */
    function processPlyBuffer(inputBuffer) {
        // STEP 1: PARSE PLY HEADER
        const ubuf = new Uint8Array(inputBuffer);
        // Read first 10KB as header (should be enough for any reasonable PLY)
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = header.indexOf(header_end);
        if (header_end_index < 0)
            throw new Error("Unable to read .ply file header");
            
        // Extract vertex count from header
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        console.log("Vertex Count", vertexCount);
        
        // STEP 2: PARSE PROPERTY DEFINITIONS
        // PLY files define data layout in header - parse this to understand format
        let row_offset = 0,
            offsets = {},    // Byte offset for each property
            types = {};      // Data type for each property
            
        // Map PLY data types to JavaScript DataView methods
        const TYPE_MAP = {
            double: "getFloat64",
            int: "getInt32",
            uint: "getUint32", 
            float: "getFloat32",
            short: "getInt16",
            ushort: "getUint16",
            uchar: "getUint8",
        };
        
        // Parse each property line in header
        for (let prop of header
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            const [p, type, name] = prop.split(" ");
            const arrayType = TYPE_MAP[type] || "getInt8";
            types[name] = arrayType;
            offsets[name] = row_offset;
            row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
        }
        console.log("Bytes per row", row_offset, types, offsets);

        // STEP 3: CREATE DATA VIEW FOR BINARY DATA
        let dataView = new DataView(
            inputBuffer,
            header_end_index + header_end.length,
        );
        let row = 0;
        
        // Create proxy object for easy property access
        const attrs = new Proxy(
            {},
            {
                get(target, prop) {
                    if (!types[prop]) throw new Error(prop + " not found");
                    return dataView[types[prop]](
                        row * row_offset + offsets[prop],
                        true,  // little-endian
                    );
                },
            },
        );

        // STEP 4: CALCULATE IMPORTANCE FOR SORTING
        // Sort splats by visual importance (size × opacity) to prioritize visible ones
        console.time("calculate importance");
        let sizeList = new Float32Array(vertexCount);
        let sizeIndex = new Uint32Array(vertexCount);
        for (row = 0; row < vertexCount; row++) {
            sizeIndex[row] = row;
            if (!types["scale_0"]) continue;
            
            // Calculate splat volume: exp(scale_x) × exp(scale_y) × exp(scale_z)
            const size =
                Math.exp(attrs.scale_0) *
                Math.exp(attrs.scale_1) *
                Math.exp(attrs.scale_2);
            // Calculate opacity: sigmoid(opacity_logit)    
            const opacity = 1 / (1 + Math.exp(-attrs.opacity));
            // Importance = volume × opacity (bigger and more opaque = more important)
            sizeList[row] = size * opacity;
        }
        console.timeEnd("calculate importance");

        // STEP 5: SORT BY IMPORTANCE (most important first)
        console.time("sort");
        sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
        console.timeEnd("sort");

        // STEP 6: BUILD OUTPUT BUFFER IN NEW FORMAT
        // Convert sorted PLY data to 20-element format (80 bytes per splat)
        const rowLength = 20 * 4;  // 80 bytes total (20 × 4 bytes)
        const buffer = new ArrayBuffer(rowLength * vertexCount);

        console.time("build buffer");
        for (let j = 0; j < vertexCount; j++) {
            row = sizeIndex[j];  // Process in importance order

            // Create views for different sections of the 80-byte splat data
            const position = new Float32Array(buffer, j * rowLength, 3);      // [0-2]: XYZ
            const beta = new Float32Array(buffer, j * rowLength + 12, 1);     // [3]: Beta Parameter
            const scales = new Float32Array(buffer, j * rowLength + 16, 3);   // [4-6]: Scale XYZ
            const rotation = new Float32Array(buffer, j * rowLength + 28, 4); // [7-10]: Quaternion
            const baseColor = new Uint32Array(buffer, j * rowLength + 44, 1); // [11]: RGBA Color
            const sb1GeoData = new Float32Array(buffer, j * rowLength + 48, 3);  // [12-14]: SB1 data
            const sb1ColorData = new Uint32Array(buffer, j * rowLength + 60, 1);  // [15]: SB1 color
            const sb2GeoData = new Float32Array(buffer, j * rowLength + 64, 3);  // [16-18]: SB2 data
            const sb2ColorData = new Uint32Array(buffer, j * rowLength + 76, 1);  // [19]: SB2 color

            // COPY POSITION DATA
            position[0] = attrs.x;
            position[1] = attrs.y;
            position[2] = attrs.z;

            // COPY BETA PARAMETER
            beta[0] = attrs.beta;

            // HANDLE SCALE AND ROTATION DATA
            if (types["scale_0"]) {
                // Normalize quaternion and store as float
                const qlen = Math.sqrt(
                    attrs.rot_0 ** 2 +
                        attrs.rot_1 ** 2 +
                        attrs.rot_2 ** 2 +
                        attrs.rot_3 ** 2,
                );

                rotation[0] = attrs.rot_0 / qlen;  // Normalized quaternion as float
                rotation[1] = attrs.rot_1 / qlen;
                rotation[2] = attrs.rot_2 / qlen;
                rotation[3] = attrs.rot_3 / qlen;

                // Convert log-space scales to linear
                scales[0] = Math.exp(attrs.scale_0);
                scales[1] = Math.exp(attrs.scale_1);
                scales[2] = Math.exp(attrs.scale_2);
            } else {
                // Default values for missing scale/rotation
                scales[0] = 0.01;
                scales[1] = 0.01;
                scales[2] = 0.01;

                rotation[0] = 1.0;  // Identity quaternion
                rotation[1] = 0.0;
                rotation[2] = 0.0;
                rotation[3] = 0.0;
            }

            // STEP 7: SPHERICAL HARMONICS TO RGB CONVERSION
            // Convert spherical harmonic coefficients to base color
            let r, g, b, a;
            if (types["sh0_0"]) {
                // Convert spherical harmonic coefficients to RGB
                const SH_C0 = 0.28209479177387814;  // Normalization constant for SH basis function Y₀⁰
                
                // Evaluate SH at view direction (simplified to DC term only)
                r = Math.max(0, Math.min(255, (0.5 + SH_C0 * attrs.sh0_0) * 255));
                g = Math.max(0, Math.min(255, (0.5 + SH_C0 * attrs.sh0_1) * 255));
                b = Math.max(0, Math.min(255, (0.5 + SH_C0 * attrs.sh0_2) * 255));
            } else {
                // Default RGB if no SH data
                r = 128;
                g = 128;
                b = 128;
            }
            
            // HANDLE OPACITY
            if (types["opacity"]) {
                // Convert opacity logit to [0,1] range then to [0,255]
                a = Math.max(0, Math.min(255, (1 / (1 + Math.exp(-attrs.opacity))) * 255));
            } else {
                a = 255;  // Fully opaque
            }

            // Pack RGBA into uint32
            baseColor[0] = (a << 24) | (b << 16) | (g << 8) | r;

            // STEP 8: PACK SPHERICAL BETA PARAMETERS
            // SB1 data: Theta, Phi, Beta, RGBA
            sb1GeoData[0] = attrs.sb_params_0;  // Theta
            sb1GeoData[1] = attrs.sb_params_1;  // Phi
            sb1GeoData[2] = attrs.sb_params_2;  // Beta

            // Convert float spherical beta parameters to uint8 for efficient packing
            let sb1_r = Math.max(0, Math.min(255, Math.floor(attrs.sb_params_3 * 255)));  // Convert float to uint8
            let sb1_g = Math.max(0, Math.min(255, Math.floor(attrs.sb_params_4 * 255)));  // Convert float to uint8  
            let sb1_b = Math.max(0, Math.min(255, Math.floor(attrs.sb_params_5 * 255)));  // Convert float to uint8
            
            let sb2_r = Math.max(0, Math.min(255, Math.floor(attrs.sb_params_9 * 255)));  // Convert float to uint8
            let sb2_g = Math.max(0, Math.min(255, Math.floor(attrs.sb_params_10 * 255))); // Convert float to uint8
            let sb2_b = Math.max(0, Math.min(255, Math.floor(attrs.sb_params_11 * 255))); // Convert float to uint8

            sb1ColorData[0] = (sb1_r << 24) | (sb1_g << 16) | (sb1_b << 8) | 0xFF; // RGBA packed

            // SB2 data: Theta, Phi, Beta, RGBA
            sb2GeoData[0] = attrs.sb_params_6;  // Theta
            sb2GeoData[1] = attrs.sb_params_7;  // Phi
            sb2GeoData[2] = attrs.sb_params_8;  // Beta

            sb2ColorData[0] = (sb2_r << 24) | (sb2_g << 16) | (sb2_b << 8) | 0xFF; // RGBA packed
        }
        console.timeEnd("build buffer");
        return buffer;
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.ply) {
            vertexCount = 0;
            runSort(viewProj);
            buffer = processPlyBuffer(e.data.ply);
            vertexCount = Math.floor(buffer.byteLength / rowLength);
            postMessage({ buffer: buffer, save: !!e.data.save });
        } else if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        }
    };
}

/*
 * VERTEX SHADER - 3D GAUSSIAN TO 2D SPLAT PROJECTION
 * =================================================
 * This shader transforms 3D Gaussian ellipsoids into 2D screen-space ellipses.
 * 
 * KEY OPERATIONS:
 * 1. Project 3D Gaussian center to screen space
 * 2. Compute 2D covariance matrix via Jacobian transformation  
 * 3. Eigendecomposition to find ellipse axes
 * 4. Generate quad vertices for splat rendering
 * 
 * INPUT DATA FORMAT (from texture):
 * - Texel (2*i, y):   Position XYZ + Color RGBA
 * - Texel (2*i+1, y): 3D Covariance matrix (6 values as 3×half2)
 */
const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

// UNIFORMS: Data passed from JavaScript
uniform highp usampler2D u_texture;  // Packed splat data texture
uniform mat4 projection, view;       // Camera matrices
uniform vec2 focal;                  // Camera focal lengths (fx, fy)
uniform vec2 viewport;               // Screen dimensions
uniform vec3 cameraPos;              // Camera position

// INPUTS: Per-vertex attributes  
in vec2 position;  // Quad vertex position [-2,-2] to [2,2]
in int index;      // Splat index (which Gaussian we're rendering)

// OUTPUTS: Data passed to fragment shader
out vec4 vColor;    // Splat color and depth-based alpha
out vec2 vPosition; // Position within quad for Gaussian evaluation
out float vBeta;    // Beta parameter

// Helper function to convert spherical coordinates to cartesian
vec3 sphericalToCartesian(float theta, float phi) {
    return vec3(
        sin(theta) * cos(phi),
        sin(theta) * sin(phi),
        cos(theta)
    );
}

// Helper function to compute spherical gaussian contribution
vec3 computeSphericalBeta(vec3 dir, vec3 meanDir, float beta, vec3 color) {
    // Normalize direction vector
    float dirNorm = length(dir);
    vec3 dirNormalized = dir / dirNorm;
    
    // Compute dot product between normalized direction and mean direction
    float dot = dot(dirNormalized, meanDir);
    
    // Compute beta term - only contribute if dot product is positive
    float betaTerm = 0.0;
    if (dot > 0.0) {
        betaTerm = pow(dot, 4.0 * exp(beta));
    }
    
    // Return color contribution
    return betaTerm * color;
}

// Main spherical gaussian evaluation function
vec3 evaluateSphericalBeta(vec3 dir, vec3 baseColor, vec3 SB1Geo, vec4 SB1Color, vec3 SB2Geo, vec4 SB2Color) {
    // Start with base color
    vec3 result = baseColor;
    
    // Convert spherical coordinates to cartesian for first spherical gaussian
    vec3 meanDir1 = sphericalToCartesian(SB1Geo.x, SB1Geo.y);
    result += computeSphericalBeta(dir, meanDir1, SB1Geo.z, SB1Color.rgb);
    
    // Convert spherical coordinates to cartesian for second spherical gaussian
    vec3 meanDir2 = sphericalToCartesian(SB2Geo.x, SB2Geo.y);
    result += computeSphericalBeta(dir, meanDir2, SB2Geo.z, SB2Color.rgb);
    
    return result;
}



// MAIN FUNCTION
void main () {
    // Each splat occupies 4 consecutive texels horizontally
    int texelsPerSplat = 4;
    int NumSplatsPerRow = 1024;  // Maximum texels per row
    int splatX = (index % NumSplatsPerRow) * texelsPerSplat;
    int splatY = index / NumSplatsPerRow;
    
    // ------------------------------------------------------------
    // 1Fetch splat data from texture
    // ------------------------------------------------------------

    // TEXEL 0: Position (XYZ) + Beta Parameter
    // TEXEL 1: Covariance Matrix (6 values packed as 3 half2 pairs) + Base Color (RGBA)
    // TEXEL 2: Theta, Phi, Beta Param, RGBA Color (uint8 × 4)
    // TEXEL 3: Theta, Phi, Beta Param, RGBA Color (uint8 × 4)
    uvec4 texel0 = texelFetch(u_texture, ivec2(splatX, splatY), 0);
    uvec4 texel1 = texelFetch(u_texture, ivec2(splatX + 1, splatY), 0);
    uvec4 texel2 = texelFetch(u_texture, ivec2(splatX + 2, splatY), 0);
    uvec4 texel3 = texelFetch(u_texture, ivec2(splatX + 3, splatY), 0);

    vec3 pos = uintBitsToFloat(texel0.xyz);
    float beta = uintBitsToFloat(texel0.w);

    // Unpack 6 covariance values from 3×32-bit integers
    vec2 u1 = unpackHalf2x16(texel1.x), u2 = unpackHalf2x16(texel1.y), u3 = unpackHalf2x16(texel1.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);  // 3D covariance matrix

    // Unpack base color from texel1
    vec4 color = vec4(
        float((texel1.w >> 0) & 0xffu) / 255.0,
        float((texel1.w >> 8) & 0xffu) / 255.0,
        float((texel1.w >> 16) & 0xffu) / 255.0,
        float((texel1.w >> 24) & 0xffu) / 255.0
    );

    // Spherical Beta Parameters from texel2 and texel3
    vec3 SB1Geo = uintBitsToFloat(texel2.xyz);
    vec4 SB1Color = vec4(
        float((texel2.w >> 0) & 0xffu) / 255.0,
        float((texel2.w >> 8) & 0xffu) / 255.0,
        float((texel2.w >> 16) & 0xffu) / 255.0,
        float((texel2.w >> 24) & 0xffu) / 255.0
    );

    vec3 SB2Geo = uintBitsToFloat(texel3.xyz);
    vec4 SB2Color = vec4(
        float((texel3.w >> 0) & 0xffu) / 255.0,
        float((texel3.w >> 8) & 0xffu) / 255.0,
        float((texel3.w >> 16) & 0xffu) / 255.0,
        float((texel3.w >> 24) & 0xffu) / 255.0
    );

    // ------------------------------------------------------------
    // 2. Projective Transform and Cull
    // ------------------------------------------------------------

    // Projective Transform
    vec4 cam = view * vec4(pos, 1);  // Transform to camera space
    vec4 pos2d = projection * cam;   // Project to clip space

    // STEP 3: FRUSTUM CULLING
    // Cull splats outside view frustum for performance
    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // Place outside clip space
        return;
    }

    // ------------------------------------------------------------
    // 3. 2D Projection of Covariance Matrix
    // ------------------------------------------------------------
    
    // Compute Jacobian of perspective projection
    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),  // ∂u/∂x, ∂u/∂y, ∂u/∂z
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),  // ∂v/∂x, ∂v/∂y, ∂v/∂z  
        0., 0., 0.                                                   // No depth derivatives needed
    );

    // Apply view transformation to Jacobian
    mat3 T = transpose(mat3(view)) * J;
    
    // Transform 3D covariance to 2D: Σ' = T * Σ * T^T
    mat3 cov2d = transpose(T) * Vrk * T;

    // STEP 6: EIGENDECOMPOSITION FOR ELLIPSE AXES
    // Convert 2D covariance matrix to ellipse major/minor axes
    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;         // Mean eigenvalue
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));  // Eigenvalue difference
    float lambda1 = mid + radius, lambda2 = mid - radius;   // Actual eigenvalues

    // Cull if covariance is degenerate (ellipse too thin)
    if(lambda2 < 0.0) return;
    
    // Compute eigenvectors (ellipse axes directions)
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;        // Major axis (scaled)
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);  // Minor axis

    // ------------------------------------------------------------
    // 4. Cumpute Spherical Beta Parameters
    // ------------------------------------------------------------

    // Extract camera position from view matrix (last column of inverse view matrix)
    vec3 viewingDirection = normalize(cameraPos - pos);

    // evaluate the spherical gaussian
    vec3 result = evaluateSphericalBeta(viewingDirection, color.rgb, SB1Geo, SB1Color, SB2Geo, SB2Color);

    // normalize the result
    result = normalize(result);

    // ------------------------------------------------------------
    // 5. Prepare output data
    // ------------------------------------------------------------

    // STEP 7: PREPARE OUTPUT DATA
    // Depth-based alpha blending weight + color from texture
    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4(result, 1.0);
    vBeta = beta;
    vPosition = position;  // Pass quad position for Gaussian evaluation

    // STEP 8: GENERATE QUAD VERTEX
    // Transform quad vertex to screen space using ellipse axes
    vec2 vCenter = vec2(pos2d) / pos2d.w;  // Normalized device coordinates
    gl_Position = vec4(
        vCenter
        + position.x * majorAxis / viewport    // Scale by major axis
        + position.y * minorAxis / viewport,   // Scale by minor axis
        0.0, 1.0);

}
`.trim();

/*
 * FRAGMENT SHADER - GAUSSIAN KERNEL EVALUATION & VOLUMETRIC RENDERING
 * ==================================================================
 * THIS IS WHERE THE GAUSSIAN KERNEL IS EVALUATED AND VOLUMETRIC RENDERING HAPPENS!
 * 
 * This shader:
 * 1. Evaluates the 2D Gaussian function at each pixel
 * 2. Applies alpha blending for volumetric composition
 * 3. Produces the final splat contribution to the image
 * 
 * MATHEMATICAL FOUNDATION:
 * - Gaussian function: G(x) = exp(-0.5 * x^T * Σ^(-1) * x)
 * - Alpha blending: C = Σ(αᵢ * Cᵢ * Πⱼ<ᵢ(1-αⱼ))
 */
const fragmentShaderSource = `
#version 300 es
precision highp float;

// INPUTS: From vertex shader
in vec4 vColor;     // Splat color (RGB) + depth weight (A) 
in vec2 vPosition;  // Position within quad [-2,-2] to [2,2]
in float vBeta;    // Beta parameter

// OUTPUT: Final pixel color
out vec4 fragColor;

void main () {
    // STEP 1: EVALUATE GAUSSIAN KERNEL
    // Calculate squared distance from splat center in ellipse space
    // vPosition is in normalized ellipse coordinates where the ellipse has unit scale
    float A = -dot(vPosition, vPosition);
    
    // STEP 2: GAUSSIAN CUTOFF
    // Discard pixels beyond 1
    // This prevents rendering pixels with negligible contribution
    if (A < -1.0) discard;

    float betaVal = 4.0 * exp(vBeta);
    
    // STEP 3: GAUSSIAN EVALUATION  
    // THIS IS THE CORE GAUSSIAN KERNEL EVALUATION!
    // B = exp(A) * α where A = -||x||² in ellipse space
    float B = pow(1.0 - A, betaVal);

    // STEP 4: VOLUMETRIC RENDERING OUTPUT
    // THIS IS WHERE VOLUMETRIC/SPLATTING RENDERING HAPPENS!
    // Output premultiplied alpha for proper blending:
    // - RGB channels: color weighted by alpha (B * color)  
    // - Alpha channel: transparency for next layer (B)
    //
    // The GPU will blend this with existing pixels using:
    // final_color = (1-dst_alpha)*src_color + dst_color
    // final_alpha = (1-dst_alpha)*src_alpha + dst_alpha
    fragColor = vec4(B * vColor.rgb, B);
}
`.trim();

let defaultViewMatrix = [
    0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
    0.03, 6.55, 1,
];
let viewMatrix = defaultViewMatrix;

/**
 * MAIN RENDERING FUNCTION
 * =======================
 * Sets up the entire Gaussian splatting rendering pipeline:
 * 1. Initialize WebGL context and shaders
 * 2. Set up data loading and worker communication  
 * 3. Configure input handling for camera controls
 * 4. Run the main rendering loop
 */
async function main() {
    // CAMERA AND VIEW SETUP
    let carousel = true;  // Auto-rotate camera mode
    const params = new URLSearchParams(location.search);
    
    // Try to load saved camera view from URL hash
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {}
    
    // SPLAT DATA LOADING SETUP
    // Only load data if URL parameter is provided
    let splatData = new Uint8Array(0);
    let reader = null;
    let req = null;
    
    const urlParam = params.get("url");
    if (urlParam) {
        const url = new URL(urlParam, window.location.origin);
        
        // Fetch splat data with streaming support
        req = await fetch(url, {
            mode: "cors",
            credentials: "omit",
        });
        console.log(req);
        if (req.status != 200)
            throw new Error(req.status + " Unable to load " + req.url);
        
        reader = req.body.getReader();
        splatData = new Uint8Array(req.headers.get("content-length"));
    }

    // RENDERING PARAMETERS
    const rowLength = 20 * 4;  // 80 bytes per splat (new format with spherical beta parameters)

    // Adaptive downsampling based on dataset size and device capability
    const downsample =
        splatData.length / rowLength > 500000 ? 1 : 1 / devicePixelRatio;
    console.log(splatData.length / rowLength, downsample);

    // WORKER THREAD SETUP
    // Create worker for background processing (sorting, texture generation)
    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    // DOM ELEMENT REFERENCES
    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    // WEBGL CONTEXT AND SHADER SETUP
    // ==============================
    const gl = canvas.getContext("webgl2", {
        antialias: false,  // Disable antialiasing for performance
    });

    // COMPILE VERTEX SHADER
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    // COMPILE FRAGMENT SHADER  
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    // LINK SHADER PROGRAM
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    // CONFIGURE BLENDING FOR VOLUMETRIC RENDERING
    // ===========================================
    // This is critical for proper Gaussian splatting alpha composition!
    gl.disable(gl.DEPTH_TEST); // Disable depth testing (we handle depth via sorting)

    // Enable alpha blending with specific blend function for volumetric rendering
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,  // Source RGB factor: (1 - dst_alpha)
        gl.ONE,                  // Dest RGB factor: 1
        gl.ONE_MINUS_DST_ALPHA,  // Source alpha factor: (1 - dst_alpha)  
        gl.ONE,                  // Dest alpha factor: 1
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    // GET UNIFORM LOCATIONS
    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");
    const u_cameraPos = gl.getUniformLocation(program, "cameraPos");

    // SETUP VERTEX BUFFERS
    // ===================
    // Quad vertices for splat rendering (each splat renders as a quad)
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    // SETUP SPLAT DATA TEXTURE
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    // SETUP INSTANCE INDEX BUFFER (for sorting)
    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);  // One index per instance (splat)

    // RESIZE HANDLER
    // =============
    const resize = () => {
        // Update camera uniforms
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));

        // Recompute projection matrix for new viewport
        projectionMatrix = getProjectionMatrix(
            camera.fx,
            camera.fy,
            innerWidth,
            innerHeight,
        );

        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

        // Resize canvas with optional downsampling
        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();

    worker.onmessage = (e) => {
        if (e.data.buffer) {
            splatData = new Uint8Array(e.data.buffer);
            const splatCount = Math.floor(splatData.length / rowLength);
            
            if (e.data.save) {
                showFileInfo(`🎉 PLY converted! Downloading ${splatCount.toLocaleString()} splats...`);
                const blob = new Blob([splatData.buffer], {
                    type: "application/octet-stream",
                });
                const link = document.createElement("a");
                link.download = "model.splat";
                link.href = URL.createObjectURL(blob);
                document.body.appendChild(link);
                link.click();
                setTimeout(() => document.body.removeChild(link), 100);
            } else {
                showFileInfo(`✅ Ready! Rendering ${splatCount.toLocaleString()} splats`);
            }
        } else if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            
            // Store latest texture data for debugging
            latestTexData = { texdata, texwidth, texheight };
            
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_S,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_T,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA32UI,
                texwidth,
                texheight,
                0,
                gl.RGBA_INTEGER,
                gl.UNSIGNED_INT,
                texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex, viewProj } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    let activeKeys = [];
    let currentCameraIndex = 0;
    let latestTexData = null;

    // Helper function to unpack half precision floats
    function unpackHalf2x16(packed) {
        const half1 = packed & 0xFFFF;
        const half2 = (packed >>> 16) & 0xFFFF;
        
        function halfToFloat(half) {
            const sign = (half & 0x8000) ? -1 : 1;
            const exp = (half & 0x7C00) >> 10;
            const frac = half & 0x03FF;
            
            if (exp === 0) {
                return sign * Math.pow(2, -14) * (frac / 1024);
            } else if (exp === 31) {
                return frac ? NaN : sign * Infinity;
            } else {
                return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
            }
        }
        
        return [halfToFloat(half1), halfToFloat(half2)];
    }

    // Debug function to print first primitive data
    function debugFirstPrimitive() {
        if (!latestTexData) {
            console.log("No texture data available yet");
            return;
        }
        
        const { texdata, texwidth } = latestTexData;
        const TOTAL_SIZE = 16; // 4 texels per splat
        
        console.log("=== DEBUG: First Primitive Data ===");
        
        // Each splat occupies 4 consecutive texels horizontally
        const primitiveIndex = 0;
        const splatX = primitiveIndex * 4;
        const splatY = 0;
        
        // Get the 4 texels for this primitive
        const texelOffset = splatY * texwidth + splatX;
        
        // TEXEL 0: Position (XYZ) + Beta Parameter
        const texel0_offset = (texelOffset + 0) * 4;
        const texdata_f = new Float32Array(texdata.buffer);
        const texdata_u8 = new Uint8Array(texdata.buffer);
        
        console.log("TEXEL 0 - Position + Beta:");
        console.log("  Position X:", texdata_f[texel0_offset + 0]);
        console.log("  Position Y:", texdata_f[texel0_offset + 1]);
        console.log("  Position Z:", texdata_f[texel0_offset + 2]);
        console.log("  Beta Parameter:", texdata_f[texel0_offset + 3]);
        
        // TEXEL 1: Covariance Matrix + Base Color
        const texel1_offset = (texelOffset + 1) * 4;
        console.log("\nTEXEL 1 - Covariance Matrix + Base Color:");
        
        // Unpack covariance matrix (6 values as 3 half2 pairs)
        const cov1 = unpackHalf2x16(texdata[texel1_offset + 0]);
        const cov2 = unpackHalf2x16(texdata[texel1_offset + 1]);
        const cov3 = unpackHalf2x16(texdata[texel1_offset + 2]);
        
        console.log("  Covariance σ00:", cov1[0] / 4.0); // Divide by 4 (scale factor)
        console.log("  Covariance σ01:", cov1[1] / 4.0);
        console.log("  Covariance σ02:", cov2[0] / 4.0);
        console.log("  Covariance σ11:", cov2[1] / 4.0);
        console.log("  Covariance σ12:", cov3[0] / 4.0);
        console.log("  Covariance σ22:", cov3[1] / 4.0);
        
        // Base color RGBA (uint8 × 4)
        const baseColorOffset = (texel1_offset + 3) * 4;
        console.log("  Base Color R:", texdata_u8[baseColorOffset + 0]);
        console.log("  Base Color G:", texdata_u8[baseColorOffset + 1]);
        console.log("  Base Color B:", texdata_u8[baseColorOffset + 2]);
        console.log("  Base Color A:", texdata_u8[baseColorOffset + 3]);
        
        // TEXEL 2: SB1 Parameters
        const texel2_offset = (texelOffset + 2) * 4;
        console.log("\nTEXEL 2 - SB1 Spherical Beta Parameters:");
        console.log("  SB1 Theta:", texdata_f[texel2_offset + 0]);
        console.log("  SB1 Phi:", texdata_f[texel2_offset + 1]);
        console.log("  SB1 Beta:", texdata_f[texel2_offset + 2]);
        
        // SB1 Color RGBA (uint8 × 4)
        const sb1ColorOffset = (texel2_offset + 3) * 4;
        console.log("  SB1 Color R:", texdata_u8[sb1ColorOffset + 0]);
        console.log("  SB1 Color G:", texdata_u8[sb1ColorOffset + 1]);
        console.log("  SB1 Color B:", texdata_u8[sb1ColorOffset + 2]);
        console.log("  SB1 Color A:", texdata_u8[sb1ColorOffset + 3]);
        
        // TEXEL 3: SB2 Parameters
        const texel3_offset = (texelOffset + 3) * 4;
        console.log("\nTEXEL 3 - SB2 Spherical Beta Parameters:");
        console.log("  SB2 Theta:", texdata_f[texel3_offset + 0]);
        console.log("  SB2 Phi:", texdata_f[texel3_offset + 1]);
        console.log("  SB2 Beta:", texdata_f[texel3_offset + 2]);
        
        // SB2 Color RGBA (uint8 × 4)
        const sb2ColorOffset = (texel3_offset + 3) * 4;
        console.log("  SB2 Color R:", texdata_u8[sb2ColorOffset + 0]);
        console.log("  SB2 Color G:", texdata_u8[sb2ColorOffset + 1]);
        console.log("  SB2 Color B:", texdata_u8[sb2ColorOffset + 2]);
        console.log("  SB2 Color A:", texdata_u8[sb2ColorOffset + 3]);
        
        console.log("=== End Debug ===");
    }

    window.addEventListener("keydown", (e) => {
        // if (document.activeElement != document.body) return;
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key);
            camera = cameras[currentCameraIndex];
            viewMatrix = getViewMatrix(camera);
        }
        if (["-", "_"].includes(e.key)) {
            currentCameraIndex =
                (currentCameraIndex + cameras.length - 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        if (["+", "="].includes(e.key)) {
            currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        camid.innerText = "cam  " + currentCameraIndex;
        if (e.code == "KeyV") {
            location.hash =
                "#" +
                JSON.stringify(
                    viewMatrix.map((k) => Math.round(k * 100) / 100),
                );
            camid.innerText = "";
        } else if (e.code === "KeyP") {
            carousel = true;
            camid.innerText = "";
        } else if (e.code === "KeyO") {
            debugFirstPrimitive();
        }
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            carousel = false;
            e.preventDefault();
            const lineHeight = 10;
            const scale =
                e.deltaMode == 1
                    ? lineHeight
                    : e.deltaMode == 2
                      ? innerHeight
                      : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / innerWidth,
                    (e.deltaY * scale) / innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
                // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
                // let preY = inv[13];
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / innerHeight,
                );
                // inv[13] = preY;
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            // let postAngle = Math.atan2(inv[0], inv[10])
            // inv = rotate4(inv, postAngle - preAngle, 0, 0, 1)
            // console.log(postAngle)
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            // inv = rotateY(inv, );
            // let preY = inv[13];
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / innerWidth,
                0,
                (10 * (e.clientY - startY)) / innerHeight,
            );
            // inv[13] = preY;
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    let altX = 0,
        altY = 0;
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                // console.log('beep')
                carousel = false;
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                // inv = translate4(inv,  -x, -y, -z);
                // inv = translate4(inv,  x, y, z);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // alert('beep')
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                // inv = translate4(inv,  0, 0, d);
                inv = rotate4(inv, dtheta, 0, 0, 1);

                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

                // let preY = inv[13];
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                // inv[13] = preY;

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    let jumpDelta = 0;
    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;

    window.addEventListener("gamepadconnected", (e) => {
        const gp = navigator.getGamepads()[e.gamepad.index];
        console.log(
            `Gamepad connected at index ${gp.index}: ${gp.id}. It has ${gp.buttons.length} buttons and ${gp.axes.length} axes.`,
        );
    });
    window.addEventListener("gamepaddisconnected", (e) => {
        console.log("Gamepad disconnected");
    });

    let leftGamepadTrigger, rightGamepadTrigger;

    const frame = (now) => {
        let inv = invert4(viewMatrix);
        let shiftKey =
            activeKeys.includes("Shift") ||
            activeKeys.includes("ShiftLeft") ||
            activeKeys.includes("ShiftRight");

        if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translate4(inv, 0, -0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translate4(inv, 0, 0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft"))
            inv = translate4(inv, -0.03, 0, 0);
        //
        if (activeKeys.includes("ArrowRight"))
            inv = translate4(inv, 0.03, 0, 0);
        // inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        let isJumping = activeKeys.includes("Space");
        for (let gamepad of gamepads) {
            if (!gamepad) continue;

            const axisThreshold = 0.1; // Threshold to detect when the axis is intentionally moved
            const moveSpeed = 0.06;
            const rotateSpeed = 0.02;

            // Assuming the left stick controls translation (axes 0 and 1)
            if (Math.abs(gamepad.axes[0]) > axisThreshold) {
                inv = translate4(inv, moveSpeed * gamepad.axes[0], 0, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[1]) > axisThreshold) {
                inv = translate4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
                carousel = false;
            }
            if (gamepad.buttons[12].pressed || gamepad.buttons[13].pressed) {
                inv = translate4(
                    inv,
                    0,
                    -moveSpeed *
                        (gamepad.buttons[12].pressed -
                            gamepad.buttons[13].pressed),
                    0,
                );
                carousel = false;
            }

            if (gamepad.buttons[14].pressed || gamepad.buttons[15].pressed) {
                inv = translate4(
                    inv,
                    -moveSpeed *
                        (gamepad.buttons[14].pressed -
                            gamepad.buttons[15].pressed),
                    0,
                    0,
                );
                carousel = false;
            }

            // Assuming the right stick controls rotation (axes 2 and 3)
            if (Math.abs(gamepad.axes[2]) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * gamepad.axes[2], 0, 1, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[3]) > axisThreshold) {
                inv = rotate4(inv, -rotateSpeed * gamepad.axes[3], 1, 0, 0);
                carousel = false;
            }

            let tiltAxis = gamepad.buttons[6].value - gamepad.buttons[7].value;
            if (Math.abs(tiltAxis) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * tiltAxis, 0, 0, 1);
                carousel = false;
            }
            if (gamepad.buttons[4].pressed && !leftGamepadTrigger) {
                camera =
                    cameras[(cameras.indexOf(camera) + 1) % cameras.length];
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            if (gamepad.buttons[5].pressed && !rightGamepadTrigger) {
                camera =
                    cameras[
                        (cameras.indexOf(camera) + cameras.length - 1) %
                            cameras.length
                    ];
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            leftGamepadTrigger = gamepad.buttons[4].pressed;
            rightGamepadTrigger = gamepad.buttons[5].pressed;
            if (gamepad.buttons[0].pressed) {
                isJumping = true;
                carousel = false;
            }
            if (gamepad.buttons[3].pressed) {
                carousel = true;
            }
        }

        if (
            ["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))
        ) {
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(
                inv,
                activeKeys.includes("KeyJ")
                    ? -0.05
                    : activeKeys.includes("KeyL")
                      ? 0.05
                      : 0,
                0,
                1,
                0,
            );
            inv = rotate4(
                inv,
                activeKeys.includes("KeyI")
                    ? 0.05
                    : activeKeys.includes("KeyK")
                      ? -0.05
                      : 0,
                1,
                0,
                0,
            );
            inv = translate4(inv, 0, 0, -d);
        }

        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(defaultViewMatrix);

            const t = Math.sin((Date.now() - start) / 5000);
            inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
            inv = rotate4(inv, -0.6 * t, 0, 1, 0);

            viewMatrix = invert4(inv);
        }

        if (isJumping) {
            jumpDelta = Math.min(1, jumpDelta + 0.05);
        } else {
            jumpDelta = Math.max(0, jumpDelta - 0.05);
        }

        let inv2 = invert4(viewMatrix);
        inv2 = translate4(inv2, 0, -jumpDelta, 0);
        inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
        let actualViewMatrix = invert4(inv2);

        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        worker.postMessage({ view: viewProj });

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            
            // Extract camera position from inverse view matrix
            let invView = invert4(actualViewMatrix);
            let cameraPos = [invView[12], invView[13], invView[14]];
            gl.uniform3fv(u_cameraPos, cameraPos);
            
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }
        const progress = (100 * vertexCount) / (splatData.length / rowLength);
        if (progress < 100) {
            document.getElementById("progress").style.width = progress + "%";
        } else {
            document.getElementById("progress").style.display = "none";
        }
        fps.innerText = Math.round(avgFps) + " fps";
        if (isNaN(currentCameraIndex)) {
            camid.innerText = "";
        }
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    frame();

    const isPly = (splatData) =>
        splatData[0] == 112 &&
        splatData[1] == 108 &&
        splatData[2] == 121 &&
        splatData[3] == 10;

    const showFileInfo = (message, isSuccess = true) => {
        const fileInfo = document.getElementById("fileInfo");
        fileInfo.innerText = message;
        fileInfo.style.display = "block";
        fileInfo.style.background = isSuccess ? "rgba(0, 150, 0, 0.9)" : "rgba(150, 0, 0, 0.9)";
        setTimeout(() => {
            fileInfo.style.display = "none";
        }, 3000);
    };

    const selectFile = (file) => {
        const fr = new FileReader();
        
        if (/\.json$/i.test(file.name)) {
            showFileInfo("Loading camera file...");
            fr.onload = () => {
                try {
                    cameras = JSON.parse(fr.result);
                    viewMatrix = getViewMatrix(cameras[0]);
                    projectionMatrix = getProjectionMatrix(
                        camera.fx / downsample,
                        camera.fy / downsample,
                        canvas.width,
                        canvas.height,
                    );
                    gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

                    showFileInfo(`✅ Loaded ${cameras.length} camera positions`);
                    console.log("Loaded Cameras");
                } catch (err) {
                    showFileInfo("❌ Failed to load camera file", false);
                }
            };
            fr.readAsText(file);
        } else if (/\.ply$/i.test(file.name)) {
            showFileInfo("🔄 Processing PLY file...");
            stopLoading = true;
            fr.onload = () => {
                splatData = new Uint8Array(fr.result);
                const splatCount = Math.floor(splatData.length / rowLength);
                
                if (isPly(splatData)) {
                    showFileInfo(`✅ Converting PLY file with ${splatCount.toLocaleString()} points...`);
                    worker.postMessage({ ply: splatData.buffer, save: true });
                } else {
                    showFileInfo("❌ Invalid PLY file format", false);
                }
            };
            fr.readAsArrayBuffer(file);
        } else {
            // Handle .splat files or other formats
            showFileInfo("Loading splat file...");
            stopLoading = true;
            fr.onload = () => {
                splatData = new Uint8Array(fr.result);
                const splatCount = Math.floor(splatData.length / rowLength);
                
                showFileInfo(`✅ Loaded ${splatCount.toLocaleString()} splats`);
                worker.postMessage({
                    buffer: splatData.buffer,
                    vertexCount: splatCount,
                });
            };
            fr.readAsArrayBuffer(file);
        }
    };

    window.addEventListener("hashchange", (e) => {
        try {
            viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
            carousel = false;
        } catch (err) {}
    });

    // Enhanced drag and drop with visual feedback
    const dragOverlay = document.getElementById("dragOverlay");
    let dragCounter = 0;

    const preventDefault = (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Explicitly set dropEffect to prevent browser's default download behavior
        if (e.dataTransfer) {
            e.dataTransfer.dropEffect = "copy";
        }
    };

    // Prevent default drag and drop behavior on the entire document
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, preventDefault, false);
    });

    // Add specific handlers with our custom logic
    document.addEventListener("dragenter", (e) => {
        dragCounter++;
        if (e.dataTransfer && e.dataTransfer.types.includes("Files")) {
            dragOverlay.classList.add("active");
            e.dataTransfer.dropEffect = "copy";
        }
    }, false);

    document.addEventListener("dragover", (e) => {
        if (e.dataTransfer && e.dataTransfer.types.includes("Files")) {
            dragOverlay.classList.add("active");
            e.dataTransfer.dropEffect = "copy";
        }
    }, false);

    document.addEventListener("dragleave", (e) => {
        dragCounter--;
        if (dragCounter === 0) {
            dragOverlay.classList.remove("active");
        }
    }, false);

    document.addEventListener("drop", (e) => {
        dragCounter = 0;
        dragOverlay.classList.remove("active");
        
        if (e.dataTransfer && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            console.log(`Dropped file: ${file.name} (${file.size} bytes)`);
            selectFile(file);
        }
    }, false);

    // Only attempt to load data if we have a reader (URL was provided)
    if (reader) {
        let bytesRead = 0;
        let lastVertexCount = -1;
        let stopLoading = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done || stopLoading) break;

            splatData.set(value, bytesRead);
            bytesRead += value.length;

            if (vertexCount > lastVertexCount) {
                if (!isPly(splatData)) {
                    worker.postMessage({
                        buffer: splatData.buffer,
                        vertexCount: Math.floor(bytesRead / rowLength),
                    });
                }
                lastVertexCount = vertexCount;
            }
        }
        if (!stopLoading) {
            if (isPly(splatData)) {
                // ply file magic header means it should be handled differently
                worker.postMessage({ ply: splatData.buffer, save: false });
            } else {
                worker.postMessage({
                    buffer: splatData.buffer,
                    vertexCount: Math.floor(bytesRead / rowLength),
                });
            }
        }
    }
}

main().catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});
