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

    // Unpack base color from texel1 (R,G,B,A)
    vec4 color = vec4(
        float((texel1.w >> 24) & 0xffu) / 255.0,
        float((texel1.w >> 16) & 0xffu) / 255.0,
        float((texel1.w >> 8) & 0xffu) / 255.0,
        float((texel1.w >> 0) & 0xffu) / 255.0
    );

    // Spherical Beta Parameters from texel2 and texel3
    vec3 SB1Geo = uintBitsToFloat(texel2.xyz);
    vec4 SB1Color = vec4(
        float((texel2.w >> 24) & 0xffu) / 255.0,
        float((texel2.w >> 16) & 0xffu) / 255.0,
        float((texel2.w >> 8) & 0xffu) / 255.0,
        float((texel2.w >> 0) & 0xffu) / 255.0
    );

    vec3 SB2Geo = uintBitsToFloat(texel3.xyz);
    vec4 SB2Color = vec4(
        float((texel3.w >> 24) & 0xffu) / 255.0,
        float((texel3.w >> 16) & 0xffu) / 255.0,
        float((texel3.w >> 8) & 0xffu) / 255.0,
        float((texel3.w >> 0) & 0xffu) / 255.0
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
    // vec3 result = evaluateSphericalBeta(viewingDirection, color.rgb, SB1Geo, SB1Color, SB2Geo, SB2Color);
    vec3 result = color.rgb;

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