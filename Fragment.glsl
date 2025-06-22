#version 300 es
precision highp float;

// INPUTS: From vertex shader
in vec4 vColor;     // Splat color (RGB) + depth weight (A) 
in vec2 vPosition;  // Position within quad [-2,-2] to [2,2]
in float vBeta;    // Beta parameter

// OUTPUT: Final pixel color
out vec4 fragColor;

void main () {
    // STEP 1: EVALUATE BETA  KERNEL
    // Calculate squared distance from splat center in ellipse space
    // vPosition is in normalized ellipse coordinates where the ellipse has unit scale
    float A = dot(vPosition, vPosition);
    
    // Discard pixels beyond 1
    // This prevents rendering pixels with negligible contribution
    if (A > 1.0) discard;

    float betaVal = 4.0 * exp(vBeta);
    
    // STEP 3: BETA EVALUATION  
    // THIS IS THE CORE BETA KERNEL EVALUATION!
    // B = pow(1.0 - A, betaVal) * α where A = -||x||² in ellipse space
    float B = pow(1.0 - A, betaVal) * vColor.a;

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