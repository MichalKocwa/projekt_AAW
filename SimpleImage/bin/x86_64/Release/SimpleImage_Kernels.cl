/**********************************************************************
Copyright �2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 

/* Copy input 2D image to output 2D image */
__kernel void image2dCopy(__read_only image2d_t input, __write_only image2d_t output)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	uint4 temp = read_imageui(input, imageSampler, coord);

	write_imageui(output, coord, temp);
}

/* Copy input 3D image to 2D image */
__kernel void image3dCopy(__read_only image3d_t input, __write_only image2d_t output)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	/* Read first slice into lower half */
	uint4 temp0 = read_imageui(input, imageSampler, (int4)(coord, 0, 0));

	/* Read second slice into upper half */
	uint4 temp1 = read_imageui(input, imageSampler, (int4)((int2)(get_global_id(0), get_global_id(1) - get_global_size(1)/2), 1, 0));
	
	write_imageui(output, coord, temp0 + temp1);
}


//__kernel void ms_filter(__read_only image3d_t luv, __write_only image2d_t output, int width, int height, int spatial_radius, float color_radius_squared, int num_iters, ) {

__kernel void ms_filter(__read_only image2d_t luv, __write_only image2d_t outputImage) {
    
    int spatial_radius = 20;
    float color_radius_squared = 12.0f/255.0f/2.0f/2.0f/2.0f/2.0f/2.0f;
    int num_iters = 100;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int2 coord = (int2)(i, j);

    int width = get_global_size(0);
    int height = get_global_size(1);

    int ic = i;
    int jc = j;
    int icOld, jcOld;
    float LOld, UOld, VOld;

    uint4 temp = read_imageui(luv, imageSampler, coord);

    // Normalize to [0, 1] range
    float L = temp.x / 255.0f;
    float U = temp.y / 255.0f;
    float V = temp.z / 255.0f;

    float ms_shift = 5.0f; // initial value of mean shift

    for (int iters = 0; ms_shift > 1.0f && iters < num_iters; iters++) {
        float mi = 0;
        float mj = 0;
        float mL = 0;
        float mU = 0;
        float mV = 0;
        int num = 0;

        int ifrom = max(0, i - spatial_radius);
        int ito = min(width, i + spatial_radius + 1);
        int jfrom = max(0, j - spatial_radius);
        int jto = min(height, j + spatial_radius + 1);

        for (int jj = jfrom; jj < jto; jj++) {
            for (int ii = ifrom; ii < ito; ii++) {
                int2 coord_loop = (int2)(ii, jj);
                uint4 temp_loop = read_imageui(luv, imageSampler, coord_loop);

                float L2 = temp_loop.x / 255.0f;
                float U2 = temp_loop.y / 255.0f;
                float V2 = temp_loop.z / 255.0f;

                float dL = L2 - L;
                float dU = U2 - U;
                float dV = V2 - V;

                if (dL * dL + dU * dU + dV * dV <= color_radius_squared) {
                    mi += ii;
                    mj += jj;
                    mL += L2;
                    mU += U2;
                    mV += V2;
                    num++;
                }
            }
        }

        if (num == 0) break; // Avoid division by zero

        icOld = ic;
        jcOld = jc;
        LOld = L;
        UOld = U;
        VOld = V;

        float num_inv = 1.0f / num;
        L = mL * num_inv;
        U = mU * num_inv;
        V = mV * num_inv;
        ic = (int)(mi * num_inv + 0.5f);
        jc = (int)(mj * num_inv + 0.5f);
        int di = ic - icOld;
        int dj = jc - jcOld;
        float dL = L - LOld;
        float dU = U - UOld;
        float dV = V - VOld;

        ms_shift = di * di + dj * dj + dL * dL + dU * dU + dV * dV;
    }

    // Convert back to uchar range [0, 255]
    uint4 outputPixel = (uint4)((uchar)(L * 255.0f), (uchar)(U * 255.0f), (uchar)(V * 255.0f), 255);

    write_imageui(outputImage, coord, outputPixel);
}
