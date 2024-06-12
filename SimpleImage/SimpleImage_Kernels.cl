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


__kernel void to_hsv(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);

		pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));

		pixel.x = pixel.x / 255.0;
		pixel.y = pixel.y / 255.0;
		pixel.z = pixel.z / 255.0;

		double cmax = max(pixel.x, max(pixel.y, pixel.z)); 
		double cmin = min(pixel.x, min(pixel.y, pixel.z)); 
		double diff = cmax - cmin; 
		double h = -1, s = -1; 

		if (cmax == cmin) 
			h = 0; 
			else if (cmax == pixel.x) 
			h = fmod(60 * ((pixel.y - pixel.z) / diff) + 360, 360); 
	
		else if (cmax == pixel.y) 
			h = fmod(60 * ((pixel.z - pixel.x) / diff) + 120, 360); 
	
		else if (cmax == pixel.z) 
			h = fmod(60 * ((pixel.x - pixel.y) / diff) + 240, 360); 
	
		if (cmax == 0) 
			s = 0; 
		else
			s = (diff / cmax) * 100; 

    	double v = cmax * 100;

		pixel.x = convert_float(h);
		pixel.y = convert_float(s);
		pixel.z = convert_float(v);
		//float wyn = clamp(sum*g, 0.0f, 255.0f);
		
		write_imageui(outputImage, coord, convert_uint4(pixel));

}

/*__kernel void to_LUV2(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);

	pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));

    float r = pixel.x / 255.0f;
    float g = pixel.y / 255.0f;
    float b = pixel.z / 255.0f;

    r = (r > 0.04045f) ? pow((r + 0.055f) / 1.055f, 2.4f) : (r / 12.92f);
    g = (g > 0.04045f) ? pow((g + 0.055f) / 1.055f, 2.4f) : (g / 12.92f);
    b = (b > 0.04045f) ? pow((b + 0.055f) / 1.055f, 2.4f) : (b / 12.92f);

    float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

    float Xn = 0.95047f;
    float Yn = 1.0f;
    float Zn = 1.08883f;

    float u_prime = (4.0f * X) / (X + 15.0f * Y + 3.0f * Z);
    float v_prime = (9.0f * Y) / (X + 15.0f * Y + 3.0f * Z);

    float Yr = Y / Yn;
    float L = (Yr > 0.008856f) ? (116.0f * cbrt(Yr) - 16.0f) : (903.3f * Yr);
    float u = 13.0f * L * (u_prime - 0.1978398f); 
    float v = 13.0f * L * (v_prime - 0.4683363f); 

    L = clamp(L, 0.0f, 100.0f) * 255.0f / 100.0f;
    u = clamp(u, -134.0f, 220.0f) + 134.0f;
    v = clamp(v, -140.0f, 122.0f) + 140.0f;

    pixel.x = L;
    pixel.y = u;
    pixel.z = v;

	write_imageui(outputImage, coord, convert_uint4(pixel));
}*/


__kernel void to_lab(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);

	pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));

    float r = pixel.x / 255.0f;
    float g = pixel.y / 255.0f;
    float b = pixel.z / 255.0f;

    r = (r > 0.04045f) ? pow((r + 0.055f) / 1.055f, 2.4f) : (r / 12.92f);
    g = (g > 0.04045f) ? pow((g + 0.055f) / 1.055f, 2.4f) : (g / 12.92f);
    b = (b > 0.04045f) ? pow((b + 0.055f) / 1.055f, 2.4f) : (b / 12.92f);

    float X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

    float Xn = 0.95047f;
    float Yn = 1.0f;
    float Zn = 1.08883f;

    X /= Xn;
    Y /= Yn;
    Z /= Zn;

    X = (X > 0.008856f) ? pow(X, 1.0f/3.0f) : (7.787f * X + 16.0f / 116.0f);
    Y = (Y > 0.008856f) ? pow(Y, 1.0f/3.0f) : (7.787f * Y + 16.0f / 116.0f);
    Z = (Z > 0.008856f) ? pow(Z, 1.0f/3.0f) : (7.787f * Z + 16.0f / 116.0f);

    float L = 116.0f * Y - 16.0f;
    float a = 500.0f * (X - Y);
    float bb = 200.0f * (Y - Z);

    L = clamp(L, 0.0f, 100.0f) * 255.0f / 100.0f;
    a = clamp(a, -128.0f, 127.0f) + 128.0f;
    bb = clamp(bb, -128.0f, 127.0f) + 128.0f;

    pixel.x = L;
    pixel.y = a;
    pixel.z = bb;

	write_imageui(outputImage, coord, convert_uint4(pixel));
}


__kernel void to_YCrCb(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);

		pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));



		float Y = 0.257 * pixel.x + 0.504 * pixel.y + 0.098 * pixel.z + 16;

		float Cb = -0.148 * pixel.x - 0.291 * pixel.y + 0.439 * pixel.z + 128;

		float Cr = 0.439 * pixel.x - 0.368 * pixel.y - 0.071 * pixel.z + 128;

		pixel.x = Y;
		pixel.y = Cb;
		pixel.z = Cr;
				
		write_imageui(outputImage, coord, convert_uint4(pixel));

}


__kernel void to_LUV(__read_only image2d_t rgb, __write_only image2d_t luv)
{


	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);

	pixel = convert_float4(read_imageui(rgb, imageSampler, (int2)(coord.x, coord.y)));

    uchar R = pixel.x;
    uchar G = pixel.y;
    uchar B = pixel.z;

    // RGB to LUV conversion logic
    float var_R = ((float)R) / 255;
    float var_G = ((float)G) / 255;
    float var_B = ((float)B) / 255;

    float var_min = min(var_R, min(var_G, var_B));    // Min. value of RGB
    float var_max = max(var_R, max(var_G, var_B));    // Max. value of RGB
    float del_Max = var_max - var_min;               // Delta RGB value

    float L, U, V;
    float var_Y = (var_R + var_G + var_B) / 3.0;

    if (var_max > 0.008856) {
        L = (116.0 * var_Y) - 16.0;
    } else {
        L = 903.3 * var_Y;
    }

    if (del_Max == 0) {
        U = 0;
        V = 0;
    } else {
        U = (4 * var_R) / (var_R + (15 * var_G) + (3 * var_B));
        V = (9 * var_G) / (var_R + (15 * var_G) + (3 * var_B));
    }

    // Scale to byte range
    pixel.x = (uchar)(L * 255 / 100);
    pixel.y = (uchar)((U + 134) * 255 / 354);
    pixel.z = (uchar)((V + 140) * 255 / 262);

	write_imageui(luv, coord, convert_uint4(pixel));

}


__kernel void LUV2RGB(__read_only image2d_t luv, __write_only image2d_t rgb)
{

     int2 coord = (int2)(get_global_id(0), get_global_id(1));

    // Read LUV pixel values
    uint4 luvPixel = read_imageui(luv, imageSampler, coord);

    int LL = (int)luvPixel.x;
    int uu = (int)luvPixel.y;
    int vv = (int)luvPixel.z;

    float L = (float)(LL * 100.0f / 255.0f);
    float u = (float)((uu) * 354.0f / 255.0f - 134);
    float v = (float)((vv) * 262.0f / 255.0f - 140);


    float X, Y, Z, ud, vd, u0, v0, TEMP, L1;
    int r, g, b;
    float eps = 216.0 / 24389.0;
    float k = 24389.0 / 27.0;
    float Xr = 0.964221;
    float Yr = 1.0;
    float Zr = 0.825211;

    u0 = 4.0 * Xr / (Xr + 15.0 * Yr + 3.0 * Zr);
    v0 = 9.0 * Yr / (Xr + 15.0 * Yr + 3.0 * Zr);
    L1 = ((float)(L)) / 1.0;
    if((float)(L1) > k * eps)
    {
        TEMP = (((float)(L1) + 16.0) / 116.0);
        Y = TEMP * TEMP * TEMP;
    }
    else
    {
        Y = ((float)(L1)) / k;
    }
    if((L == 0) && (u == 0) && (v == 0))
    {
        X = 0;
        Y = 0;
        Z = 0;
    }
    else
    {
        ud = (u / (13.0 * L1) + u0);
        vd = (v / (13.0 * L1) + v0);
        X = (ud / vd) * Y * 9.0 / 4.0;
        Z = (Y / vd - ((ud / vd) * Y / 4.0 + 15.0 * Y / 9.0)) * 3.0;
    }

    X = X * 255.0;
    Y = Y * 255.0;
    Z = Z * 255.0;

    r = (int)(3.2404813432005 * X - 1.5371515162713 * Y - 0.49853632616889 * Z + 0.5);
    g = (int)(-0.96925494999657 * X + 1.8759900014899 * Y + 0.041555926558293 * Z + 0.5);
    b = (int)(0.055646639135177 * X - 0.20404133836651 * Y + 1.0573110696453 * Z + 0.5);

    int R = r < 0 ? 0 : r > 255 ? 255 : r;
    int G = g < 0 ? 0 : g > 255 ? 255 : g;
    int B = b < 0 ? 0 : b > 255 ? 255 : b;

    // Write RGB pixel values
    write_imageui(rgb, coord, (uint4)(R, G, B, 255));


}


