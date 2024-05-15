

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__kernel void to_LUV(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
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
}

	

	 






	

	




	

	

	
	
