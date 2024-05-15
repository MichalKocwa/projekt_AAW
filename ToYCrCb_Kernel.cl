
__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

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

	

	 






	

	




	

	

	
	
