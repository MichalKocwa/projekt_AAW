

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

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

	

	 






	

	




	

	

	
	
