
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.cpp"

struct Pixel {
	unsigned char r, g, b, a;
};

void convertImageToGrayCPU(unsigned char *imageRGBA, int width, int height) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Pixel* pixelPtr = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
			unsigned char pixelVal = (unsigned char)(pixelPtr->r * 0.2126f + pixelPtr->g * 0.7152f + pixelPtr->b * 0.0722f);
			pixelPtr->r = pixelVal;
			pixelPtr->g = pixelVal;
			pixelPtr->b = pixelVal;
			pixelPtr->a = 255;
		}
	}

}

__global__ void convertImageToGrayGPU(unsigned char *imageRGBA) {

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t idx = (y * blockDim.x * gridDim.x + x) * 4;

	Pixel* pixelPtr = (Pixel*)&imageRGBA[idx];
	unsigned char pixelVal = (unsigned char)(pixelPtr->r * 0.2126f + pixelPtr->g * 0.7152f + pixelPtr->b * 0.0722f);
	pixelPtr->r = pixelVal;
	pixelPtr->g = pixelVal;
	pixelPtr->b = pixelVal;
	pixelPtr->a = 255;

}


int main(int argc, char** argv) {
	// Cheking Arguments Count
	if (argc < 2) {
		std::cout << "Usage: 02_ImageToGrayScale <filename>" << std::endl;
		return -1;
	}

	// Opening Image 
	int width, height, compCount;

	std::cout << "Loading File..." << std::endl;
	unsigned char* imageData = stbi_load(argv[1], &width, &height, &compCount, 4);
	
	if (!imageData) {
		std::cout << "Falied to open \"" << argv[1] << "\"" << std::endl;
		return -1;
	}
	
	// Validating image sizes
	if (width % 32 || height % 32) {
		// memory leak
		std::cout << "Wight or Height not divisible by 32." << std::endl;
		return -1;
	}

	// -- Doing Processing using CPU
	//std::cout << "Processing..." << std::endl;
	//convertImageToGrayCPU(imageData, width, height);
	

	// Copy data to GPU
	std::cout << "Coping data to GPU..." << std::endl;
	unsigned char* imageDataGPU = nullptr;
	assert(cudaMalloc(&imageDataGPU, width * height * 4) == cudaSuccess);
	assert(cudaMemcpy(imageDataGPU, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
	
	// -- Doing Processing using GPU
	std::cout << "Running CUDA Kernel..." << std::endl;
	dim3 blockSize(32, 32);
	dim3 gridSize(width / blockSize.x, height / blockSize.y);

	convertImageToGrayGPU<<<gridSize, blockSize>>>(imageDataGPU);


	// Copy data from GPU
	std::cout << "Coping data from GPU..." << std::endl;
	assert(cudaMemcpy(imageData, imageDataGPU, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);

	
	// Output FileName
	std::string fileOutName = argv[1];
	fileOutName = fileOutName.substr(0, fileOutName.find_last_of('.')) + "_gray.png";

	
	// Write Image Back to Disk
	std::cout << "Writing PNG file to disk..." << std::endl;
	stbi_write_png(fileOutName.c_str(), width, height, 4, imageData, 4 * width);
	
	
	// Closing Imgae
	stbi_image_free(imageData);
	cudaFree(imageDataGPU);

	return 0;
}
