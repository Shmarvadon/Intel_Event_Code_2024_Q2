#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sycl/sycl.hpp>
#include <atomic_ref.hpp>


#include <iostream>

/*          GPU codes           */

// Input: 3 channels RGB 8bit,  Output: 1 channel 8 bit grayscale.
void convertToGrayscale_GPU(sycl::buffer<uint8_t,3>& inp, sycl::buffer<uint8_t,2>& oup, sycl::queue q);

// DO NOT USE IT DOES NOT WORK PROPERLY, NEED TO MAKE MODIFICATIONS TO XMX OPERAND SIZE TO PREVENT CRASHING
void convertToGrayscale_GPU_Better(sycl::buffer<uint8_t, 3>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q);

// Input: 1 channel 8 bit grayscale,  Output: 1 channel 8 bit grayscale.
void sobel_GPU(sycl::buffer<uint8_t,2>& inp, sycl::buffer<uint8_t,2>& oup, sycl::queue q);

//Input : 1 channel 8 bit grayscale,	Output: 1 channel 8 bit grayscale.
void floodFill_GPU(sycl::buffer<uint8_t, 2>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q, std::pair<size_t, size_t> centre, std::pair<uint8_t, uint8_t> threashhold);

//Input : 1 channel 8 bit grayscale,	Output: 1 channel 8 bit grayscale.
void floodFill_GPU_Better(sycl::buffer<uint8_t, 2>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q, std::pair<size_t, size_t> centre, std::pair<uint8_t, uint8_t> threashhold);

//Input : 1 channel 8 bit grayscale,	Output: 1 channel 8 bit grayscale.
void floodFill_GPU_Better_Better(sycl::buffer<uint8_t, 2>& inp, sycl::buffer<uint8_t, 2>& oup, sycl::queue q, std::pair<size_t, size_t> centre, std::pair<uint8_t, uint8_t> threashhold);


/*          CPU codes           */

// Input: 3 channels RGB 8bit,  Output: 1 channel 8 bit grayscale.
void convertToGrayscale_CPU(cv::Mat& inp, cv::Mat oup);

// Input: 1 channel 8 bit grayscale,  Output: 1 channel 8 bit grayscale.
void sobel_CPU(cv::Mat& inp, cv::Mat& oup);

//Input : 1 channel 8 bit grayscale,    Output: vector of coordinates for outline.
void floodFill_CPU(cv::Mat& inp, cv::Mat& oup, std::pair<size_t, size_t> centre, std::pair<uint8_t, uint8_t> threashhold);