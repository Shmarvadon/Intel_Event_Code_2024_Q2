#include "ImageProcessing.hpp"

#include <chrono>

#define IMG_REZ 600

auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception const& e) {
            std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
        }
    }
};

void test_GPU(cv::Mat img) {
    sycl::default_selector d_selector;
    sycl::queue q(d_selector);


    sycl::buffer<uint8_t, 3> GPU_img_colour((uint8_t*)img.data, { img.size().width, img.size().height, 3 });
    sycl::buffer<uint8_t, 2> GPU_img_grayscale({ img.size().width, img.size().height });
    sycl::buffer<uint8_t, 2> GPU_img_gradient({ img.size().width, img.size().height });
    sycl::buffer<uint8_t, 2> GPU_img_floodfill({ img.size().width, img.size().height });

    // Warm up the kernels to prevent first time compilation overhead from impacting perf numbers.
    convertToGrayscale_GPU(GPU_img_colour, GPU_img_grayscale, q);
    sobel_GPU(GPU_img_grayscale, GPU_img_gradient, q);
    floodFill_GPU_Better_Better(GPU_img_gradient, GPU_img_floodfill, q, { IMG_REZ/2,IMG_REZ/2 }, { 80,254 });


    auto FunctionStartTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    convertToGrayscale_GPU(GPU_img_colour, GPU_img_grayscale, q);
    auto EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::cout << "Convert to grayscale took: " << EndTime - FunctionStartTime << "us\n";
    

    auto StartTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    sobel_GPU(GPU_img_grayscale, GPU_img_gradient, q);
    EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::cout << "Sobel Kernel took: " << EndTime - StartTime << "us\n";


    StartTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    floodFill_GPU_Better_Better(GPU_img_gradient, GPU_img_floodfill, q, { IMG_REZ/2,IMG_REZ/2 }, { 79,245 });
    EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::cout << "Flood Fill took: " << EndTime - StartTime << "us\n";


    std::cout << "GPU took: " << EndTime - FunctionStartTime << "us\n";


    cv::Mat grayscale_img(img.size().width, img.size().height, CV_8U);
    grayscale_img.data = GPU_img_grayscale.get_host_access().get_pointer();

    cv::Mat gradient_img(img.size().width, img.size().height, CV_8U);
    gradient_img.data = GPU_img_gradient.get_host_access().get_pointer();

    cv::Mat floodfill_img(img.size().width, img.size().height, CV_8U);
    floodfill_img.data = GPU_img_floodfill.get_host_access().get_pointer();

    cv::imshow("GPU | Grayscale Image", grayscale_img);
    cv::imshow("GPU | Gradient Image", gradient_img);
    cv::imshow("GPU | Floodfill Image", floodfill_img);
}

void test_CPU(cv::Mat img) {

    cv::Mat CPU_img_grayscale(img.size().width, img.size().height, CV_8U);
    cv::Mat CPU_img_gradient  = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat CPU_img_floodfill = cv::Mat::zeros(img.size(), CV_8U);



    auto FunctionStartTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    convertToGrayscale_CPU(img, CPU_img_grayscale);
    auto EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::cout << "Convert to grayscale took: " << EndTime - FunctionStartTime << "us\n";


    auto StartTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    sobel_CPU(CPU_img_grayscale, CPU_img_gradient);
    EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::cout << "Sobel Kernel took: " << EndTime - StartTime << "us\n";


    StartTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    floodFill_CPU(CPU_img_gradient, CPU_img_floodfill, { IMG_REZ/2,IMG_REZ/2 }, { 79,245 });
    EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::cout << "Flood Fill took: " << EndTime - StartTime << "us\n";


    std::cout << "CPU took: " << EndTime - FunctionStartTime << "us\n";

    cv::imshow("CPU | Grayscale Image", CPU_img_grayscale);
    cv::imshow("CPU | Gradient Image", CPU_img_gradient);
    cv::imshow("CPU | Floodfill Image", CPU_img_floodfill);
}

int main(){

    std::cout << "Hello world.\n";

    auto img = cv::imread("./TestImg.jpg", cv::IMREAD_COLOR);
    cv::resize(img, img, { IMG_REZ, IMG_REZ });

    test_GPU(img);

    test_CPU(img);

    std::cout << (uint8_t)245 / 255 << "\n";
    
    int k = cv::waitKey(0);

    return 0;
}